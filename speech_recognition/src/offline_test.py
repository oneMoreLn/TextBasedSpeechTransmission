import argparse
import os
import queue
import sys
import threading
import time
import json
from typing import Optional, Any, cast

import numpy as np
import soundfile as sf

# Import from our streaming module
from streaming_stt import (
    TranscriptQueue,
    VADStreamer,
    FasterWhisperWorker,
    audio_chunks,
    float32_to_pcm16,
)


def queue_printer(stop_event: threading.Event, log_path: str = ""):
    rtf_hist = []
    e2e_hist = []
    pair: dict = {}
    while not stop_event.is_set():
        try:
            item = TranscriptQueue.get(timeout=0.2)
        except queue.Empty:
            continue
        t = item.get("type")
        if t in {"partial", "final"}:
            print(f"[{t}] {item.get('text','')}")
            if t == "final" and log_path:
                cid = item.get("chunk_id")
                if cid is not None:
                    rec = pair.get(cid, {})
                    rec["text"] = item.get("text", "")
                    pair[cid] = rec
                    # If metrics already present, write
                    if rec.get("rtf") is not None:
                        out = {
                            "ts": time.time(),
                            "chunk_id": cid,
                            "text": rec.get("text", ""),
                            "rtf": rec.get("rtf"),
                            "e2e_s": rec.get("e2e_s"),
                            "dur_s": rec.get("dur_s"),
                            "proc_s": rec.get("proc_s"),
                            "model": rec.get("model"),
                            "device": rec.get("device"),
                        }
                        try:
                            with open(log_path, "a", encoding="utf-8") as f:
                                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                        except Exception:
                            pass
                        pair.pop(cid, None)
        elif t == "metrics":
            rtf = item.get("rtf")
            e2e = item.get("e2e_s")
            if isinstance(rtf, (int, float)):
                rtf_hist.append(float(rtf))
                if len(rtf_hist) > 10:
                    rtf_hist.pop(0)
            if isinstance(e2e, (int, float)):
                e2e_hist.append(float(e2e))
                if len(e2e_hist) > 10:
                    e2e_hist.pop(0)
            rtf_str = f"{rtf:.3f}x" if isinstance(rtf, (int, float)) else "n/a"
            e2e_str = f"{e2e:.3f}s" if isinstance(e2e, (int, float)) else "n/a"
            print(f"[metrics] chunk={item.get('chunk_id')} dur={item.get('dur_s'):.3f}s proc={item.get('proc_s'):.3f}s rtf={rtf_str} e2e={e2e_str}")
            if rtf_hist or e2e_hist:
                avg_rtf_str = f"{(sum(rtf_hist)/len(rtf_hist)):.3f}x" if rtf_hist else "n/a"
                avg_e2e_str = f"{(sum(e2e_hist)/len(e2e_hist)):.3f}s" if e2e_hist else "n/a"
                n = max(len(rtf_hist), len(e2e_hist))
                print(f"[metrics-avg10] rtf={avg_rtf_str} e2e={avg_e2e_str} n={n}")
            # logging: store metrics and write if text ready
            if log_path:
                cid = item.get("chunk_id")
                if cid is not None:
                    rec = pair.get(cid, {})
                    rec.update({
                        "rtf": rtf,
                        "e2e_s": e2e,
                        "dur_s": item.get("dur_s"),
                        "proc_s": item.get("proc_s"),
                        "model": item.get("model"),
                        "device": item.get("device"),
                    })
                    pair[cid] = rec
                    if rec.get("text") is not None and rec.get("rtf") is not None:
                        out = {
                            "ts": time.time(),
                            "chunk_id": cid,
                            "text": rec.get("text", ""),
                            "rtf": rec.get("rtf"),
                            "e2e_s": rec.get("e2e_s"),
                            "dur_s": rec.get("dur_s"),
                            "proc_s": rec.get("proc_s"),
                            "model": rec.get("model"),
                            "device": rec.get("device"),
                        }
                        try:
                            with open(log_path, "a", encoding="utf-8") as f:
                                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                        except Exception:
                            pass
                        pair.pop(cid, None)
        else:
            print(f"[{t}] {item}")


def main():
    parser = argparse.ArgumentParser(description="Offline test using WAV file")
    parser.add_argument("wav", help="Path to WAV file")
    parser.add_argument("--model", default="base")
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--frame-ms", type=int, default=20, choices=[10, 20, 30])
    parser.add_argument("--print-queue", action="store_true")
    parser.add_argument("--log-file", default="", help="Append JSON lines with {chunk_id, text, rtf, e2e_s, dur_s, proc_s, model, device, ts}")
    parser.add_argument("--log-dir", default="log", help="Directory to save logs when --log-file is empty (filename auto-generated)")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--task", default="transcribe", choices=["transcribe","translate"])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--initial-prompt", default="")
    args = parser.parse_args()

    # Prepare logging path: if no --log-file, auto-generate under --log-dir
    if not args.log_file:
        try:
            os.makedirs(args.log_dir, exist_ok=True)
            auto_name = f"stt_offline_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
            args.log_file = os.path.join(args.log_dir, auto_name)
        except Exception:
            args.log_file = ""

    # Write a run header with args if logging is enabled
    if args.log_file:
        try:
            with open(args.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "type": "run",
                    "ts": time.time(),
                    "cwd": os.getcwd(),
                    "argv": sys.argv,
                    "args": vars(args),
                }, ensure_ascii=False) + "\n")
        except Exception:
            pass

    # Start worker
    # Force CPU when auto to avoid cuDNN crashes on misconfigured systems
    import os as _os
    dev = args.device
    if dev == "auto":
        dev = "cpu"
    if dev == "cpu":
        _os.environ.setdefault("CT2_FORCE_CPU", "1")

    # Create worker (avoid named kwargs that may confuse type-checkers across files)
    worker = FasterWhisperWorker(model_size=args.model, device=dev, beam_size=args.beam_size)
    # Attach decoding-related attributes if available
    for name, value in (
        ("language", args.language),
        ("task", args.task),
        ("temperature", args.temperature),
        ("initial_prompt", args.initial_prompt),
    ):
        try:
            setattr(worker, name, value)
        except Exception:
            pass
    worker.start()

    # Optional consumer
    stop = threading.Event()
    printer_thread: Optional[threading.Thread] = None
    if args.print_queue or args.log_file:
        printer_thread = threading.Thread(target=queue_printer, args=(stop, args.log_file), daemon=True)
        printer_thread.start()

    # Read file
    data, sr = sf.read(args.wav, always_2d=False)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if sr != args.sample_rate:
        # Simple resampler via soundfile is not available; fallback to numpy rate change (naive)
        # Better to use resampy or librosa if higher quality is needed.
        import math
        ratio = args.sample_rate / sr
        new_len = int(math.floor(len(data) * ratio))
        x_old = np.linspace(0, 1, num=len(data), endpoint=False)
        x_new = np.linspace(0, 1, num=new_len, endpoint=False)
        data = np.interp(x_new, x_old, data).astype(np.float32)
    else:
        data = data.astype(np.float32)

    # Convert to PCM16 bytes
    pcm = float32_to_pcm16(data)

    # Feed through VAD into audio_chunks
    vad = VADStreamer(sample_rate=args.sample_rate, frame_ms=args.frame_ms)
    frame_bytes = int(args.sample_rate * args.frame_ms / 1000) * 2

    import itertools
    _cid = itertools.count(1)
    queued_chunks = 0
    queued_dur = 0.0
    for i in range(0, len(pcm), frame_bytes):
        frame = pcm[i : i + frame_bytes]
        if len(frame) < frame_bytes:
            frame = frame + b"\x00" * (frame_bytes - len(frame))
        chunk = vad.process_frame(frame)
        if chunk is not None:
            try:
                cid = next(_cid)
                dur_s = len(chunk) / (2 * args.sample_rate)
                audio_q: "queue.Queue[Any]" = cast("queue.Queue[Any]", audio_chunks)
                audio_q.put({"pcm": chunk, "id": cid, "t0": time.monotonic(), "dur_s": dur_s, "sr": args.sample_rate}, timeout=1.0)
                queued_chunks += 1
                queued_dur += dur_s
            except queue.Full:
                TranscriptQueue.put({"type": "info", "message": "Dropping chunk due to backlog (offline)"})

    tail = vad.flush()
    if tail:
        try:
            cid = next(_cid)
            dur_s = len(tail) / (2 * args.sample_rate)
            audio_q: "queue.Queue[Any]" = cast("queue.Queue[Any]", audio_chunks)
            audio_q.put({"pcm": tail, "id": cid, "t0": time.monotonic(), "dur_s": dur_s, "sr": args.sample_rate}, timeout=1.0)
            queued_chunks += 1
            queued_dur += dur_s
        except queue.Full:
            TranscriptQueue.put({"type": "info", "message": "Dropping tail due to backlog (offline)"})

    # Notify how much work was queued (helps when model loading is slow)
    try:
        TranscriptQueue.put({
            "type": "info",
            "message": f"Queued {queued_chunks} chunks, ~{queued_dur:.2f}s audio; waiting for transcription..."
        })
    except Exception:
        pass

    # Wait for processing: adapt to audio length and device speed
    # Heuristic: longer wait for CPU and larger models
    total_audio_s = float(len(data)) / float(args.sample_rate)
    cpu_slow_factor = 4.0 if dev == "cpu" else 2.0
    model_boost = 1.5 if str(args.model).lower().startswith(("medium", "large")) else 1.0
    wait_secs = max(5.0, min(300.0, total_audio_s * cpu_slow_factor * model_boost))

    deadline = time.time() + wait_secs
    # Poll until queue drains or deadline, then linger a short tail to flush last outputs
    while time.time() < deadline:
        try:
            empty = cast("queue.Queue[Any]", audio_chunks).empty()
        except Exception:
            empty = True
        if empty:
            break
        time.sleep(0.2)
    # Linger tail to allow final metrics/text to be emitted
    time.sleep(1.0)

    # Shutdown
    if printer_thread is not None:
        stop.set()
        printer_thread.join(timeout=1.0)

    worker.stop()
    worker.join(timeout=3.0)


if __name__ == "__main__":
    main()
