import argparse
import json
import os
import queue
import socket
import sys
import threading
import time
from typing import Optional

# Add project paths to import streaming_stt and local modules
_THIS_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir))
_SR_SRC = os.path.join(_PROJECT_ROOT, "speech_recognition", "src")
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, _SR_SRC)

from codec import tokenize, pack_tokens, lz4_compress, frame
from common import prepare_log_path
from streaming_stt import TranscriptQueue, FasterWhisperWorker, mic_loop


def sender_worker(host: str, port: int, send_q: "queue.Queue[bytes]", stop: threading.Event):
    print(f"[sender] Connecting to {host}:{port} ...")
    with socket.create_connection((host, port)) as s:
        print("[sender] Connected.")
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        while not stop.is_set():
            try:
                item = send_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                break
            try:
                s.sendall(item)
            except (BrokenPipeError, ConnectionResetError) as e:
                print(f"[sender] Connection closed by peer: {e}")
                break


def bridge_asr_to_textq(text_q: "queue.Queue[Optional[str]]", stop: threading.Event, print_asr: bool = False):
    # Read from TranscriptQueue; forward only final texts
    while not stop.is_set():
        try:
            item = TranscriptQueue.get(timeout=0.2)
        except queue.Empty:
            continue
        t = item.get("type")
        if t == "final":
            txt = item.get("text", "")
            if txt:
                if print_asr:
                    print(f"[asr-final] {txt}")
                text_q.put(txt)
        elif t == "error":
            msg = item.get("message")
            print(f"[asr-error] {msg}")
        elif t == "info" and print_asr:
            print(f"[asr-info] {item.get('message')}")
        # ignore partial/metrics for bridging


def main():
    ap = argparse.ArgumentParser(description="ASR → LZ4 streaming text sender")
    # Network
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9520)
    ap.add_argument("--log-file", default="")
    ap.add_argument("--log-dir", default="log")
    ap.add_argument("--print-asr", action="store_true")
    # ASR / mic options (subset of streaming_stt)
    ap.add_argument("--model", default="tiny")
    ap.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--frame-ms", type=int, default=20, choices=[10, 20, 30])
    ap.add_argument("--device-index", type=int, default=None)
    ap.add_argument("--vad-aggr", type=int, default=2, choices=[0,1,2,3])
    ap.add_argument("--max-silence-ms", type=int, default=800)
    ap.add_argument("--min-speech-ms", type=int, default=200)
    ap.add_argument("--max-chunk-ms", type=int, default=1500)
    ap.add_argument("--energy-threshold", type=float, default=0.006)
    ap.add_argument("--language", default="zh")
    ap.add_argument("--task", default="transcribe", choices=["transcribe","translate"])
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--initial-prompt", default="")
    args = ap.parse_args()

    log_path = prepare_log_path(args.log_file, args.log_dir, prefix="tx_asr")

    # Start ASR worker
    worker = FasterWhisperWorker(
        model_size=args.model,
        device=args.device,
        beam_size=args.beam_size,
        language=args.language,
        task=args.task,
        temperature=args.temperature,
        initial_prompt=args.initial_prompt,
    )
    worker.start()

    # Start mic capture (in a separate daemon thread)
    def _mic_thread():
        try:
            mic_loop(sample_rate=args.sample_rate, frame_ms=args.frame_ms, device_index=args.device_index,
                     vad_aggr=args.vad_aggr, max_silence_ms=args.max_silence_ms,
                     min_speech_ms=args.min_speech_ms, max_chunk_ms=args.max_chunk_ms,
                     energy_threshold=args.energy_threshold)
        except KeyboardInterrupt:
            pass

    mic_t = threading.Thread(target=_mic_thread, daemon=True)
    mic_t.start()

    # Queues and network thread
    text_q: "queue.Queue[Optional[str]]" = queue.Queue()
    send_q: "queue.Queue[Optional[bytes]]" = queue.Queue(maxsize=64)
    stop = threading.Event()

    net_t = threading.Thread(target=sender_worker, args=(args.host, args.port, send_q, stop), daemon=True)
    net_t.start()

    bridge_t = threading.Thread(target=bridge_asr_to_textq, args=(text_q, stop, args.print_asr), daemon=True)
    bridge_t.start()

    # Metrics accumulators (same as sender)
    total_lines = 0
    sum_raw = 0
    sum_packed = 0
    sum_comp = 0
    sum_framed = 0
    sum_tokenize_ms = 0.0
    sum_pack_ms = 0.0
    sum_compress_ms = 0.0
    sum_frame_ms = 0.0
    sum_total_ms = 0.0
    seq = 0

    try:
        while True:
            try:
                line = text_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if line is None:
                send_q.put(None)
                break
            # encode → tokens → pack → compress → frame
            import time as _t
            t0 = _t.perf_counter()
            raw_b = line.encode("utf-8")
            t1 = _t.perf_counter()
            tokens = tokenize(line)
            t2 = _t.perf_counter()
            payload = pack_tokens(tokens)
            t3 = _t.perf_counter()
            comp = lz4_compress(payload)
            t4 = _t.perf_counter()
            framed = frame(comp)
            t5 = _t.perf_counter()

            tok_ms = (t2 - t1) * 1000.0
            pack_ms = (t3 - t2) * 1000.0
            comp_ms = (t4 - t3) * 1000.0
            frame_ms = (t5 - t4) * 1000.0
            total_ms = (t5 - t0) * 1000.0

            raw_len = len(raw_b)
            packed_len = len(payload)
            comp_len = len(comp)
            framed_len = len(framed)

            cr_comp_vs_raw = (comp_len / raw_len) if raw_len else None
            cr_comp_vs_packed = (comp_len / packed_len) if packed_len else None
            cr_framed_vs_raw = (framed_len / raw_len) if raw_len else None

            total_lines += 1
            sum_raw += raw_len
            sum_packed += packed_len
            sum_comp += comp_len
            sum_framed += framed_len
            sum_tokenize_ms += tok_ms
            sum_pack_ms += pack_ms
            sum_compress_ms += comp_ms
            sum_frame_ms += frame_ms
            sum_total_ms += total_ms
            seq += 1

            send_q.put(framed)
            if log_path:
                try:
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "type": "tx",
                            "ts": time.time(),
                            "seq": seq,
                            "len_raw": raw_len,
                            "len_tokens": len(tokens),
                            "len_payload": packed_len,
                            "len_comp": comp_len,
                            "len_framed": framed_len,
                            "cr_comp_vs_raw": cr_comp_vs_raw,
                            "cr_comp_vs_packed": cr_comp_vs_packed,
                            "cr_framed_vs_raw": cr_framed_vs_raw,
                            "enc_ms": {
                                "tokenize": tok_ms,
                                "pack": pack_ms,
                                "compress": comp_ms,
                                "frame": frame_ms,
                                "total": total_ms,
                            },
                            "text": line,
                        }, ensure_ascii=False) + "\n")
                except Exception:
                    pass
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        send_q.put(None)
        net_t.join(timeout=2.0)
        if log_path and total_lines > 0:
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "type": "tx-summary",
                        "ts": time.time(),
                        "lines": total_lines,
                        "sum_len": {
                            "raw": sum_raw,
                            "packed": sum_packed,
                            "comp": sum_comp,
                            "framed": sum_framed,
                        },
                        "avg_cr": {
                            "comp_vs_raw": (sum_comp / sum_raw) if sum_raw else None,
                            "comp_vs_packed": (sum_comp / sum_packed) if sum_packed else None,
                            "framed_vs_raw": (sum_framed / sum_raw) if sum_raw else None,
                        },
                        "avg_enc_ms": {
                            "tokenize": sum_tokenize_ms / total_lines,
                            "pack": sum_pack_ms / total_lines,
                            "compress": sum_compress_ms / total_lines,
                            "frame": sum_frame_ms / total_lines,
                            "total": sum_total_ms / total_lines,
                        },
                    }, ensure_ascii=False) + "\n")
            except Exception:
                pass
        worker.stop()


if __name__ == "__main__":
    main()
