import argparse
import json
import os
import queue
import socket
import sys
import threading
import time
from typing import Optional
import base64

# Add project paths to import streaming_stt and local modules
_THIS_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir))
_SR_SRC = os.path.join(_PROJECT_ROOT, "speech_recognition", "src")
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, _SR_SRC)

from codec import tokenize, pack_tokens, lz4_compress, frame, zstd_compress
from common import prepare_log_path
from streaming_stt import (
    TranscriptQueue,
    FasterWhisperWorker,
    mic_loop,
    VADStreamer,
    audio_chunks,
    float32_to_pcm16,
)


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
    # Packing & compression
    ap.add_argument("--pack-mode", choices=["varint", "raw"], default="raw", help="Payload packing: varint tokens or raw UTF-8")
    ap.add_argument("--codec", choices=["raw", "lz4", "zstd"], default="zstd")
    ap.add_argument("--compress-threshold", type=int, default=256, help="Min payload bytes to apply compression; below sends raw")
    ap.add_argument("--zstd-level", type=int, default=3)
    ap.add_argument("--zstd-dict", default="", help="Path to zstd dictionary file (optional)")
    ap.add_argument("--batch-ms", type=int, default=150)
    ap.add_argument("--batch-bytes", type=int, default=2048)
    # Source selection: mic (ASR) | audio (offline audio file for ASR) | file (plain text lines)
    ap.add_argument("--source", choices=["mic", "audio", "file"], default="mic", help="Input source: microphone ASR, offline audio file ASR, or a text file")
    ap.add_argument("--input-file", default="", help="Text file path when --source=file")
    ap.add_argument("--audio-file", default="", help="Audio file path when --source=audio (wav/flac etc.)")
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

    # Queues and network thread
    text_q: "queue.Queue[Optional[str]]" = queue.Queue()
    send_q: "queue.Queue[Optional[bytes]]" = queue.Queue(maxsize=64)
    stop = threading.Event()

    net_t = threading.Thread(target=sender_worker, args=(args.host, args.port, send_q, stop), daemon=True)
    net_t.start()

    worker = None
    mic_t = None
    bridge_t = None

    if args.source == "mic":
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

        bridge_t = threading.Thread(target=bridge_asr_to_textq, args=(text_q, stop, args.print_asr), daemon=True)
        bridge_t.start()
        print("[mode] Using ASR microphone source")
    else:
        if args.source == "file":
            # File mode: read lines into text_q
            if not args.input_file:
                print("[error] --input-file is required when --source=file")
                return
            def _producer_lines():
                try:
                    with open(args.input_file, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.rstrip("\n")
                            if line:
                                text_q.put(line)
                except Exception as e:
                    print(f"[file-source] read error: {e}")
                finally:
                    # signal end
                    text_q.put(None)
            prod_t = threading.Thread(target=_producer_lines, daemon=True)
            prod_t.start()
            print(f"[mode] Using text file source: {args.input_file}")
        else:
            # Offline audio file → VAD chunks → audio_chunks → ASR → TranscriptQueue
            if not args.audio_file:
                print("[error] --audio-file is required when --source=audio")
                return
            def _audio_file_feeder():
                try:
                    import soundfile as sf
                    import numpy as np
                    data, sr = sf.read(args.audio_file, always_2d=False)
                    if getattr(data, 'ndim', 1) > 1:
                        data = np.mean(data, axis=1)
                    if sr != args.sample_rate:
                        import math
                        ratio = args.sample_rate / sr
                        new_len = int(math.floor(len(data) * ratio))
                        x_old = np.linspace(0, 1, num=len(data), endpoint=False)
                        x_new = np.linspace(0, 1, num=new_len, endpoint=False)
                        data = np.interp(x_new, x_old, data).astype(np.float32)
                    else:
                        data = data.astype(np.float32)
                    pcm = float32_to_pcm16(data)
                    vad = VADStreamer(sample_rate=args.sample_rate, frame_ms=args.frame_ms,
                                      vad_aggressiveness=args.vad_aggr,
                                      max_silence_ms=args.max_silence_ms,
                                      min_speech_ms=args.min_speech_ms,
                                      max_chunk_ms=args.max_chunk_ms,
                                      energy_threshold=args.energy_threshold)
                    frame_bytes = int(args.sample_rate * args.frame_ms / 1000) * 2
                    import itertools
                    _cid = itertools.count(1)
                    for i in range(0, len(pcm), frame_bytes):
                        frame = pcm[i : i + frame_bytes]
                        if len(frame) < frame_bytes:
                            frame = frame + b"\x00" * (frame_bytes - len(frame))
                        chunk = vad.process_frame(frame)
                        if chunk is not None:
                            try:
                                cid = next(_cid)
                                dur_s = len(chunk) / (2 * args.sample_rate)
                                audio_chunks.put({"pcm": chunk, "id": cid, "t0": time.monotonic(), "dur_s": dur_s, "sr": args.sample_rate}, timeout=1.0)
                            except queue.Full:
                                TranscriptQueue.put({"type": "info", "message": "Dropping chunk due to backlog (audio-file)"})
                    tail = vad.flush()
                    if tail:
                        try:
                            cid = next(_cid)
                            dur_s = len(tail) / (2 * args.sample_rate)
                            audio_chunks.put({"pcm": tail, "id": cid, "t0": time.monotonic(), "dur_s": dur_s, "sr": args.sample_rate}, timeout=1.0)
                        except queue.Full:
                            TranscriptQueue.put({"type": "info", "message": "Dropping tail due to backlog (audio-file)"})
                    TranscriptQueue.put({"type": "info", "message": f"Audio file feed done: {args.audio_file}"})
                except Exception as e:
                    TranscriptQueue.put({"type": "error", "message": f"Audio feeder error: {e}"})
            # Start ASR worker for audio-file mode
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
            feeder_t = threading.Thread(target=_audio_file_feeder, daemon=True)
            feeder_t.start()
            bridge_t = threading.Thread(target=bridge_asr_to_textq, args=(text_q, stop, args.print_asr), daemon=True)
            bridge_t.start()
            print(f"[mode] Using offline audio source: {args.audio_file}")

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

    # Optional zstd dict
    zstd_dict_bytes = None
    if args.zstd_dict:
        try:
            with open(args.zstd_dict, "rb") as df:
                zstd_dict_bytes = df.read()
        except Exception as e:
            print(f"[tx_asr] failed to read zstd dict: {e}")

    def encode_one(text: str):
        import time as _t
        t0 = _t.perf_counter()
        raw_b = text.encode("utf-8")
        t1 = _t.perf_counter()
        if args.pack_mode == "raw":
            tokens = []
            payload = raw_b
            t2 = _t.perf_counter()
            t3 = t2
        else:
            tokens = tokenize(text)
            t2 = _t.perf_counter()
            payload = pack_tokens(tokens)
            t3 = _t.perf_counter()

        if args.compress_threshold > 0 and len(payload) < args.compress_threshold:
            comp = payload
            compressed = False
            codec_used = "raw"
        else:
            if args.codec == "lz4":
                comp = lz4_compress(payload)
                codec_used = "lz4"
            elif args.codec == "zstd":
                comp = zstd_compress(payload, level=args.zstd_level, dict_bytes=zstd_dict_bytes)
                codec_used = "zstd"
            else:
                comp = payload
                codec_used = "raw"
            compressed = codec_used != "raw"
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

        return {
            "framed": framed,
            "log": {
                "len_raw": raw_len,
                "len_tokens": len(tokens),
                "len_payload": packed_len,
                "len_comp": comp_len,
                "len_framed": framed_len,
                "cr_comp_vs_raw": (comp_len / raw_len) if raw_len else None,
                "cr_comp_vs_packed": (comp_len / packed_len) if packed_len else None,
                "cr_framed_vs_raw": (framed_len / raw_len) if raw_len else None,
                "payload_b64": base64.b64encode(payload).decode("ascii"),
                "framed_b64": base64.b64encode(framed).decode("ascii"),
                "enc_ms": {
                    "tokenize": tok_ms,
                    "pack": pack_ms,
                    "compress": comp_ms,
                    "frame": frame_ms,
                    "total": total_ms,
                },
                "pack_mode": args.pack_mode,
                "compressed": compressed,
                "compress_threshold": args.compress_threshold,
                "codec": codec_used,
                "zstd_level": args.zstd_level,
                "zstd_dict": bool(zstd_dict_bytes),
            }
        }

    try:
        while True:
            try:
                line = text_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if line is None:
                send_q.put(None)
                break
            # micro-batch: accumulate lines until window triggers
            # We reuse the batching code from sender: for ASR，通常一条条 final 文本到来，这里按时间窗/字节窗合并
            batch_buf = [line]
            batch_bytes = len(line.encode("utf-8"))
            start = time.monotonic()
            while (time.monotonic() - start) * 1000.0 < args.batch_ms and batch_bytes < args.batch_bytes:
                try:
                    nxt = text_q.get(timeout=0.01)
                    if nxt is None:
                        text_q.put(None)
                        break
                    batch_buf.append(nxt)
                    batch_bytes += len(nxt.encode("utf-8")) + 1
                except queue.Empty:
                    pass
            text = "\n".join(batch_buf)
            rec = encode_one(text)
            send_q.put(rec["framed"])  # type: ignore
            # accumulate metrics and log once per batch
            total_lines += len(batch_buf)
            sum_raw += rec["log"]["len_raw"]
            sum_packed += rec["log"]["len_payload"]
            sum_comp += rec["log"]["len_comp"]
            sum_framed += rec["log"]["len_framed"]
            sum_tokenize_ms += rec["log"]["enc_ms"]["tokenize"]
            sum_pack_ms += rec["log"]["enc_ms"]["pack"]
            sum_compress_ms += rec["log"]["enc_ms"]["compress"]
            sum_frame_ms += rec["log"]["enc_ms"]["frame"]
            sum_total_ms += rec["log"]["enc_ms"]["total"]
            seq += 1
            if log_path:
                try:
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "type": "tx",
                            "ts": time.time(),
                            "seq": seq,
                            **rec["log"],
                            "text": text,
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
        if worker is not None:
            try:
                worker.stop()
            except Exception:
                pass


if __name__ == "__main__":
    main()
