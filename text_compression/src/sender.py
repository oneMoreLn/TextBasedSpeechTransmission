import argparse
import json
import queue
import socket
import threading
import time
from typing import Optional
import os
import sys

# Allow importing sibling modules when run as a script
sys.path.insert(0, os.path.dirname(__file__))

from codec import tokenize, pack_tokens, lz4_compress, frame
from common import prepare_log_path


def producer_lines(file_path: str, q: "queue.Queue[str]"):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            q.put(line)
    # Signal end
    q.put(None)  # type: ignore


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


def main():
    ap = argparse.ArgumentParser(description="Streaming text sender with LZ4 over TCP")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9500)
    ap.add_argument("--input-file", required=True)
    ap.add_argument("--log-file", default="")
    ap.add_argument("--log-dir", default="log")
    args = ap.parse_args()

    log_path = prepare_log_path(args.log_file, args.log_dir, prefix="tx")

    text_q: "queue.Queue[Optional[str]]" = queue.Queue()
    send_q: "queue.Queue[Optional[bytes]]" = queue.Queue(maxsize=64)

    # Metrics accumulators
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

    prod_t = threading.Thread(target=producer_lines, args=(args.input_file, text_q), daemon=True)
    prod_t.start()

    stop = threading.Event()
    net_t = threading.Thread(target=sender_worker, args=(args.host, args.port, send_q, stop), daemon=True)
    net_t.start()

    try:
        while True:
            line = text_q.get()
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

            # Ratios
            cr_comp_vs_raw = (comp_len / raw_len) if raw_len else None
            cr_comp_vs_packed = (comp_len / packed_len) if packed_len else None
            cr_framed_vs_raw = (framed_len / raw_len) if raw_len else None

            # Accumulate
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
    finally:
        stop.set()
        net_t.join(timeout=2.0)
        # Write summary metrics
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


if __name__ == "__main__":
    main()
