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

from codec import detokenize, unpack_tokens, lz4_decompress, deframe
from common import prepare_log_path


def recv_worker(host: str, port: int, out_q: "queue.Queue[bytes]", stop: threading.Event):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    try:
        print(f"[receiver] Listening on {host}:{port} ...")
        conn, addr = srv.accept()
        print(f"[receiver] Accepted connection from {addr}")
        with conn:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            buf = bytearray()
            while not stop.is_set():
                data = conn.recv(8192)
                if not data:
                    break
                buf.extend(data)
                for payload in deframe(buf):
                    out_q.put(payload)
    finally:
        srv.close()


def main():
    ap = argparse.ArgumentParser(description="Streaming text receiver with LZ4 over TCP")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9500)
    ap.add_argument("--log-file", default="")
    ap.add_argument("--log-dir", default="log")
    args = ap.parse_args()

    log_path = prepare_log_path(args.log_file, args.log_dir, prefix="rx")

    out_q: "queue.Queue[bytes]" = queue.Queue()
    stop = threading.Event()
    t = threading.Thread(target=recv_worker, args=(args.host, args.port, out_q, stop), daemon=True)
    t.start()

    # Metrics accumulators
    total_lines = 0
    sum_dec_ms = 0.0
    sum_unpack_ms = 0.0
    sum_detok_ms = 0.0
    sum_total_ms = 0.0
    sum_len_payload = 0
    sum_len_dec = 0
    seq = 0

    try:
        while True:
            try:
                payload = out_q.get(timeout=0.5)
            except queue.Empty:
                if not t.is_alive():
                    break
                continue
            try:
                import time as _t
                r0 = _t.perf_counter()
                dec = lz4_decompress(payload)
                r1 = _t.perf_counter()
                tokens = unpack_tokens(dec)
                r2 = _t.perf_counter()
                text = detokenize(tokens)
                r3 = _t.perf_counter()

                dec_ms = (r1 - r0) * 1000.0
                unpack_ms = (r2 - r1) * 1000.0
                detok_ms = (r3 - r2) * 1000.0
                total_ms = (r3 - r0) * 1000.0

                seq += 1
                total_lines += 1
                sum_dec_ms += dec_ms
                sum_unpack_ms += unpack_ms
                sum_detok_ms += detok_ms
                sum_total_ms += total_ms
                sum_len_payload += len(payload)
                sum_len_dec += len(dec)
                print(text)
                if log_path:
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "type": "rx",
                            "ts": time.time(),
                            "seq": seq,
                            "len_tokens": len(tokens),
                            "len_dec": len(dec),
                            "len_payload": len(payload),
                            "cr_payload_vs_dec": (len(payload) / len(dec)) if len(dec) else None,
                            "dec_ms": {
                                "decompress": dec_ms,
                                "unpack": unpack_ms,
                                "detokenize": detok_ms,
                                "total": total_ms,
                            },
                            "text": text,
                        }, ensure_ascii=False) + "\n")
            except Exception as e:
                if log_path:
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"type": "error", "ts": time.time(), "err": str(e)}) + "\n")
    finally:
        stop.set()
        t.join(timeout=2.0)
        # Summary metrics
        if log_path and total_lines > 0:
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "type": "rx-summary",
                        "ts": time.time(),
                        "lines": total_lines,
                        "avg_dec_ms": {
                            "decompress": sum_dec_ms / total_lines,
                            "unpack": sum_unpack_ms / total_lines,
                            "detokenize": sum_detok_ms / total_lines,
                            "total": sum_total_ms / total_lines,
                        },
                        "avg_cr": {
                            "payload_vs_dec": (sum_len_payload / sum_len_dec) if sum_len_dec else None,
                        },
                        "sum_len": {
                            "payload": sum_len_payload,
                            "dec": sum_len_dec,
                        },
                    }, ensure_ascii=False) + "\n")
            except Exception:
                pass


if __name__ == "__main__":
    main()
