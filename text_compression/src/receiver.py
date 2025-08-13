import argparse
import json
import queue
import socket
import threading
import time
from typing import Optional
import os
import sys
import base64

# Allow importing sibling modules when run as a script
sys.path.insert(0, os.path.dirname(__file__))

from codec import detokenize, unpack_tokens, lz4_decompress, deframe, frame, is_lz4_frame, is_zstd_frame, zstd_decompress, rs_fec_decode
from common import prepare_log_path


def recv_worker(host: str, port: int, out_q: "queue.Queue[bytes]", stop: threading.Event):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    srv.settimeout(1.0)
    try:
        print(f"[receiver] Listening on {host}:{port} ...")
        while not stop.is_set():
            try:
                conn, addr = srv.accept()
            except socket.timeout:
                continue
            print(f"[receiver] Accepted connection from {addr}")
            with conn:
                try:
                    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                except Exception:
                    pass
                conn.settimeout(1.0)
                buf = bytearray()
                while not stop.is_set():
                    try:
                        data = conn.recv(8192)
                    except socket.timeout:
                        continue
                    if not data:
                        print("[receiver] Peer closed. Back to listening...")
                        break
                    buf.extend(data)
                    for payload in deframe(buf):
                        out_q.put(payload)
    finally:
        try:
            srv.close()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(description="Streaming text receiver with LZ4 over TCP")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9500)
    ap.add_argument("--log-file", default="")
    ap.add_argument("--log-dir", default="log")
    ap.add_argument("--expect", choices=["auto", "lz4", "zstd", "raw"], default="auto", help="How to interpret payload")
    ap.add_argument("--zstd-dict", default="", help="Path to zstd dictionary file (optional)")
    ap.add_argument("--fec-nsym", type=int, default=0, help="FEC parity symbols used by sender (Reed-Solomon). 0 to disable.")
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
    # RX throughput accumulators (bits and seconds)
    avg_rx_bits = 0.0
    avg_rx_time = 0.0

    # Optional zstd dict
    zstd_dict_bytes = None
    if args.zstd_dict:
        try:
            with open(args.zstd_dict, "rb") as df:
                zstd_dict_bytes = df.read()
        except Exception as e:
            print(f"[receiver] failed to read zstd dict: {e}")

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
                rx_t0 = _t.perf_counter()
                r0 = _t.perf_counter()
                # Optional FEC decode first
                fec_ok = None
                if args.fec_nsym and args.fec_nsym > 0 and len(payload) > 0:
                    payload, fec_ok = rs_fec_decode(payload, args.fec_nsym)
                # Decide whether to decompress
                used_lz4 = False
                if args.expect == "lz4" or (args.expect == "auto" and is_lz4_frame(payload)):
                    dec = lz4_decompress(payload)
                    used_lz4 = True
                elif args.expect == "zstd" or (args.expect == "auto" and is_zstd_frame(payload)):
                    dec = zstd_decompress(payload, dict_bytes=zstd_dict_bytes)
                    used_lz4 = False
                else:
                    dec = payload
                r1 = _t.perf_counter()
                # Try varint-unpack first; if fails, fallback to UTF-8 (strict→replace)
                tokens = []
                text = None
                decode_mode = None
                decode_repl = 0
                def _utf8_best_effort(b: bytes):
                    try:
                        s = b.decode("utf-8", errors="strict")
                        return s, "strict", 0
                    except UnicodeDecodeError:
                        s = b.decode("utf-8", errors="replace")
                        return s, "replace", s.count("\ufffd")
                try:
                    tokens = unpack_tokens(dec)
                    r2 = _t.perf_counter()
                    if len(tokens) == 0 and len(dec) > 0:
                        # Likely not a varint-packed payload; treat as UTF-8 with best-effort fallback
                        text, decode_mode, decode_repl = _utf8_best_effort(dec)
                    else:
                        text = detokenize(tokens)
                        decode_mode, decode_repl = "tokens", 0
                except Exception:
                    r2 = _t.perf_counter()
                    text, decode_mode, decode_repl = _utf8_best_effort(dec)
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
                            "payload_b64": base64.b64encode(payload).decode("ascii"),
                            "framed_b64": base64.b64encode(frame(payload)).decode("ascii"),
                            "dec_ms": {
                                "decompress": dec_ms,
                                "unpack": unpack_ms,
                                "detokenize": detok_ms,
                                "total": total_ms,
                            },
                            "used_lz4": used_lz4,
                            "fec_nsym": args.fec_nsym,
                            "fec_ok": fec_ok,
                            "expect": args.expect,
                            "text": text,
                            "decode": {
                                "mode": decode_mode,
                                "replacements": decode_repl,
                            },
                        }, ensure_ascii=False) + "\n")
                # RX throughput log (instantaneous and rolling)
                if log_path:
                    # instantaneous rx_bps based on this payload size and processing duration
                    rx_ms = (r3 - rx_t0) * 1000.0
                    rx_bps = (len(payload) * 8.0) / max(1e-6, (r3 - rx_t0))
                    # update rolling averages
                    avg_rx_bits += len(payload) * 8.0
                    avg_rx_time += (r3 - rx_t0)
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "type": "rx-net",
                            "ts": time.time(),
                            "seq": seq,
                            "bytes": len(payload),
                            "rx_ms": rx_ms,
                            "rx_bps": rx_bps,
                            "avg_rx_bps": (avg_rx_bits / avg_rx_time) if avg_rx_time > 0 else None,
                        }) + "\n")
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
                        "avg_rx_bps": (avg_rx_bits / avg_rx_time) if avg_rx_time > 0 else None,
                    }, ensure_ascii=False) + "\n")
            except Exception:
                pass


if __name__ == "__main__":
    main()
