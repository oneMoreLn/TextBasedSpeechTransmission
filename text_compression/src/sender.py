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

from codec import tokenize, pack_tokens, lz4_compress, frame, zstd_compress, rs_fec_encode
from common import prepare_log_path


def producer_lines(file_path: str, q: "queue.Queue[str]"):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            q.put(line)
    # Signal end
    q.put(None)  # type: ignore


def sender_worker(host: str, port: int, send_q: "queue.Queue", stop: threading.Event, net_log_q: "queue.Queue"):
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
                # item expected as (seq, framed_bytes)
                seq, data = item  # type: ignore
                import time as _t
                t0 = _t.perf_counter()
                s.sendall(data)
                t1 = _t.perf_counter()
                send_ms = (t1 - t0) * 1000.0
                tx_bps = (len(data) * 8.0) / max(1e-6, (t1 - t0))
                net_log_q.put({
                    "seq": int(seq),
                    "bytes": len(data),
                    "send_ms": send_ms,
                    "tx_bps": tx_bps,
                    "ts": time.time(),
                })
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
    ap.add_argument("--pack-mode", choices=["varint", "raw"], default="raw", help="Payload packing: varint tokens or raw UTF-8")
    ap.add_argument("--codec", choices=["raw", "lz4", "zstd"], default="zstd", help="Compression codec when payload >= threshold")
    ap.add_argument("--compress-threshold", type=int, default=256, help="Min payload bytes to apply compression; below sends raw")
    # FEC (Reed-Solomon) optional parity symbols
    ap.add_argument("--fec-nsym", type=int, default=0, help="FEC parity symbols (Reed-Solomon). 0 to disable.")
    # channel error injection
    ap.add_argument("--bitflip-prob", type=float, default=0.001, help="Per-bit flip probability injected after encoding (0 disables)")
    ap.add_argument("--bitflip-seed", type=int, default=None, help="Random seed for bit flipping (optional)")
    # zstd options
    ap.add_argument("--zstd-level", type=int, default=3)
    ap.add_argument("--zstd-dict", default="", help="Path to zstd dictionary file (optional)")
    # micro-batching
    ap.add_argument("--batch-ms", type=int, default=150, help="Batching time window in ms")
    ap.add_argument("--batch-bytes", type=int, default=2048, help="Batching size window in bytes")
    args = ap.parse_args()

    log_path = prepare_log_path(args.log_file, args.log_dir, prefix="tx")

    # RNG for bit flipping
    import random
    rng = random.Random(args.bitflip_seed)

    text_q: "queue.Queue[Optional[str]]" = queue.Queue()
    send_q: "queue.Queue" = queue.Queue(maxsize=64)
    net_log_q: "queue.Queue" = queue.Queue(maxsize=256)

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
    net_t = threading.Thread(target=sender_worker, args=(args.host, args.port, send_q, stop, net_log_q), daemon=True)
    # Network throughput accumulators
    sum_tx_bits = 0.0
    sum_tx_time = 0.0
    net_t.start()

    # Optional zstd dict
    zstd_dict_bytes = None
    if args.zstd_dict:
        try:
            with open(args.zstd_dict, "rb") as df:
                zstd_dict_bytes = df.read()
        except Exception as e:
            print(f"[sender] failed to read zstd dict: {e}")

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
        # Inject random bit flips on the encoded bytes (after compression, before framing)
        bitflips = 0
        flipped_bytes = 0
        if args.bitflip_prob and args.bitflip_prob > 0.0 and len(comp) > 0:
            ba = bytearray(comp)
            for i in range(len(ba)):
                mask = 0
                for b in range(8):
                    if rng.random() < args.bitflip_prob:
                        mask |= (1 << b)
                if mask:
                    ba[i] ^= mask
                    flipped_bytes += 1
                    # count set bits in mask (Hamming weight)
                    bitflips += (mask & 1) + ((mask >> 1) & 1) + ((mask >> 2) & 1) + ((mask >> 3) & 1) + ((mask >> 4) & 1) + ((mask >> 5) & 1) + ((mask >> 6) & 1) + ((mask >> 7) & 1)
            comp = bytes(ba)
        # Apply optional FEC (adds parity; increases size; can correct limited errors)
        if args.fec_nsym and args.fec_nsym > 0 and len(comp) > 0:
            comp = rs_fec_encode(comp, args.fec_nsym)
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
                "enc_ms": {
                    "tokenize": tok_ms,
                    "pack": pack_ms,
                    "compress": comp_ms,
                    "frame": frame_ms,
                    "total": total_ms,
                },
                "payload_b64": base64.b64encode(payload).decode("ascii"),
                "framed_b64": base64.b64encode(framed).decode("ascii"),
                "pack_mode": args.pack_mode,
                "compressed": compressed,
                "compress_threshold": args.compress_threshold,
                "codec": codec_used,
                "zstd_level": args.zstd_level,
                "zstd_dict": bool(zstd_dict_bytes),
                "fec_nsym": args.fec_nsym,
                "bitflip_prob": args.bitflip_prob,
                "bitflips": bitflips,
                "bitflip_bytes": flipped_bytes,
            }
        }

    try:
        # micro-batch loop
        batch_buf: list[str] = []
        batch_bytes = 0
        last_flush = time.monotonic()
        while True:
            timeout = max(0.0, (last_flush + args.batch_ms / 1000.0) - time.monotonic())
            try:
                line = text_q.get(timeout=timeout)
            except queue.Empty:
                line = None
            if line is None:
                # flush if any
                if batch_buf:
                    text = "\n".join(batch_buf)
                    rec = encode_one(text)
                    seq = seq + 1
                    send_q.put((seq, rec["framed"]))  # type: ignore
                    # update metrics and log
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
                    # drain net logs
                    while True:
                        try:
                            nl = net_log_q.get_nowait()
                        except queue.Empty:
                            break
                        sum_tx_bits += nl["bytes"] * 8.0
                        sum_tx_time += nl["send_ms"] / 1000.0
                        if log_path:
                            try:
                                with open(log_path, "a", encoding="utf-8") as f:
                                    f.write(json.dumps({
                                        "type": "tx-net",
                                        "ts": nl["ts"],
                                        "seq": nl["seq"],
                                        "bytes": nl["bytes"],
                                        "send_ms": nl["send_ms"],
                                        "tx_bps": nl["tx_bps"],
                                        "avg_tx_bps": (sum_tx_bits / sum_tx_time) if sum_tx_time > 0 else None,
                                    }) + "\n")
                            except Exception:
                                pass
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
                    batch_buf.clear()
                    batch_bytes = 0
                send_q.put(None)
                break
            # accumulate
            batch_buf.append(line)
            batch_bytes += len(line.encode("utf-8")) + 1  # +1 for newline we add on join
            now = time.monotonic()
            if batch_bytes >= args.batch_bytes or (now - last_flush) * 1000.0 >= args.batch_ms:
                text = "\n".join(batch_buf)
                rec = encode_one(text)
                seq = seq + 1
                send_q.put((seq, rec["framed"]))  # type: ignore
                # drain net logs
                while True:
                    try:
                        nl = net_log_q.get_nowait()
                    except queue.Empty:
                        break
                    sum_tx_bits += nl["bytes"] * 8.0
                    sum_tx_time += nl["send_ms"] / 1000.0
                    if log_path:
                        try:
                            with open(log_path, "a", encoding="utf-8") as f:
                                f.write(json.dumps({
                                    "type": "tx-net",
                                    "ts": nl["ts"],
                                    "seq": nl["seq"],
                                    "bytes": nl["bytes"],
                                    "send_ms": nl["send_ms"],
                                    "tx_bps": nl["tx_bps"],
                                    "avg_tx_bps": (sum_tx_bits / sum_tx_time) if sum_tx_time > 0 else None,
                                }) + "\n")
                        except Exception:
                            pass
                # update metrics and log
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
                batch_buf.clear()
                batch_bytes = 0
                last_flush = now
    finally:
        stop.set()
        net_t.join(timeout=2.0)
        # final drain net logs
        while True:
            try:
                nl = net_log_q.get_nowait()
            except queue.Empty:
                break
            sum_tx_bits += nl["bytes"] * 8.0
            sum_tx_time += nl["send_ms"] / 1000.0
            if log_path:
                try:
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "type": "tx-net",
                            "ts": nl["ts"],
                            "seq": nl["seq"],
                            "bytes": nl["bytes"],
                            "send_ms": nl["send_ms"],
                            "tx_bps": nl["tx_bps"],
                            "avg_tx_bps": (sum_tx_bits / sum_tx_time) if sum_tx_time > 0 else None,
                        }) + "\n")
                except Exception:
                    pass
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
                        "avg_tx_bps": (sum_tx_bits / sum_tx_time) if sum_tx_time > 0 else None,
                    }, ensure_ascii=False) + "\n")
            except Exception:
                pass


if __name__ == "__main__":
    main()
