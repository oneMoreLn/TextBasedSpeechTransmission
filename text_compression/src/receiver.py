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
    # TTS options
    ap.add_argument("--tts-enable", action="store_true", help="Enable TTS synthesis using coqui-ai/TTS")
    ap.add_argument("--tts-model", default="tts_models/zh-CN/baker/tacotron2-DDC-GST", help="Coqui TTS model name or path")
    ap.add_argument("--tts-speaker", default=None, help="Optional speaker ID/name (multi-speaker models)")
    ap.add_argument("--tts-language", default=None, help="Optional language code (multilingual models)")
    ap.add_argument("--tts-wav", default="", help="Output WAV path; default under log/ with timestamp")
    # TTS pacing and quality
    ap.add_argument("--tts-trim-silence", action="store_true", help="Trim leading/trailing silence of each synthesized chunk")
    ap.add_argument("--tts-trim-db", type=int, default=40, help="librosa trim top_db for silence detection (when --tts-trim-silence)")
    ap.add_argument("--tts-gap-ms", type=int, default=150, help="Fixed short gap (ms) inserted between chunks after trimming")
    ap.add_argument("--tts-batch-wait-ms", type=int, default=200, help="Wait window (ms) to merge multiple texts into one TTS call")
    # TTS duration control
    ap.add_argument("--tts-speed", type=float, default=None, help="Optional synthesis speed factor if model supports it (e.g., 1.0 default, >1 faster)")
    ap.add_argument("--tts-max-sec", type=float, default=0.0, help="Hard cap for each chunk duration in seconds (0 to disable)")
    ap.add_argument("--tts-char-rate", type=float, default=3.0, help="Adaptive cap by text length: characters per second (0 to disable)")
    ap.add_argument("--tts-min-sec", type=float, default=0.0, help="Minimum duration per chunk before capping (seconds)")
    ap.add_argument("--tts-fade-ms", type=int, default=20, help="Fade-out duration when capping (ms)")
    # TTS chunk saving
    ap.add_argument("--tts-save-chunks", action="store_true", help="Also save each synthesized chunk as its own WAV file")
    ap.add_argument("--tts-chunk-dir", default="", help="Directory to save per-chunk WAVs (default is created under log/)")
    args = ap.parse_args()

    log_path = prepare_log_path(args.log_file, args.log_dir, prefix="rx")

    # Prepare TTS output wav path if needed
    if args.tts_enable:
        if not args.tts_wav:
            try:
                os.makedirs(args.log_dir, exist_ok=True)
                args.tts_wav = os.path.join(args.log_dir, f"tts_rx_{int(time.time())}.wav")
            except Exception:
                args.tts_wav = f"tts_rx_{int(time.time())}.wav"

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

    # Optional TTS worker
    tts_q: "queue.Queue[str]" = queue.Queue()
    tts_thread = None
    tts_stop = threading.Event()
    if args.tts_enable:
        def _float32_to_pcm16(arr):
            import numpy as _np
            arr = _np.asarray(arr, dtype=_np.float32)
            arr = _np.clip(arr, -1.0, 1.0)
            return (_np.round(arr * 32767.0).astype(_np.int16)).tobytes()

        def _tts_worker():
            # Lazy import to avoid hard dependency when disabled
            t_start_load = time.time()
            # Workaround for PyTorch>=2.6 default weights_only=True during torch.load
            # Allowlist custom optimizer/classes used by older TTS checkpoints (e.g., TTS.utils.radam.RAdam)
            try:
                import torch  # type: ignore
                import collections as _collections  # type: ignore
                try:
                    from TTS.utils.radam import RAdam as _RAdam  # type: ignore
                except Exception:
                    _RAdam = None  # type: ignore
                try:
                    globs = []
                    if _RAdam is not None:
                        globs.append(_RAdam)
                    globs.append(_collections.defaultdict)
                    try:
                        torch.serialization.add_safe_globals(globs)  # type: ignore[attr-defined]
                    except Exception:
                        pass
                except Exception:
                    pass
            except Exception:
                pass
            try:
                # Patch torch.load to default weights_only=False for legacy checkpoints
                try:
                    import torch as _torch  # type: ignore
                    _orig_load = _torch.load
                    def _patched_torch_load(*a, **k):
                        k.setdefault("weights_only", False)
                        return _orig_load(*a, **k)
                    _torch.load = _patched_torch_load  # type: ignore
                except Exception:
                    pass
                from TTS.api import TTS as _COQUI_TTS
            except Exception as e:
                msg = f"[tts] Import TTS failed: {e}"
                print(msg)
                if log_path:
                    try:
                        with open(log_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps({"type": "tts-error", "ts": time.time(), "err": msg}) + "\n")
                    except Exception:
                        pass
                return
            try:
                tts_obj = _COQUI_TTS(model_name=args.tts_model)
            except Exception as e:
                msg = f"[tts] Load model failed: {e}"
                print(msg)
                if log_path:
                    try:
                        with open(log_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps({"type": "tts-error", "ts": time.time(), "err": msg, "model": args.tts_model}) + "\n")
                    except Exception:
                        pass
                return
            # Resolve sample rate
            sr = None
            try:
                sr = int(getattr(tts_obj, "output_sample_rate", None) or getattr(getattr(tts_obj, "synthesizer", object()), "output_sample_rate", None) or 22050)
            except Exception:
                sr = 22050
            load_ms = (time.time() - t_start_load) * 1000.0
            if log_path:
                try:
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"type": "tts-info", "ts": time.time(), "event": "loaded", "model": args.tts_model, "sr": sr, "load_ms": load_ms}) + "\n")
                except Exception:
                    pass

            # Open wav for incremental write
            import wave as _wave
            try:
                wf = _wave.open(args.tts_wav, "wb")
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
            except Exception as e:
                msg = f"[tts] Open wav failed: {e}"
                print(msg)
                if log_path:
                    try:
                        with open(log_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps({"type": "tts-error", "ts": time.time(), "err": msg, "path": args.tts_wav}) + "\n")
                    except Exception:
                        pass
                return

            # Optional helpers
            def _trim_silence_if_needed(y):
                if not args.tts_trim_silence:
                    return y, 0.0
                try:
                    import numpy as _np
                    import librosa as _librosa
                    y = _np.asarray(y, dtype=_np.float32)
                    orig_len = int(y.shape[0])
                    yt, _ = _librosa.effects.trim(y, top_db=max(1, int(args.tts_trim_db)), frame_length=2048, hop_length=512)
                    trimmed = float(max(0, orig_len - int(yt.shape[0]))) * 1000.0 / float(sr)
                    return yt, trimmed
                except Exception:
                    return y, 0.0

            def _apply_time_limit(y, text_len: int):
                """Cap duration by adaptive rule and/or hard max, apply fade-out to avoid clicks."""
                try:
                    import numpy as _np
                    y = _np.asarray(y, dtype=_np.float32)
                    target_sec = 0.0
                    if args.tts_char_rate and args.tts_char_rate > 0:
                        target_sec = max(target_sec, float(text_len) / float(args.tts_char_rate))
                    if args.tts_min_sec and args.tts_min_sec > 0:
                        target_sec = max(target_sec, float(args.tts_min_sec))
                    if args.tts_max_sec and args.tts_max_sec > 0:
                        target_sec = target_sec if target_sec > 0 else float(args.tts_max_sec)
                        target_sec = min(target_sec, float(args.tts_max_sec))
                    if target_sec <= 0:
                        return y, 0.0
                    lim = int(float(sr) * target_sec)
                    if y.shape[0] <= lim:
                        return y, 0.0
                    fade = max(0, int(float(sr) * (max(0, int(args.tts_fade_ms)) / 1000.0)))
                    start = max(0, lim - fade)
                    if fade > 0 and lim > 0:
                        ramp = _np.linspace(1.0, 0.0, lim - start, dtype=_np.float32)
                        y[start:lim] = y[start:lim] * ramp
                    return y[:lim], float((y.shape[0] - lim) * 1000.0 / float(sr))
                except Exception:
                    return y, 0.0

            wrote_any = False
            # Prepare chunk output directory if requested
            chunk_dir = None
            chunk_idx = 0
            if args.tts_save_chunks:
                try:
                    base_dir = args.tts_chunk_dir or os.path.join(args.log_dir or ".", f"tts_chunks_{int(time.time())}")
                    os.makedirs(base_dir, exist_ok=True)
                    chunk_dir = base_dir
                    if log_path:
                        with open(log_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps({"type": "tts-info", "ts": time.time(), "event": "chunk-dir", "dir": chunk_dir}) + "\n")
                except Exception as e:
                    chunk_dir = None
                    if log_path:
                        try:
                            with open(log_path, "a", encoding="utf-8") as f:
                                f.write(json.dumps({"type": "tts-error", "ts": time.time(), "err": f"chunk dir prepare failed: {e}"}) + "\n")
                        except Exception:
                            pass

            # Consume texts and synthesize
            while not tts_stop.is_set():
                try:
                    txt = tts_q.get(timeout=0.2)
                except queue.Empty:
                    continue
                # sanitize input
                txt_norm = (txt or "").strip()
                if not txt_norm:
                    try:
                        tts_q.task_done()
                    except Exception:
                        pass
                    continue

                # Small wait window to merge consecutive texts
                if args.tts_batch_wait_ms and args.tts_batch_wait_ms > 0:
                    merged = [txt_norm]
                    deadline = time.time() + (args.tts_batch_wait_ms / 1000.0)
                    while time.time() < deadline:
                        try:
                            nxt = tts_q.get(timeout=max(0.0, deadline - time.time()))
                        except queue.Empty:
                            break
                        nxt = (nxt or "").strip()
                        if nxt:
                            merged.append(nxt)
                        try:
                            tts_q.task_done()
                        except Exception:
                            pass
                    txt_use = " ".join(merged)
                    batched_n = len(merged)
                else:
                    txt_use = txt_norm
                    batched_n = 1

                try:
                    t0 = time.time()
                    # try passing speed if provided, else fallback
                    if args.tts_speed is not None:
                        try:
                            wav = tts_obj.tts(text=txt_use, speaker=args.tts_speaker, language=args.tts_language, speed=float(args.tts_speed))
                        except Exception:
                            wav = tts_obj.tts(text=txt_use, speaker=args.tts_speaker, language=args.tts_language)
                    else:
                        wav = tts_obj.tts(text=txt_use, speaker=args.tts_speaker, language=args.tts_language)
                    
                    # Trim silence if requested
                    wav, trimmed_ms = _trim_silence_if_needed(wav)
                    # Duration cap (adaptive/hard)
                    wav, limited_ms = _apply_time_limit(wav, len(txt_use))
                    # Prepare PCM once
                    pcm = _float32_to_pcm16(wav)
                    # Optionally write this chunk into its own WAV
                    if chunk_dir is not None:
                        try:
                            chunk_path = os.path.join(chunk_dir, f"chunk_{chunk_idx:06d}.wav")
                            cwf = _wave.open(chunk_path, "wb")
                            cwf.setnchannels(1)
                            cwf.setsampwidth(2)
                            cwf.setframerate(sr)
                            cwf.writeframes(pcm)
                            cwf.close()
                            chunk_idx += 1
                            if log_path:
                                with open(log_path, "a", encoding="utf-8") as f:
                                    f.write(json.dumps({
                                        "type": "tts-chunk",
                                        "ts": time.time(),
                                        "path": chunk_path,
                                        "bytes": len(pcm),
                                        "len_text": len(txt_use),
                                        "batched": batched_n,
                                        "trimmed_ms": trimmed_ms,
                                        "sr": sr,
                                    }) + "\n")
                        except Exception as e:
                            if log_path:
                                try:
                                    with open(log_path, "a", encoding="utf-8") as f:
                                        f.write(json.dumps({"type": "tts-error", "ts": time.time(), "err": f"chunk save failed: {e}"}) + "\n")
                                except Exception:
                                    pass
                    # Optional fixed small gap between chunks to control pacing
                    gap_bytes = 0
                    if wrote_any and args.tts_gap_ms and args.tts_gap_ms > 0:
                        import numpy as _np
                        gap_len = int(sr * (args.tts_gap_ms / 1000.0))
                        if gap_len > 0:
                            wf.writeframes(_np.zeros(gap_len, dtype=_np.int16).tobytes())
                            gap_bytes = gap_len * 2
                    # Write current audio
                    wf.writeframes(pcm)
                    wrote_any = True
                    t1 = time.time()
                    if log_path:
                        with open(log_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps({
                                "type": "tts",
                                "ts": time.time(),
                                "len_text": len(txt_use),
                                "batched": batched_n,
                                "dur_ms": (t1 - t0) * 1000.0,
                                "trimmed_ms": trimmed_ms,
                                "gap_ms": (args.tts_gap_ms if gap_bytes > 0 else 0),
                                "limited_ms": limited_ms,
                                "speed": args.tts_speed,
                                "bytes": gap_bytes + len(pcm),
                                "sr": sr,
                            }) + "\n")
                except Exception as e:
                    msg = f"[tts] synth failed: {e}"
                    print(msg)
                    if log_path:
                        try:
                            with open(log_path, "a", encoding="utf-8") as f:
                                f.write(json.dumps({"type": "tts-error", "ts": time.time(), "err": msg}) + "\n")
                        except Exception:
                            pass
                finally:
                    try:
                        tts_q.task_done()
                    except Exception:
                        pass

            try:
                wf.close()
            except Exception:
                pass

        tts_thread = threading.Thread(target=_tts_worker, daemon=True)
        tts_thread.start()

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
                # Forward to TTS if enabled
                if args.tts_enable and text:
                    try:
                        tts_q.put_nowait(text)
                    except Exception:
                        pass
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
        if args.tts_enable:
            try:
                tts_stop.set()
            except Exception:
                pass
            if tts_thread is not None:
                try:
                    tts_thread.join(timeout=3.0)
                except Exception:
                    pass
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
