import argparse
import collections
import os
import queue
import sys
import threading
import time
import json
from typing import Deque, Optional

import numpy as np
import webrtcvad
import itertools

# Avoid local folder name shadowing the PyPI package `speech_recognition`
_SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
_WORKSPACE_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir))
_LOCAL_SHADOW_DIR = os.path.join(_WORKSPACE_ROOT, "speech_recognition")
if os.path.isdir(_LOCAL_SHADOW_DIR):
    # Remove workspace root ('') from sys.path if present to prefer site-packages
    # Keep script dir so our own imports still work.
    for p in ("", _WORKSPACE_ROOT):
        try:
            sys.path.remove(p)
        except ValueError:
            pass

# Import PyPI package safely
import speech_recognition as sr
from faster_whisper import WhisperModel


# Contract
# - Input: Microphone audio via SpeechRecognition (16kHz mono PCM)
# - Processing: WebRTC VAD frame gating -> buffer into speech chunks
# - Output: Put partial and final transcript strings into TranscriptQueue
# - Error modes: Enqueue an 'error' item and continue/stop accordingly


TranscriptQueue: "queue.Queue[dict]" = queue.Queue()


def seconds() -> float:
    return time.monotonic()


def pcm16_to_float32(pcm: bytes) -> np.ndarray:
    data = np.frombuffer(pcm, dtype=np.int16)
    return data.astype(np.float32) / 32768.0


def float32_to_pcm16(arr: np.ndarray) -> bytes:
    arr = np.clip(arr, -1.0, 1.0)
    return (arr * 32767.0).astype(np.int16).tobytes()


class VADStreamer:
    def __init__(self, sample_rate: int = 16000, frame_ms: int = 20, vad_aggressiveness: int = 2,
                 max_silence_ms: int = 800, min_speech_ms: int = 200, max_chunk_ms: int = 2000,
                 energy_threshold: float = 0.008):
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        self.max_silence_ms = max_silence_ms
        self.min_speech_ms = min_speech_ms
        self.max_chunk_ms = max_chunk_ms
        self.energy_threshold = energy_threshold

        self.frame_bytes = int(sample_rate * frame_ms / 1000) * 2  # 16-bit mono
        self.frames: Deque[bytes] = collections.deque()
        self.in_speech = False
        self.speech_frames: Deque[bytes] = collections.deque()
        self.speech_ms = 0
        self.silence_ms = 0

    def _is_speech_energy(self, frame: bytes) -> bool:
        arr = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(arr * arr) + 1e-12))
        return rms >= self.energy_threshold

    def process_frame(self, frame: bytes):
        """Feed 20ms PCM16 frame; yields b"chunk" when a speech segment ends."""
        assert len(frame) == self.frame_bytes, f"Expected {self.frame_bytes} bytes"
        # Try WebRTC VAD; if it errors, fallback to energy-based decision
        try:
            is_speech = self.vad.is_speech(frame, self.sample_rate)
        except Exception:
            is_speech = self._is_speech_energy(frame)

        if is_speech:
            if not self.in_speech:
                # Start of speech
                self.in_speech = True
                self.speech_frames.clear()
                self.speech_ms = 0
                self.silence_ms = 0
            self.speech_frames.append(frame)
            self.speech_ms += self.frame_ms
            # Force a chunk if speech is ongoing for too long without silence
            if self.speech_ms >= self.max_chunk_ms:
                chunk = b"".join(self.speech_frames)
                self.in_speech = False
                self.speech_frames.clear()
                self.speech_ms = 0
                self.silence_ms = 0
                return chunk
            else:
                return None
        else:
            if self.in_speech:
                self.silence_ms += self.frame_ms
                # Still append some trailing silence for context
                self.speech_frames.append(frame)
                if self.silence_ms >= self.max_silence_ms and self.speech_ms >= self.min_speech_ms:
                    # End of speech; emit chunk
                    chunk = b"".join(self.speech_frames)
                    self.in_speech = False
                    self.speech_frames.clear()
                    self.speech_ms = 0
                    self.silence_ms = 0
                    return chunk
            else:
                # idle
                return None

    def flush(self) -> Optional[bytes]:
        if self.in_speech and self.speech_ms >= self.min_speech_ms:
            chunk = b"".join(self.speech_frames)
            self.in_speech = False
            self.speech_frames.clear()
            self.speech_ms = 0
            self.silence_ms = 0
            return chunk
        return None


class SRMicSource:
    """Microphone source using SpeechRecognition to ensure cross-platform mic access.

    Captures raw PCM16 frames at 16kHz mono, 20ms per frame.
    """

    def __init__(self, sample_rate: int = 16000, frame_ms: int = 20, device_index: Optional[int] = None):
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_bytes = int(sample_rate * frame_ms / 1000) * 2
        self.device_index = device_index
        self.recognizer = sr.Recognizer()
        # Auto-pick a likely working device if none provided
        if self.device_index is None:
            try:
                names = sr.Microphone.list_microphone_names()
                preferred = ("pulse", "default")
                pick = None
                for pref in preferred:
                    for i, name in enumerate(names):
                        if name and pref in str(name).lower():
                            pick = i
                            break
                    if pick is not None:
                        break
                if pick is None and names:
                    pick = 0
                self.device_index = pick
                if pick is not None:
                    TranscriptQueue.put({"type": "info", "message": f"Auto-selected input device index: {pick} ({names[pick]})"})
                else:
                    TranscriptQueue.put({"type": "error", "message": "No input audio device found"})
            except Exception as e:
                TranscriptQueue.put({"type": "error", "message": f"List devices failed: {e}"})

    def frames(self):
        # Prepare candidate indices: user-specified, pulse, default
        candidates = []
        if self.device_index is not None:
            candidates.append(self.device_index)
        try:
            names = sr.Microphone.list_microphone_names()
        except Exception:
            names = []
        # try to find 'pulse' and 'default'
        def _find_by_keyword(keyword: str):
            try:
                for i, n in enumerate(names):
                    if n and keyword in str(n).lower():
                        return i
            except Exception:
                return None
            return None
        for key in ("pulse", "default"):
            idx = _find_by_keyword(key)
            if idx is not None and idx not in candidates:
                candidates.append(idx)
        # as last resort, index 0
        if names and 0 not in candidates:
            candidates.append(0)

        last_err = None
        for idx in candidates:
            try:
                # Log device info
                try:
                    name = names[idx] if (idx is not None and 0 <= idx < len(names)) else "unknown"
                    TranscriptQueue.put({"type": "info", "message": f"Opening microphone: index={idx}, name={name}"})
                except Exception:
                    pass
                with sr.Microphone(sample_rate=self.sample_rate, device_index=idx) as source:
                    # Calibrate ambient noise briefly
                    try:
                        self.recognizer.adjust_for_ambient_noise(source, duration=1)
                    except Exception:
                        pass
                    # Stream raw 16-bit PCM from the mic
                    for pcm in self._gen_raw_frames(source):
                        yield pcm
                return
            except Exception as e:
                last_err = e
                TranscriptQueue.put({"type": "error", "message": f"Open microphone failed (index={idx}): {e}"})
                continue

        TranscriptQueue.put({"type": "error", "message": f"All microphone candidates failed. Last error: {last_err}"})
        return

    def _gen_raw_frames(self, source):
        frames_per_read = int(self.sample_rate * self.frame_ms / 1000)
        while True:
            try:
                # Read raw frames directly from PyAudio stream via SpeechRecognition source
                try:
                    raw = source.stream.read(frames_per_read, exception_on_overflow=False)
                except TypeError:
                    # Older PyAudio may not support the kwarg; fallback
                    raw = source.stream.read(frames_per_read)
            except Exception as e:
                TranscriptQueue.put({"type": "error", "message": f"Mic read error: {e}"})
                time.sleep(0.05)
                continue
            if not raw:
                continue
            if len(raw) < self.frame_bytes:
                raw = raw + b"\x00" * (self.frame_bytes - len(raw))
            yield raw[: self.frame_bytes]


class FasterWhisperWorker(threading.Thread):
    def __init__(self, model_size: str = "tiny", device: str = "cpu", beam_size: int = 1,
                 language: str = "zh", task: str = "transcribe", temperature: float = 0.0,
                 initial_prompt: str = ""):
        super().__init__(daemon=True)
        self.model_size = model_size
        self.device = device
        self.beam_size = beam_size
        self.language = language
        self.task = task
        self.temperature = temperature
        self.initial_prompt = initial_prompt
        self.model: Optional[WhisperModel] = None
        self.running = threading.Event()
        self.running.set()

    def run(self):
        # Avoid accidental CUDA on misconfigured systems
        dev = self.device
        if dev == "auto":
            dev = "cpu"
        if dev == "cpu":
            os.environ.setdefault("CT2_FORCE_CPU", "1")
        try:
            self.model = WhisperModel(self.model_size, device=dev, compute_type="float16" if dev == "cuda" else "int8")
            TranscriptQueue.put({"type": "info", "message": f"Model loaded: {self.model_size} on {dev}"})
        except Exception as e:
            TranscriptQueue.put({"type": "error", "message": f"Model load failed: {e}"})
            return

        while self.running.is_set():
            try:
                item = audio_chunks.get(timeout=0.1)
            except queue.Empty:
                continue

            # Support both legacy bytes and new dict payload with metadata
            if isinstance(item, dict) and "pcm" in item:
                pcm = item.get("pcm", b"")
                chunk_id = item.get("id")
                enq_t0 = item.get("t0")
                dur_s = float(item.get("dur_s", 0.0))
                sr = int(item.get("sr", 16000))
            else:
                pcm = item if isinstance(item, (bytes, bytearray)) else b""
                chunk_id = None
                enq_t0 = None
                sr = 16000
                dur_s = (len(pcm) / (2 * sr)) if pcm else 0.0

            # Convert PCM16 -> float32 numpy
            audio_f32 = pcm16_to_float32(pcm)
            t_start = seconds()
            # Faster-Whisper expects float32 numpy array at original sample rate
            try:
                segments, info = self.model.transcribe(
                    audio_f32,
                    beam_size=self.beam_size,
                    vad_filter=False,
                    language=self.language if self.language else None,
                    task=self.task,
                    temperature=self.temperature,
                    initial_prompt=self.initial_prompt if self.initial_prompt else None,
                    condition_on_previous_text=True,
                )
                # Emit partials as we iterate, finals after chunk
                partial_text = []
                for seg in segments:
                    if seg.text:
                        TranscriptQueue.put({"type": "partial", "text": seg.text, "chunk_id": chunk_id})
                        partial_text.append(seg.text)
                if partial_text:
                    TranscriptQueue.put({"type": "final", "text": " ".join(partial_text).strip(), "chunk_id": chunk_id})
                t_end = seconds()
                proc_s = max(0.0, t_end - t_start)
                e2e_s = (t_end - enq_t0) if isinstance(enq_t0, (int, float)) else None
                rtf = (proc_s / dur_s) if dur_s and dur_s > 0 else None
                TranscriptQueue.put({
                    "type": "metrics",
                    "chunk_id": chunk_id,
                    "dur_s": dur_s,
                    "proc_s": proc_s,
                    "e2e_s": e2e_s,
                    "rtf": rtf,
                    "device": self.device,
                    "model": self.model_size,
                })
            except Exception as e:
                TranscriptQueue.put({"type": "error", "message": f"Transcribe failed: {e}"})
                continue

    def stop(self):
        self.running.clear()


audio_chunks: "queue.Queue[object]" = queue.Queue(maxsize=8)
_chunk_id_counter = itertools.count(1)


def mic_loop(sample_rate: int, frame_ms: int, device_index: Optional[int], vad_aggr: int, max_silence_ms: int,
             min_speech_ms: int, max_chunk_ms: int, energy_threshold: float):
    src = SRMicSource(sample_rate=sample_rate, frame_ms=frame_ms, device_index=device_index)
    vad = VADStreamer(sample_rate=sample_rate, frame_ms=frame_ms, vad_aggressiveness=vad_aggr,
                      max_silence_ms=max_silence_ms, min_speech_ms=min_speech_ms,
                      max_chunk_ms=max_chunk_ms, energy_threshold=energy_threshold)
    TranscriptQueue.put({"type": "info", "message": "Mic loop started"})
    for frame in src.frames():
        chunk = vad.process_frame(frame)
        if chunk is not None:
            try:
                cid = next(_chunk_id_counter)
                dur_s = len(chunk) / (2 * sample_rate)
                audio_chunks.put({"pcm": chunk, "id": cid, "t0": seconds(), "dur_s": dur_s, "sr": sample_rate}, timeout=0.5)
            except queue.Full:
                # Drop if backlog; report
                TranscriptQueue.put({"type": "info", "message": "Dropping chunk due to backlog"})
    # Flush residual
    tail = vad.flush()
    if tail:
        try:
            cid = next(_chunk_id_counter)
            dur_s = len(tail) / (2 * sample_rate)
            audio_chunks.put({"pcm": tail, "id": cid, "t0": seconds(), "dur_s": dur_s, "sr": sample_rate}, timeout=0.5)
        except queue.Full:
            TranscriptQueue.put({"type": "info", "message": "Dropping tail due to backlog"})
    TranscriptQueue.put({"type": "info", "message": "Mic loop ended"})


def main():
    parser = argparse.ArgumentParser(description="Streaming STT with SpeechRecognition + Faster-Whisper")
    parser.add_argument("--model", default="tiny")
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--frame-ms", type=int, default=20, choices=[10, 20, 30])
    parser.add_argument("--device-index", type=int, default=None)
    parser.add_argument("--print-queue", action="store_true", help="Print transcripts from queue for testing")
    parser.add_argument("--log-file", default="", help="Append JSON lines with {chunk_id, text, rtf, e2e_s, dur_s, proc_s, model, device, ts}")
    parser.add_argument("--log-dir", default="log", help="Directory to save logs when --log-file is empty (filename auto-generated)")
    parser.add_argument("--list-devices", action="store_true", help="List available input audio devices and exit")
    # VAD tuning
    parser.add_argument("--vad-aggr", type=int, default=2, choices=[0,1,2,3], help="WebRTC VAD aggressiveness")
    parser.add_argument("--max-silence-ms", type=int, default=800)
    parser.add_argument("--min-speech-ms", type=int, default=200)
    parser.add_argument("--max-chunk-ms", type=int, default=1500, help="Force-cut chunk if speech runs longer than this")
    parser.add_argument("--energy-threshold", type=float, default=0.006, help="Fallback energy VAD RMS threshold")
    # Decoding tuning
    parser.add_argument("--language", default="zh", help="Language hint, e.g., zh, en. Empty for auto-detect.")
    parser.add_argument("--task", default="transcribe", choices=["transcribe","translate"], help="Whisper task")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 for greedy)")
    parser.add_argument("--initial-prompt", default="", help="Optional initial prompt to bias decoding context")
    args = parser.parse_args()

    # Prepare logging path: if no --log-file, auto-generate under --log-dir
    if not args.log_file:
        try:
            os.makedirs(args.log_dir, exist_ok=True)
            auto_name = f"stt_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
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
            TranscriptQueue.put({"type": "info", "message": f"Logging to {args.log_file}"})
        except Exception:
            pass

    if args.list_devices:
        try:
            names = sr.Microphone.list_microphone_names()
            print("SpeechRecognition input devices:")
            for i, name in enumerate(names):
                print(f"  [{i}] {name}")
        except Exception as e:
            print("Unable to list devices via SpeechRecognition:", e)
        # Try PyAudio directly for more detail
        try:
            import pyaudio
            pa = pyaudio.PyAudio()
            print("\nPyAudio devices:")
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                if int(info.get('maxInputChannels', 0)) > 0:
                    print(f"  [{i}] {info.get('name')} sr={int(info.get('defaultSampleRate', 0))} ch={int(info.get('maxInputChannels', 0))}")
            pa.terminate()
        except Exception as e:
            print("Unable to list devices via PyAudio:", e)
        return

    # Worker
    worker = FasterWhisperWorker(model_size=args.model, device=args.device, beam_size=args.beam_size,
                                 language=args.language, task=args.task, temperature=args.temperature,
                                 initial_prompt=args.initial_prompt)
    worker.start()

    # Optional queue consumer for testing output
    stop_printer = threading.Event()
    printer_thread = None
    if args.print_queue or args.log_file:
        def _printer():
            rtf_hist = collections.deque(maxlen=10)
            e2e_hist = collections.deque(maxlen=10)
            # Accumulate per-chunk info to pair final text with metrics
            pair: dict = {}
            while not stop_printer.is_set():
                try:
                    item = TranscriptQueue.get(timeout=0.2)
                except queue.Empty:
                    continue
                t = item.get("type")
                if t in {"partial", "final"}:
                    if args.print_queue:
                        print(f"[{t}] {item.get('text','')}")
                    if t == "final" and args.log_file:
                        cid = item.get("chunk_id")
                        if cid is not None:
                            rec = pair.get(cid, {})
                            rec["text"] = item.get("text", "")
                            pair[cid] = rec
                elif t == "metrics":
                    rtf = item.get("rtf")
                    e2e = item.get("e2e_s")
                    if isinstance(rtf, (int, float)):
                        rtf_hist.append(float(rtf))
                    if isinstance(e2e, (int, float)):
                        e2e_hist.append(float(e2e))
                    rtf_str = f"{rtf:.3f}x" if isinstance(rtf, (int, float)) else "n/a"
                    e2e_str = f"{e2e:.3f}s" if isinstance(e2e, (int, float)) else "n/a"
                    if args.print_queue:
                        print(f"[metrics] chunk={item.get('chunk_id')} dur={item.get('dur_s'):.3f}s proc={item.get('proc_s'):.3f}s rtf={rtf_str} e2e={e2e_str}")
                        # Rolling averages over last 10
                        avg_rtf_str = f"{(sum(rtf_hist)/len(rtf_hist)):.3f}x" if len(rtf_hist) else "n/a"
                        avg_e2e_str = f"{(sum(e2e_hist)/len(e2e_hist)):.3f}s" if len(e2e_hist) else "n/a"
                        n = max(len(rtf_hist), len(e2e_hist))
                        print(f"[metrics-avg10] rtf={avg_rtf_str} e2e={avg_e2e_str} n={n}")
                    if args.log_file:
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
                            # If we have both text and rtf, write log
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
                                    with open(args.log_file, "a", encoding="utf-8") as f:
                                        f.write(json.dumps(out, ensure_ascii=False) + "\n")
                                except Exception:
                                    pass
                                # Cleanup to avoid unbounded dict
                                pair.pop(cid, None)
                else:
                    if args.print_queue:
                        print(f"[{t}] {item}")

        printer_thread = threading.Thread(target=_printer, daemon=True)
        printer_thread.start()

    # Producer loop in main thread to allow KeyboardInterrupt
    try:
        mic_loop(sample_rate=args.sample_rate, frame_ms=args.frame_ms, device_index=args.device_index,
                 vad_aggr=args.vad_aggr, max_silence_ms=args.max_silence_ms,
                 min_speech_ms=args.min_speech_ms, max_chunk_ms=args.max_chunk_ms,
                 energy_threshold=args.energy_threshold)
    except KeyboardInterrupt:
        pass
    finally:
        # Drain any remaining queue items to ensure logs are visible before exit
        if args.print_queue or args.log_file:
            drained = 0
            rtf_hist = collections.deque(maxlen=10)
            e2e_hist = collections.deque(maxlen=10)
            pair: dict = {}
            while True:
                try:
                    item = TranscriptQueue.get_nowait()
                except queue.Empty:
                    break
                t = item.get("type")
                if t in {"partial", "final"}:
                    if args.print_queue:
                        print(f"[{t}] {item.get('text','')}")
                    if t == "final" and args.log_file:
                        cid = item.get("chunk_id")
                        if cid is not None:
                            rec = pair.get(cid, {})
                            rec["text"] = item.get("text", "")
                            pair[cid] = rec
                elif t == "metrics":
                    rtf = item.get("rtf")
                    e2e = item.get("e2e_s")
                    if isinstance(rtf, (int, float)):
                        rtf_hist.append(float(rtf))
                    if isinstance(e2e, (int, float)):
                        e2e_hist.append(float(e2e))
                    rtf_str = f"{rtf:.3f}x" if isinstance(rtf, (int, float)) else "n/a"
                    e2e_str = f"{e2e:.3f}s" if isinstance(e2e, (int, float)) else "n/a"
                    if args.print_queue:
                        print(f"[metrics] chunk={item.get('chunk_id')} dur={item.get('dur_s'):.3f}s proc={item.get('proc_s'):.3f}s rtf={rtf_str} e2e={e2e_str}")
                        avg_rtf_str = f"{(sum(rtf_hist)/len(rtf_hist)):.3f}x" if len(rtf_hist) else "n/a"
                        avg_e2e_str = f"{(sum(e2e_hist)/len(e2e_hist)):.3f}s" if len(e2e_hist) else "n/a"
                        n = max(len(rtf_hist), len(e2e_hist))
                        print(f"[metrics-avg10] rtf={avg_rtf_str} e2e={avg_e2e_str} n={n}")
                    if args.log_file:
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
                                    with open(args.log_file, "a", encoding="utf-8") as f:
                                        f.write(json.dumps(out, ensure_ascii=False) + "\n")
                                except Exception:
                                    pass
                                pair.pop(cid, None)
                else:
                    if args.print_queue:
                        print(f"[{t}] {item}")
                drained += 1
            if drained == 0:
                # Give worker/printer a brief moment to emit late logs
                time.sleep(0.2)
        if printer_thread is not None:
            stop_printer.set()
            printer_thread.join(timeout=1.0)
        worker.stop()
        worker.join(timeout=2.0)


if __name__ == "__main__":
    main()
