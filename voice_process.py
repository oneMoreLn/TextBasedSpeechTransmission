import os
import sys
import math
import time
import queue
from typing import Optional, Dict, Tuple, Any, cast

import numpy as np

# Make text_compression/src importable
_THIS_DIR = os.path.dirname(__file__)
_CODEC_DIR = os.path.join(_THIS_DIR, "text_compression", "src")
if _CODEC_DIR not in sys.path:
    sys.path.insert(0, _CODEC_DIR)

try:
    from codec import zstd_compress, zstd_decompress  # type: ignore
except Exception:
    # Fallback no-op compressors for environments without codec deps (demo/testing)
    def zstd_compress(b: bytes, level: int = 3, dict_bytes: Optional[bytes] = None) -> bytes:  # type: ignore
        return b

    def zstd_decompress(b: bytes, dict_bytes: Optional[bytes] = None) -> bytes:  # type: ignore
        return b


class VoiceProcessor:
    """
    Voice encode/decode pipeline with fixed-size hex packet framing.

    Packet format (fixed length = packet_len bytes):
    - [0:2]   message_id (uint16, big-endian)
    - [2:4]   total_len (uint16): total compressed bytes for this message
    - [4]     total_segs (uint8): number of segments this message splits into
    - [5]     seg_idx (uint8): 0..total_segs-1
    - [6:8]   seg_len (uint16): bytes of segment payload in this packet
    - [8:...] payload (seg_len bytes)
    - [...]   zero padding to packet_len

    - For transport, we push ASCII hex string (length=packet_len*2) into voice2msg_queue.
      Padding produces trailing zeros in hex. On decode, seg_len ensures padding is ignored.

    Queues (external):
    - voice_send_queue: np.int16, shape (chunk_samples,). One chunk every chunk_interval_ms.
    - voice2msg_queue: str (hex packet) per segment.
    - msg2voice_queue: str (hex packet) or bytes per segment (incoming from transport).
    - voice_recv_queue: np.int16, shape (chunk_samples,) (synthesized; one chunk every chunk_interval_ms).
    """

    def __init__(
        self,
        voice_send_queue: Optional["queue.Queue"] = None,
        voice_recv_queue: Optional["queue.Queue"] = None,
        voice2msg_queue: Optional["queue.Queue"] = None,
        msg2voice_queue: Optional["queue.Queue"] = None,
        *,
        packet_len: int = 2088,
        cycle_seconds: float = 5.0,
        chunk_samples: int = 1024,
        chunk_interval_ms: int = 250,
        asr_model_name: str = "base",
        asr_device: str = "cpu",
        tts_model_name: str = "tts_models/zh-CN/baker/tacotron2-DDC-GST",
    ) -> None:
        # Queues
        self.voice_send_queue = voice_send_queue or queue.Queue()
        self.voice_recv_queue = voice_recv_queue or queue.Queue()
        self.voice2msg_queue = voice2msg_queue or queue.Queue()
        self.msg2voice_queue = msg2voice_queue or queue.Queue()

        # Packet/frame config
        self.packet_len = int(packet_len)
        self.header_len = 8
        assert self.packet_len > self.header_len
        self.payload_cap = self.packet_len - self.header_len

        # Timing / chunking
        self.cycle_seconds = float(cycle_seconds)
        self.chunk_samples = int(chunk_samples)
        self.chunk_interval_ms = int(chunk_interval_ms)
        self.cycle_chunks = max(1, int(round(self.cycle_seconds * 1000.0 / self.chunk_interval_ms)))

        # Models
        self.asr_model_name = asr_model_name
        self.asr_device = asr_device
        self._asr = None  # lazy ASR handle

        self.tts_model_name = tts_model_name
        self._tts = None  # lazy TTS handle
        self._tts_sr = 22050

        # State
        self._pcm_cycle = []
        self._message_id = 0
        # reassembly state: mid -> (total_len, total_segs, received_count, parts dict)
        self._rx_assemblers = {}

    # -----------------
    # Utility functions
    # -----------------
    @staticmethod
    def _to_hex(b: bytes) -> str:
        return b.hex()

    @staticmethod
    def _from_hex(s: str) -> bytes:
        return bytes.fromhex(s.strip())

    @staticmethod
    def _pcm16_to_float32(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32) / 32768.0
        return np.clip(x, -1.0, 1.0)

    @staticmethod
    def _float32_to_pcm16(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -1.0, 1.0)
        return np.round(x * 32767.0).astype(np.int16)

    @staticmethod
    def _resample_linear(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
        if src_sr == dst_sr or x.size == 0:
            return x.copy()
        ratio = dst_sr / float(src_sr)
        dst_len = max(1, int(round(x.shape[0] * ratio)))
        xp = np.linspace(0.0, 1.0, num=x.shape[0], endpoint=True)
        fp = x.astype(np.float32)
        x_new = np.linspace(0.0, 1.0, num=dst_len, endpoint=True)
        y = np.interp(x_new, xp, fp).astype(np.float32)
        return y

    def _ensure_asr(self):
        if self._asr is not None:
            return
        from faster_whisper import WhisperModel  # type: ignore
        self._asr = WhisperModel(self.asr_model_name, device=self.asr_device)

    def _ensure_tts(self):
        if self._tts is not None:
            return
        # Coqui TTS; tolerate load issues
        from TTS.api import TTS  # type: ignore
        self._tts = TTS(model_name=self.tts_model_name)
        # Try detect SR if available
        try:
            sr = int(getattr(self._tts, "output_sample_rate", None) or getattr(getattr(self._tts, "synthesizer", object()), "output_sample_rate", None) or 22050)
            self._tts_sr = sr
        except Exception:
            self._tts_sr = 22050

    # -----------------
    # Public API
    # -----------------
    def voice_encode(self, voice_data: np.ndarray):
        """
        用于输入语音数据压缩：voice_send_queue -> 语音识别+文本编码 -> 分包 -> voice2msg_queue
        输入：voice_data: np.int16, shape=(chunk_samples,)
        逻辑：每 250ms 进一帧，累计一个发送周期（默认 5s）后进行一次 ASR，将文本压缩、分段，
        生成固定长度（packet_len）的十六进制字符串包，逐包写入 voice2msg_queue。
        """
        if not isinstance(voice_data, np.ndarray) or voice_data.dtype != np.int16 or voice_data.shape[0] != self.chunk_samples:
            raise ValueError(f"voice_data must be np.int16 with shape ({self.chunk_samples},)")

        self._pcm_cycle.append(voice_data.copy())
        if len(self._pcm_cycle) < self.cycle_chunks:
            return  # not enough for one cycle yet

        # Assemble cycle PCM and reset
        pcm = np.concatenate(self._pcm_cycle, axis=0)
        self._pcm_cycle.clear()

        # Resample from device SR (derived from chunk size/interval) to 16k for ASR
        src_sr = int(round(self.chunk_samples * 1000.0 / self.chunk_interval_ms))
        pcm_f32 = self._pcm16_to_float32(pcm)
        pcm_16k = self._resample_linear(pcm_f32, src_sr=src_sr, dst_sr=16000)

        # ASR using faster-whisper
        text = ""
        try:
            self._ensure_asr()
            if self._asr is None:
                return
            asr = cast(Any, self._asr)
            segments, info = asr.transcribe(pcm_16k, language="zh", beam_size=1)
            lines = [seg.text for seg in segments]
            text = (" ".join(lines)).strip()
        except Exception:
            text = ""

        # Encode text -> compressed bytes (UTF-8 + zstd)
        payload = text.encode("utf-8")
        comp = zstd_compress(payload, level=3, dict_bytes=None)

        # Split into fixed-size packets (with header + padding)
        mid = self._message_id & 0xFFFF
        self._message_id = (self._message_id + 1) & 0xFFFF
        total_len = len(comp)
        total_segs = max(1, math.ceil(total_len / self.payload_cap))

        for seg_idx in range(total_segs):
            start = seg_idx * self.payload_cap
            end = min(total_len, start + self.payload_cap)
            part = comp[start:end]
            seg_len = len(part)

            header = bytearray(self.header_len)
            header[0:2] = (mid).to_bytes(2, "big")
            header[2:4] = (total_len).to_bytes(2, "big")
            header[4] = total_segs & 0xFF
            header[5] = seg_idx & 0xFF
            header[6:8] = (seg_len).to_bytes(2, "big")

            pkt = bytes(header) + part
            if len(pkt) < self.packet_len:
                pkt = pkt + b"\x00" * (self.packet_len - len(pkt))
            else:
                pkt = pkt[: self.packet_len]

            # push hex packet
            self.voice2msg_queue.put(self._to_hex(pkt))

    def voice_decode(self, recv_data):
        """
        用于接收语音数据解压缩：msg2voice_queue -> 解包 -> 文本解码+语音合成 -> voice_recv_queue
        输入：recv_data: 十六进制字符串（或 bytes）。解出完整消息后进行一次合成，并按 1024 样本片推送。
        """
        # Normalize to bytes
        if isinstance(recv_data, str):
            pkt = self._from_hex(recv_data)
        elif isinstance(recv_data, (bytes, bytearray)):
            pkt = bytes(recv_data)
        else:
            raise ValueError("recv_data must be hex str or bytes")

        if len(pkt) != self.packet_len:
            # Ignore malformed packets
            return

        # Parse header
        mid = int.from_bytes(pkt[0:2], "big")
        total_len = int.from_bytes(pkt[2:4], "big")
        total_segs = pkt[4]
        seg_idx = pkt[5]
        seg_len = int.from_bytes(pkt[6:8], "big")
        part = bytes(pkt[8:8+seg_len]) if seg_len > 0 else b""

        if total_segs <= 1:
            comp = part
        else:
            # reassemble
            state = self._rx_assemblers.get(mid)
            if state is None:
                state = (total_len, total_segs, 0, {})
                self._rx_assemblers[mid] = state
            tlen, tsegs, rcvd, parts = state
            parts[seg_idx] = part
            rcvd = len(parts)
            self._rx_assemblers[mid] = (tlen, tsegs, rcvd, parts)
            if rcvd < tsegs:
                return  # wait for more
            # stitch
            comp = b"".join(parts[i] for i in range(tsegs))[:tlen]
            del self._rx_assemblers[mid]

        # Decompress and decode text
        try:
            payload = zstd_decompress(comp, dict_bytes=None)
            text = payload.decode("utf-8", errors="replace").strip()
        except Exception:
            text = ""

        if not text:
            return

        # TTS synth
        try:
            self._ensure_tts()
            if self._tts is None:
                return
            tts = cast(Any, self._tts)
            # Call with text only to avoid type issues if speaker/language not required
            wav = tts.tts(text=text)
            wav = np.asarray(wav, dtype=np.float32)
        except Exception:
            return

        # Resample to device chunk rate and enqueue in 1024-sample frames
        dst_sr = int(round(self.chunk_samples * 1000.0 / self.chunk_interval_ms))
        wav_rs = self._resample_linear(wav, src_sr=self._tts_sr, dst_sr=dst_sr)
        pcm16 = self._float32_to_pcm16(wav_rs)

        # chunking
        n = pcm16.shape[0]
        step = self.chunk_samples
        for i in range(0, n, step):
            chunk = pcm16[i:i+step]
            if chunk.shape[0] < step:
                # pad last chunk to full length to maintain cadence
                pad = np.zeros(step - chunk.shape[0], dtype=np.int16)
                chunk = np.concatenate([chunk, pad], axis=0)
            self.voice_recv_queue.put(chunk)


# Optional: simple worker loops (not required by the interface, provided as helpers)
def run_encode_loop(vp: VoiceProcessor):
    """Drain voice_send_queue and call voice_encode for each chunk."""
    while True:
        try:
            data = vp.voice_send_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        if data is None:
            break
        vp.voice_encode(data)


def run_decode_loop(vp: VoiceProcessor):
    """Drain msg2voice_queue and call voice_decode for each packet."""
    while True:
        try:
            pkt = vp.msg2voice_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        if pkt is None:
            break
        vp.voice_decode(pkt)
