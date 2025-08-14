import time
import types
import numpy as np

from voice_process import VoiceProcessor


class DummyASR:
    def transcribe(self, pcm_16k, language="zh", beam_size=1):
        # Return list of objects with .text
        txt = "这是一个测试。"  # fixed text
        seg = types.SimpleNamespace(text=txt)
        return [seg], {}


class DummyTTS:
    def __init__(self, sr=16000):
        self.output_sample_rate = sr

    def tts(self, text: str):
        # Generate a short sine wave; duration scales with text length
        dur = max(0.2, min(1.0, len(text) * 0.05))
        t = np.linspace(0, dur, int(self.output_sample_rate * dur), endpoint=False)
        y = 0.2 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        return y


def main():
    vp = VoiceProcessor(cycle_seconds=1.0)  # shorter cycle for test
    # Inject dummies to avoid heavy model loads
    vp._asr = DummyASR()
    vp._tts = DummyTTS(sr=int(round(vp.chunk_samples * 1000.0 / vp.chunk_interval_ms)))
    vp._tts_sr = vp._tts.output_sample_rate

    # Feed one cycle worth of chunks
    for _ in range(vp.cycle_chunks):
        chunk = (np.random.rand(vp.chunk_samples) * 2 - 1)
        pcm16 = np.round(chunk * 32767).astype(np.int16)
        vp.voice_encode(pcm16)

    # Drain produced packets and immediately decode
    pkts = []
    while not vp.voice2msg_queue.empty():
        pkts.append(vp.voice2msg_queue.get())

    for p in pkts:
        vp.voice_decode(p)

    # Read a few output chunks
    out = 0
    while not vp.voice_recv_queue.empty() and out < 5:
        c = vp.voice_recv_queue.get()
        print("recv chunk", out, c.shape, c.dtype, int(c[:5].mean()))
        out += 1

    print("done; packets:", len(pkts), "cycle_chunks:", vp.cycle_chunks)


if __name__ == "__main__":
    main()
