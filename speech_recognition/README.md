# Speech Streaming with SpeechRecognition + Faster-Whisper

This project captures microphone audio, performs streaming VAD-based chunking, transcribes with Faster-Whisper, and pushes partial/final texts into a thread-safe queue for downstream processing.

## Features
- Streaming microphone capture (16kHz mono PCM)
- WebRTC VAD to segment speech frames
- Faster-Whisper transcription in near real-time
- Partial + final transcripts pushed into a `queue.Queue`

## Requirements
- Python 3.9+
- Linux (tested). For other OS, adjust audio device backend if needed.
- A working microphone

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Note: The local folder `speech_recognition/` is empty and will shadow the PyPI package `speechrecognition`. To avoid import issues, do not import as a module from this folder. The code imports the third-party package `speech_recognition` via `import speech_recognition as sr`, which refers to the PyPI package.

## Run (microphone streaming)

```bash
python src/streaming_stt.py --model tiny --device auto
```

Options:
- `--model`: Faster-Whisper model size or path (e.g., `tiny`, `base`, `small`, `medium`, `large-v3`)
- `--device`: `cpu`, `cuda`, or `auto`
- `--beam-size`: Beam size for decoding (default 1)

## Try offline with a WAV file (for quick smoke test)

```bash
python src/offline_test.py test.wav --model tiny
```

## Consuming the output queue
The module exposes a `TranscriptQueue` (a `queue.Queue`) that receives dict items like:

```python
{"type": "partial", "text": "..."}
{"type": "final", "text": "..."}
{"type": "info", "message": "..."}
{"type": "error", "message": "..."}
```

You can import and consume it from another thread or process as needed.
