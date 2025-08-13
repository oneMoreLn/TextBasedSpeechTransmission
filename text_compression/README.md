# Text Streaming Compression & Transmission

This module implements a streaming text compression and transport pipeline using LZ4 (py-lz4framed) and Zstandard over sockets, with optional channel error injection.

Pipeline
- Encoder (sender):
  1. Read text lines from a local file into an input queue (for testing).
  2. Tokenize lines to integer tokens; convert tokens to a compact bitstream.
  3. Compress the bitstream using LZ4 (framed).
  4. Put compressed frames into a send queue.
  5. Read from the send queue and send via TCP socket to a receiver.
- Decoder (receiver):
  1. Receive compressed frames via TCP socket.
  2. Decompress frames using LZ4 (framed).
  3. Parse bitstream back to tokens and detokenize to text lines.
  4. Print and log reconstructed text.

Notes
- Tokenizer: a simple reversible integer mapping (UTF-8 bytes → varint list), robust to arbitrary Unicode. Default pack-mode is raw UTF-8.
- Framing: network uses a minimal 4-byte big-endian length header per payload (raw/LZ4/Zstd auto-sniffed on receiver).
- Logs: JSONL under `log/` capturing tx/rx events. Sender/receiver also log network throughput (tx-net/rx-net) and summaries.
- Channel error injection: you can flip random bits in the encoded payload to test robustness.

Quick start
- Start receiver:
  - conda run -n sptrans python text_compression/src/receiver.py --host 127.0.0.1 --port 9523 --log-dir log --expect auto
- Start sender (ASR→sender example):
  - conda run -n sptrans python text_compression/src/asr_to_sender.py --host 127.0.0.1 --port 9523 --source audio --audio-file test.wav --model medium --device cpu --print-asr --log-dir log --pack-mode raw --codec zstd --compress-threshold 256 --batch-ms 150 --batch-bytes 2048 --bitflip-prob 0.001 --bitflip-seed 42

Bit flip injection
- Both sender paths support bit flip injection after compression and before framing.
- Flags:
  - --bitflip-prob: per-bit flip probability (default 0.001). Set 0 to disable.
  - --bitflip-seed: RNG seed for reproducibility (optional).
- Logs include bitflip_prob, bitflips (total flipped bits), and bitflip_bytes (bytes affected) per batch.

