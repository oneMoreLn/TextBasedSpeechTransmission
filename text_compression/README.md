# Text Streaming Compression & Transmission

This module implements a streaming text compression and transport pipeline using LZ4 (py-lz4framed) over sockets.

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
- Tokenizer: a simple reversible integer mapping (UTF-8 bytes → varint list), robust to arbitrary Unicode.
- Framing: network uses a minimal 4-byte big-endian length header per LZ4-compressed payload.
- Logs: JSONL under `log/` capturing tx/rx events.

Quick start
- Start receiver:
  - python text_compression/src/receiver.py --host 127.0.0.1 --port 9500 --log-dir log
- Start sender:
  - python text_compression/src/sender.py --host 127.0.0.1 --port 9500 --input-file test.txt --log-dir log

