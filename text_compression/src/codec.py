import io
import json
import os
import struct
from typing import Iterable, Iterator, List, Tuple

import lz4framed

# Minimal varint encode/decode (base-128) for non-negative ints

def varint_encode(n: int) -> bytes:
    if n < 0:
        raise ValueError("varint only supports non-negative integers")
    out = bytearray()
    while True:
        to_write = n & 0x7F
        n >>= 7
        if n:
            out.append(0x80 | to_write)
        else:
            out.append(to_write)
            break
    return bytes(out)


def varint_decode_stream(stream: io.BytesIO) -> Iterator[int]:
    while True:
        shift = 0
        result = 0
        while True:
            b = stream.read(1)
            if not b:
                return
            byte = b[0]
            result |= (byte & 0x7F) << shift
            if (byte & 0x80) == 0:
                yield result
                break
            shift += 7


# Tokenizer: UTF-8 bytes → varint tokens; Detokenizer: tokens → UTF-8 bytes
# This is simple and reversible; it's not semantic tokenization, but fits the spec.

def tokenize(text: str) -> List[int]:
    data = text.encode("utf-8")
    return list(data)


def detokenize(tokens: Iterable[int]) -> str:
    b = bytes(int(t) & 0xFF for t in tokens)
    return b.decode("utf-8", errors="strict")


# Pack tokens into a compact varint stream, with per-line framing
# Format: [varint token_count][varint t0][varint t1]...

def pack_tokens(tokens: List[int]) -> bytes:
    out = bytearray()
    out += varint_encode(len(tokens))
    for t in tokens:
        out += varint_encode(int(t))
    return bytes(out)


def unpack_tokens(payload: bytes) -> List[int]:
    bio = io.BytesIO(payload)
    it = varint_decode_stream(bio)
    try:
        n = next(it)
    except StopIteration:
        return []
    toks = []
    for _ in range(n):
        try:
            toks.append(next(it))
        except StopIteration:
            break
    return toks


# LZ4 framed compression helpers

def lz4_compress(data: bytes) -> bytes:
    return lz4framed.compress(data)


def lz4_decompress(data: bytes) -> bytes:
    return lz4framed.decompress(data)


# Network framing: 4-byte big-endian length prefix per LZ4 frame payload

HDR_STRUCT = struct.Struct(">I")


def frame(payload: bytes) -> bytes:
    return HDR_STRUCT.pack(len(payload)) + payload


def deframe(buffer: bytearray) -> Iterable[bytes]:
    # yields complete payloads, consumes from buffer
    off = 0
    while True:
        if len(buffer) - off < HDR_STRUCT.size:
            break
        (ln,) = HDR_STRUCT.unpack_from(buffer, off)
        if len(buffer) - off - HDR_STRUCT.size < ln:
            break
        start = off + HDR_STRUCT.size
        end = start + ln
        yield bytes(buffer[start:end])
        off = end
    if off:
        del buffer[:off]


__all__ = [
    "tokenize",
    "detokenize",
    "pack_tokens",
    "unpack_tokens",
    "lz4_compress",
    "lz4_decompress",
    "frame",
    "deframe",
]
