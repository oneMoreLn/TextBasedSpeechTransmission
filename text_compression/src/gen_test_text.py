import argparse
import random
import os

cn_sentences = [
    "这是一个流式文本压缩传输的测试。",
    "支持中文、英文、以及 emoji 😀😎🚀。",
    "今天天气不错，适合写代码。",
    "数据会被切分、压缩、再通过网络发送。",
    "接收端会解压并还原为原始文本。",
]

en_sentences = [
    "This is a streaming text compression and transport test.",
    "Mixing English with 中文 and emojis 😀.",
    "The data is framed, compressed, and sent over TCP.",
    "Receiver will decompress and reconstruct the original text.",
    "Latency depends on network and processing speed.",
]

extras = [
    "标点符号：，。！？；……—『』“”",
    "Symbols: ~!@#$%^&*()_+{}|:\"<>?`-=[]\\;',./",
]


def gen_line(i: int) -> str:
    parts = [random.choice(cn_sentences), random.choice(en_sentences)]
    if random.random() < 0.5:
        parts.append(random.choice(extras))
    return f"[{i:03d}] " + " ".join(parts)


def main():
    ap = argparse.ArgumentParser(description="Generate mixed-language test text file")
    ap.add_argument("--out", default="text_compression/test_data/stream_test.txt")
    ap.add_argument("--lines", type=int, default=50)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for i in range(1, args.lines + 1):
            f.write(gen_line(i) + "\n")
    print(f"Wrote {args.lines} lines to {args.out}")


if __name__ == "__main__":
    main()
