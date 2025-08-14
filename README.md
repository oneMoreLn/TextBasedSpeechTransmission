# Text-Based Speech Transmission

A real-time speech-to-text transmission pipeline with efficient text compression, robust transport, and optional text-to-speech synthesis on the receiver.

## Features

- Streaming ASR (microphone / audio file) with VAD chunking and Faster-Whisper
- Efficient text transport over TCP with length framing, micro-batching, compression (LZ4/Zstd), varint/raw packing
- Robustness: tolerant UTF-8 decode, optional random bit flips injection, Reed-Solomon FEC
- Extensive JSONL logging for TX/RX/metrics/throughput; includes base64 payloads for reproducibility
- Receiver-side optional TTS (Coqui TTS) to synthesize a single WAV and per-segment WAVs
- TTS pacing controls: silence trimming, small fixed inter-chunk gap, short-window merging
- Duration shaping: adaptive capping based on text length (tts-char-rate), min/max sec, and fade-out

## Project layout

```
.
├── speech_recognition/
│   ├── src/streaming_stt.py          # Real-time STT with VAD + Faster-Whisper
│   └── src/offline_test.py           # Offline test harness
├── text_compression/
│   ├── src/sender.py                 # Sender: packs/compresses and sends text
│   ├── src/receiver.py               # Receiver: deframes/decompresses/decodes, optional TTS
│   ├── src/codec.py                  # Framing, packing, compression, FEC helpers
│   └── requirements.txt              # Submodule requirements (merged in top-level)
├── requirements.txt                  # Unified project requirements
├── README.md                         # This file
└── test.wav                          # Example audio file
```

## Environment

- Recommended: Conda env with Python 3.9 (project is pinned for Py39 compatibility with TTS 0.20.x)
- GPU optional; CPU works for base models and TTS but will be slower

### Create environment

```bash
conda create -n sptrans python=3.9 -y
conda activate sptrans
pip install -r requirements.txt
```

Notes on TTS:
- TTS version < 0.21 is pinned for Python 3.9 compatibility.
- Some older checkpoints need safe load tweaks, handled in receiver runtime (allowlisted globals and torch.load weights_only=False fallback).
- If you see import/load issues, ensure the environment uses the pinned versions.

## Usage

Open two terminals (both in the `sptrans` env).

### 1) Start receiver (with TTS)

```bash
python text_compression/src/receiver.py \
  --host 127.0.0.1 --port 9500 --log-dir log --expect auto --fec-nsym 0 \
  --tts-enable --tts-model tts_models/zh-CN/baker/tacotron2-DDC-GST \
  --tts-trim-silence --tts-gap-ms 120 --tts-batch-wait-ms 200 \
  --tts-char-rate 3 --tts-min-sec 0.6 --tts-max-sec 20 --tts-fade-ms 25 \
  --tts-save-chunks --tts-wav log/tts_rx_test.wav
```

- Logs are written under `log/` (rx, rx-net, tts, tts-chunk, errors)
- A single growing WAV `log/tts_rx_test.wav` and per-chunk WAVs in a chunk dir (if enabled)

### 2A) Send recognized text from an audio file

```bash
python text_compression/src/asr_to_sender.py \
  --host 127.0.0.1 --port 9500 \
  --source audio --audio-file test.wav --model base --device cpu \
  --print-asr --log-dir log --pack-mode raw --codec zstd \
  --compress-threshold 256 --batch-ms 150 --batch-bytes 2048
```

### 2B) Or stream from mic

```bash
python text_compression/src/asr_to_sender.py \
  --host 127.0.0.1 --port 9500 \
  --source mic --model base --device cpu \
  --print-asr --log-dir log --pack-mode raw --codec zstd \
  --compress-threshold 256 --batch-ms 150 --batch-bytes 2048
```

## voice_process 模块（封装式 API）

`voice_process.py` 提供一个独立于 sender/receiver 的封装式语音处理接口，用于：

- voice_encode：采集端每 250ms 输入一帧 1024 样本（int16），每 5 秒聚合为一周期，运行 ASR，得到文本后 UTF-8 编码并进行 zstd 压缩，按照固定报文长度分包（默认 2088 字节）并转为十六进制字符串输出到 `voice2msg_queue`。
- voice_decode：接收端从 `msg2voice_queue` 取十六进制字符串数据包，按报文头重组完整消息并去除填充，解压得到文本，调用 TTS 合成为音频，再按 1024 样本分片推送到 `voice_recv_queue`。

### 固定长度报文格式

- 固定长度：`packet_len`（默认 2088 字节）
- 头部 8 字节：
  - [0:2] message_id (uint16, big-endian)
  - [2:4] total_len (uint16): 本消息压缩后总字节数
  - [4] total_segs (uint8): 当前消息被切成的总段数
  - [5] seg_idx (uint8): 当前包的段索引（0..total_segs-1）
  - [6:8] seg_len (uint16): 本包负载实际字节数
- 负载：`seg_len` 字节
- 填充：其后全部补 `0x00` 直到 `packet_len`
- 传输：以 ASCII 十六进制字符串（长度 `packet_len*2`）进行传输；解包时依赖 `seg_len/total_len` 去除填充。

### 关键参数

- packet_len：固定包长度（默认 2088）
- cycle_seconds：聚合一个 ASR 周期的时长（默认 5.0s）
- chunk_samples：每帧样本数（默认 1024）
- chunk_interval_ms：帧间隔（默认 250ms）
- asr_model_name / asr_device：Faster-Whisper 模型与设备
- tts_model_name：Coqui TTS 模型（默认 `tts_models/zh-CN/baker/tacotron2-DDC-GST`）

### 最小示例（本地自测，不加载真实模型）

项目提供 `demo_voice_process_debug.py`，通过 DummyASR/DummyTTS 验证分包/重组/合成链路：

```bash
python demo_voice_process_debug.py
```

若需集成到你的应用：

```python
from voice_process import VoiceProcessor
import numpy as np

vp = VoiceProcessor(packet_len=2088, cycle_seconds=5.0)

# 采集侧：每 250ms 推入一帧 1024 样本的 int16
vp.voice_encode(np.zeros(1024, dtype=np.int16))

# 取出固定长度十六进制数据包，发往你的传输通道
hex_packet = vp.voice2msg_queue.get()

# 接收侧：收到十六进制数据包后送入解码
vp.voice_decode(hex_packet)

# 合成的音频以 1024 样本帧输出
chunk = vp.voice_recv_queue.get()  # np.ndarray(int16, shape=(1024,))
```

### 注意事项

- `voice_process` 为轻量封装，压缩默认使用 zstd；在未安装依赖时 Demo 会回退为直通（仅便于测试），正式部署请安装 `text_compression/requirements.txt` 并使用真实压缩。
- TTS 在 Python 3.9 环境下与版本钉死更稳定（TTS<0.21），如在 Python 3.13 运行 Demo，请仅作为链路验证。
- 报文头字段需双方严格一致；`message_id` 仅用于重组与去重，未实现重传与乱序修复。
- `cycle_seconds` 会影响端到端延迟与识别稳定性，按需求权衡。

## Important notes

- Always run both sender and receiver in the same environment (e.g., `sptrans`).
- Keep port and FEC settings consistent on both sides (local test: `--port 9500 --fec-nsym 0`).
- First TTS load can take several seconds; watch for a `tts-info` "loaded" log entry.
- If the receiver prints `ModuleNotFoundError: lz4framed`, you're not in the correct environment.

## Tuning pacing and duration

- Reduce long pauses:
  - `--tts-trim-silence` to trim synthesized silence
  - `--tts-gap-ms 80..150` to control small inter-chunk gaps
  - `--tts-batch-wait-ms 150..250` to merge very short sentences
- Avoid long tails on short texts:
  - `--tts-char-rate 10..14`, `--tts-min-sec 0.5..0.8`, optional `--tts-max-sec 3..5`
  - `--tts-fade-ms 20..40` to avoid clicks on cuts

## Logging

- `rx`: per message decode details and timings
- `rx-net`: instantaneous and average bandwidth
- `tts`: synthesis timing, trimmed_ms, limited_ms (if duration-capped)
- `tts-chunk`: path and bytes of each saved chunk
- `rx-summary`: averages at shutdown

## Troubleshooting

- No audio generated:
  - Check port/FEC match and environment; ensure TTS model is loaded
- TTS import errors:
  - Confirm TTS<0.21 and `bangla<0.0.5` on Python 3.9
- Many `\ufffd` replacements:
  - Transport bit flips or decode errors; verify FEC settings and pack mode

## License

This project is for research and educational purposes. Review third-party licenses of dependencies and pretrained models.

## 命令行参数说明

以下仅列出常用关键参数，更多详见源码内 argparse 定义。

### text_compression/src/receiver.py

- 通用接收
  - --host: 监听地址，默认 127.0.0.1
  - --port: 端口，默认 9500
  - --log-file: 指定日志文件路径（可选）；为空则自动命名到 --log-dir 下
  - --log-dir: 日志目录，默认 log
  - --expect: 接收到的负载格式，auto|lz4|zstd|raw，默认 auto（自动嗅探压缩格式）
  - --zstd-dict: zstd 字典路径（可选）
  - --fec-nsym: Reed-Solomon FEC 冗余符号个数，0 关闭（需与发送端一致）

- TTS 开关与模型
  - --tts-enable: 开启接收端合成
  - --tts-model: Coqui TTS 模型名或路径（例：tts_models/zh-CN/baker/tacotron2-DDC-GST）
  - --tts-speaker: 说话人（多说话人模型时可选）
  - --tts-language: 多语模型的语言代码（可选）
  - --tts-wav: 输出单一合成 WAV 文件路径（默认：log/tts_rx_时间戳.wav）

- TTS 节奏/衔接
  - --tts-trim-silence: 修剪每段合成的前后静音
  - --tts-trim-db: librosa trim 的 top_db 阈值（默认 40）
  - --tts-gap-ms: 段间固定小间隔（毫秒，默认 150）
  - --tts-batch-wait-ms: 短窗合并多条文本到一次合成（默认 200ms）

- TTS 时长自适应与裁剪
  - --tts-speed: 合成语速（模型支持时有效；1.0 默认，>1 更快）
  - --tts-char-rate: 自适应上限：按字符速率（字符/秒）估算目标时长；0 关闭
  - --tts-min-sec: 目标时长下限（秒），避免极短文本过短
  - --tts-max-sec: 目标时长上限（秒），0 关闭
  - --tts-fade-ms: 超过目标裁剪时的淡出时长（毫秒，默认 20），避免截断爆音

- TTS 分段保存
  - --tts-save-chunks: 同时保存每段合成为独立 WAV
  - --tts-chunk-dir: 分段 WAV 保存目录（不填则在 log/ 下按时间戳创建）

日志字段补充：
- tts：包含 dur_ms、trimmed_ms、gap_ms、limited_ms（若发生裁剪）、bytes、sr 等
- tts-chunk：分段 WAV 的 path、bytes、len_text、batched、sr 等

### text_compression/src/sender.py

- 通用发送
  - --host / --port: 目标地址与端口
  - --log-file / --log-dir: 日志输出
  - --source: 输入源类型（当前仅 text-file）
  - --input-file / --text-file: 文本文件路径（等价参数，二选一）

- 编码/压缩
  - --pack-mode: varint|raw，默认 raw（直接 UTF-8）；varint 走 token pack
  - --codec: raw|lz4|zstd，默认 zstd
  - --compress-threshold: 小于该字节数不压缩，默认 256
  - --zstd-level: zstd 压缩等级（默认 3）
  - --zstd-dict: zstd 字典路径（可选）

- 渠道鲁棒性
  - --fec-nsym: Reed-Solomon 冗余符号，0 关闭
  - --bitflip-prob / --bitflip-seed: 随机比特翻转注入概率与随机种子（用于容错实验）

- 微批
  - --batch-ms: 时间窗口（毫秒，默认 150）
  - --batch-bytes: 字节窗口（默认 2048）

日志字段补充：
- tx：包含各阶段耗时 enc_ms、长度统计、编码方式与压缩比例等
- tx-net：每包发送时延与瞬时/平均带宽

### speech_recognition/src/streaming_stt.py（要点）

- 采集/模型
  - --source: mic|audio-file|text-file（结合 asr_to_sender 使用）
  - --model: Faster-Whisper 模型大小（base/small/...）
  - --device: cpu|cuda
  - 其他 VAD/采样与输出控制参数略（详见源码）

### text_compression/src/asr_to_sender.py（要点）

- 端口/日志：同 sender
- 源选择与模型：
  - --source: mic|audio
  - --audio-file: 当 source=audio 时指定文件
  - --model / --device: 选择 Faster-Whisper 模型与设备
- 传输编码：同 sender（--pack-mode/--codec/--compress-threshold/微批参数等）
