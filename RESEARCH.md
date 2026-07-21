# Qwen3-ASR-0.6B Performance Research

## The Problem

Qwen3-ASR-0.6B via PyTorch on CPU is too slow for real-time use (~15s for a 6s clip = 2.5x real-time). The `handy` tool achieves ~2-3s latency for the same model.

## Root Cause: PyTorch vs GGUF + GGML

| Aspect | Our server (malaiwah fork) | Handy |
|---|---|---|
| **Model format** | PyTorch (full float32) | GGUF (Q8_0 quantized, ~811MB) |
| **Inference engine** | torch + transformers | `transcribe-cpp` (Rust, GGML backend) |
| **Weight size** | ~2.4GB (float32) | ~811MB (Q8_0) |
| **Memory usage** | ~3-4GB | ~1.5GB |
| **Speed (6s clip)** | ~15s (2.5x RTF) | ~1-3s (~0.5x RTF) |
| **Platform** | Python | Rust + GGML |

## Key Discovery: GGUF Model Source

Handy downloads its model from HuggingFace:

- **Repo**: `handy-computer/Qwen3-ASR-0.6B-gguf`
- **Default file**: `Qwen3-ASR-0.6B-Q8_0.gguf` (~811 MB)
- **Alternate quants**: Q4_K_M (~562MB), Q5_K_M (~615MB), Q6_K (~658MB), BF16 (~1.5GB), F16 (~1.5GB)

## How Handy Works

Handy is a Tauri app (Rust backend + React frontend). Its ASR pipeline:

1. **Model loading** via `transcribe-cpp` Rust crate (v0.1.3)
   - Wraps GGML (the ML tensor library that powers llama.cpp)
   - Auto-detects architecture from GGUF file header
   - Qwen3-ASR is recognized as `qwen3_asr` architecture
2. **Audio capture** via `cpal` + `rubato` (resampling to 16kHz)
3. **VAD** (Voice Activity Detection) via `vad-rs`
4. **Transcription** via `transcribe-cpp::Model::session()`
5. **GPU support** (optional): Vulkan on Windows/Linux, Metal on macOS

## llama.cpp Server: The Best Path Forward

Research confirms that `llama.cpp` server **natively supports Qwen3-ASR** with the OpenAI-compatible `/v1/audio/transcriptions` endpoint. No Python, no PyTorch, no vLLM — pure GGML C++.

### Official Resources

| Resource | URL |
|---|---|
| Official GGUF model (ggml-org) | https://huggingface.co/ggml-org/Qwen3-ASR-0.6B-GGUF |
| llama.cpp multimodal docs | https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md |
| llama.cpp server Docker | `ghcr.io/ggml-org/llama.cpp:server` |
| llama.cpp PR #19441 (mtmd audio) | https://github.com/ggml-org/llama.cpp/pull/19441 |

### Usage

```bash
# Quickest path — llama-server auto-downloads the model
docker run -d --name qwen3-asr \
  -p 8000:8000 \
  -v model-cache:/root/.cache/llama.cpp \
  ghcr.io/ggml-org/llama.cpp:server \
  -hf ggml-org/Qwen3-ASR-0.6B-GGUF \
  --port 8000 --host 0.0.0.0 -t 4

# Then transcribe
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav -F model=qwen3-asr
```

The `-hf` flag downloads the default GGUF (Q8_0, ~811 MB) and its `mmproj` (audio encoder) automatically from HuggingFace to a local cache.

### Also Discovered

- **`femelo/py-qwen3-asr-cpp`** — Python bindings for a C++ GGML engine. Lets you write `Qwen3ASRModel(asr_model="qwen3-asr-0.6b-q8-0")` in Python, no PyTorch. https://github.com/femelo/py-qwen3-asr-cpp

- **`predict-woo/qwen3-asr.cpp`** — Standalone C++ server with OpenAI-compatible API and forced-alignment support. https://github.com/predict-woo/qwen3-asr.cpp

- **`1024th/qwen3-asr.cpp`** — Fork of above with improvements. https://github.com/1024th/qwen3-asr.cpp

- **`LocalAI`** — Supports Qwen3-ASR GGUF via `backend: llama-cpp` with `mmproj`. https://localai.io/features/audio-to-text/

### llama-cpp-python

`llama-cpp-python` supports Qwen3-ASR via `Qwen3ASRChatHandler` and `chat_handler` parameter. Example from `JamePeng2023/Qwen3-ASR-1.7B-GGUF`:

```python
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen3ASRChatHandler

llm = Llama(
    model_path="Qwen3-ASR-0.6B-Q8_0.gguf",
    chat_handler=Qwen3ASRChatHandler(
        clip_model_path="mmproj-Qwen3-ASR-0.6b-BF16.gguf"
    ),
)
```

But for a server, llama.cpp's built-in server is simpler and more maintained.

### Recommended: Use llama.cpp server Docker

The `ghcr.io/ggml-org/llama.cpp:server` image is ~300MB (vs 2-3GB for our PyTorch image), runs pure C++, and natively supports the OpenAI transcription API. This is the most practical path for CPU deployment.
