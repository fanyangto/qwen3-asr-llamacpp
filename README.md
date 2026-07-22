# qwen3-asr-llamacpp

Fast CPU-based speech-to-text using [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) via [llama.cpp](https://github.com/ggml-org/llama.cpp) GGUF, with an OpenAI-compatible proxy.

~0.5s for a 6s clip (12× real-time) on CPU. ~500 MB memory footprint, auto unload when idle. 

Why do you need it ? 
- It's so much faster and more accurate then the Hermes agent's built-in whisper the base model.
- It doesn't require you to have GPU. Runs on CPU is fast enough. It handles multiple languages: Chinese, English etc.
- It can run inside a VM (no GPU passthrough requried) or even integrate with your hermes agent containers. 
- You can have a private transcription service for sensitive audio content. 

## Quick Start

```bash
docker compose up -d
```
This will build the proxy container image on the first run. 

The first transcription request downloads the Qwen-ASR-0.6B Q8 GGUF model (~700 MB) into a named volume — subsequent startups are instant.

## Usage

```bash
curl -X POST http://localhost:8001/v1/audio/transcriptions \
  -F "file=@audio.wav" -F "response_format=json"
```

Returns `{"text":"<transcript>"}` with the `language X<asr_text>` prefix stripped.

### With Hermes Agent

Add to your Hermes `config.yaml`:

```yaml
stt:
  enabled: true
  provider: openai
  openai:
    base_url: http://asr.home:8001/v1
    api_key: sk-no-key-required
    model: qwen3-asr
```

And to `.env` (required for older Hermes versions where config `base_url` is ignored):

```
OPENAI_API_KEY=sk-no-key-required
STT_OPENAI_BASE_URL=http://asr.home:8001/v1
```

The `/v1` suffix is required — the OpenAI SDK appends `audio/transcriptions` to the base URL. The `api_key` can be any dummy value since our proxy doesn't check it.

## Performance

| Clip length | Latency | Real-time factor |
|-------------|---------|------------------|
| 6s          | ~0.5s   | ~12×             |
| 30s         | ~2.5s   | ~12×             |

Model: `Qwen3-ASR-0.6B-GGUF:Q8_0` — nearly lossless quantisation.

## Architecture

```
llama.cpp:server (ASR) ◄── proxy (FastAPI) ◄── client
    port 8000                  port 8001
    (internal)                 (published)
```

The proxy strips the `language X<asr_text>` prefix from llama.cpp's output and returns standard OpenAI `{"text":"..."}` format.
This makes the end point compatible with OpenAI ASR interface. 

## Files

```
├── docker-compose.yml     # asr + proxy services
├── proxy/
│   ├── Dockerfile         # lightweight alpine image (27 MB)
│   └── stt-proxy.py       # FastAPI proxy server
├── test/
│   └── test-asr.py        # CLI tester (no deps needed)
└── RESEARCH.md            # benchmarks & findings
```
