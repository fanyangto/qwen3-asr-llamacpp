"""Clean proxy: strips language prefix, returns standard {"text":"..."}."""
import os
import asyncio
import re
import logging

import httpx
import subprocess
import tempfile
from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse

LLAMA_HOST = os.getenv("LLAMA_HOST", "http://127.0.0.1:8000")


def _is_wav(data: bytes) -> bool:
    return data[:4] == b"RIFF" and data[8:12] == b"WAVE"


def _to_wav(data: bytes) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".in", delete=False) as inf:
        inf.write(data)
        inpath = inf.name
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as outf:
        outpath = outf.name
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", inpath, "-ar", "16000", "-ac", "1", "-f", "wav", outpath],
            capture_output=True,
        )
        if r.returncode != 0:
            err = r.stderr.decode(errors="replace")
            for line in err.splitlines():
                log.error("ffmpeg: %s", line)
            tail = err[-1500:]
            raise RuntimeError(f"ffmpeg exit {r.returncode}: ...{tail}")
        with open(outpath, "rb") as f:
            return f.read()
    finally:
        os.unlink(inpath)
        os.unlink(outpath)


_LANG_PREFIX_RE = re.compile(r"^language\s+\S+<asr_text>", re.IGNORECASE)
_RETRIES = int(os.getenv("ASR_RETRIES", "8"))
_RETRY_DELAY = float(os.getenv("ASR_RETRY_DELAY", "2.0"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("stt-proxy")
app = FastAPI(title="Qwen3-ASR Proxy")


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(None),
    language: str = Form(None),
    response_format: str = Form("json"),
):
    audio_bytes = await file.read()
    log.info("received %s (%d bytes, first 32 hex: %s)",
             file.filename or "audio", len(audio_bytes),
             audio_bytes[:32].hex())

    if not _is_wav(audio_bytes):
        log.info("converting to 16kHz mono WAV")
        audio_bytes = await asyncio.to_thread(_to_wav, audio_bytes)
        log.info("converted to %d bytes", len(audio_bytes))

    files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
    data = {"response_format": "json"}
    if model:
        data["model"] = model
    if language:
        data["language"] = language

    last_err = None
    for attempt in range(1, _RETRIES + 1):
        try:
            def _do_post():
                with httpx.Client(timeout=300) as client:
                    return client.post(f"{LLAMA_HOST}/v1/audio/transcriptions", files=files, data=data)

            resp = await asyncio.to_thread(_do_post)
            if resp.status_code == 200:
                raw = resp.json().get("text", "")
                clean = _LANG_PREFIX_RE.sub("", raw).strip()
                log.info("transcribed %d bytes → %d chars", len(audio_bytes), len(clean))
                if response_format == "text":
                    return PlainTextResponse(clean)
                return JSONResponse({"text": clean})

            last_err = f"ASR returned {resp.status_code}"
            log.warning("attempt %d/%d: %s", attempt, _RETRIES, last_err)

        except Exception as exc:
            last_err = str(exc)
            log.warning("attempt %d/%d: %s", attempt, _RETRIES, last_err)

        if attempt < _RETRIES:
            await asyncio.sleep(_RETRY_DELAY * (2 ** (attempt - 1)))

    raise HTTPException(503, f"ASR not available after {_RETRIES} retries: {last_err}")


@app.get("/health")
async def health():
    try:
        def _check():
            with httpx.Client(timeout=5) as client:
                return client.get(f"{LLAMA_HOST}/health")

        resp = await asyncio.to_thread(_check)
        if resp.status_code == 200:
            return {"status": "ok", "asr": "connected"}
    except Exception:
        pass
    return {"status": "degraded", "asr": "unreachable"}
