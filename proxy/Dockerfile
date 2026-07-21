FROM python:3.12-alpine
RUN apk add --no-cache ffmpeg
WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn httpx python-multipart
COPY stt-proxy.py /app/stt-proxy.py
EXPOSE 8000
CMD ["uvicorn", "stt-proxy:app", "--host", "0.0.0.0", "--port", "8000"]
