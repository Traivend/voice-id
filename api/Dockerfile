FROM python:3.11-slim

# Force full rebuild - v3
RUN echo "build-v3" > /tmp/.build-version

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py log_config.yaml /app/
EXPOSE 8088
CMD ["uvicorn", "main:app", "--host", "::", "--port", "8088", "--log-config", "/app/log_config.yaml"]
