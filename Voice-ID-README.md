# Voice ID -- Self Hosted Speaker Identification Service

Self-hosted speaker identification service using:

-   SpeechBrain ECAPA (Voice Embeddings)
-   FastAPI
-   PostgreSQL + pgvector
-   Docker Compose

------------------------------------------------------------------------

## Project Structure

voice-id/ │ ├── docker-compose.yml ├── init.sql └── api/ ├── Dockerfile
├── requirements.txt └── main.py

------------------------------------------------------------------------

## docker-compose.yml

services: db: image: pgvector/pgvector:pg16 environment: POSTGRES_DB:
voiceid POSTGRES_USER: voiceid POSTGRES_PASSWORD: voiceid ports: -
"5432:5432" volumes: - dbdata:/var/lib/postgresql/data -
./init.sql:/docker-entrypoint-initdb.d/init.sql:ro

api: build: ./api environment: DATABASE_URL:
postgresql+psycopg://voiceid:voiceid@db:5432/voiceid API_KEY:
change-me-please EMBEDDING_DIM: 192 ports: - "8088:8088" depends_on: -
db

volumes: dbdata:

------------------------------------------------------------------------

## init.sql

create extension if not exists vector;

create table if not exists speakers ( id bigserial primary key,
speaker_key text unique not null, display_name text, embedding
vector(192) not null, created_at timestamptz not null default now(),
meta jsonb not null default '{}'::jsonb );

create index if not exists speakers_embedding_hnsw on speakers using
hnsw (embedding vector_cosine_ops);

------------------------------------------------------------------------

## api/Dockerfile

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg
&& rm -rf /var/lib/apt/lists/\*

WORKDIR /app COPY requirements.txt /app/ RUN pip install --no-cache-dir
-r requirements.txt

COPY main.py /app/ EXPOSE 8088 CMD \["uvicorn", "main:app", "--host",
"0.0.0.0", "--port", "8088"\]

------------------------------------------------------------------------

## api/requirements.txt

fastapi==0.115.0 uvicorn\[standard\]==0.30.6 numpy==2.0.1
soundfile==0.12.1 torch==2.4.0 speechbrain==1.0.0 sqlalchemy==2.0.32
psycopg\[binary\]==3.2.1 python-multipart==0.0.9

------------------------------------------------------------------------

## Start

docker compose up -d --build

Health check:

http://localhost:8088/health

------------------------------------------------------------------------

## Extract Audio

ffmpeg -i input.mp4 -ac 1 -ar 16000 -vn audio.wav -y

------------------------------------------------------------------------

## Enroll Speaker

curl -X POST "http://localhost:8088/enroll?speaker_key=creator_markus"
-H "x-api-key: change-me-please" -F "file=@audio.wav"

------------------------------------------------------------------------

## Identify Speaker

curl -X POST "http://localhost:8088/identify?threshold=0.80" -H
"x-api-key: change-me-please" -F "file=@audio.wav"

------------------------------------------------------------------------

## Tuning

-   Lower threshold if too many unknown results (0.76--0.78)
-   Raise threshold if false matches (0.85+)
-   Use 60--120 seconds clean audio for enrollment
