import io
import logging
import os
import subprocess
import sys
import tempfile
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np

# Configure logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)
import soundfile as sf
import torch
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile, Header
from pgvector.sqlalchemy import Vector
from speechbrain.inference.speaker import EncoderClassifier
from sqlalchemy import Column, BigInteger, Text, DateTime, create_engine, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Session, sessionmaker, declarative_base

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql+psycopg://voiceid:voiceid@localhost:5432/voiceid")
API_KEY = os.environ.get("API_KEY", "change-me-please")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "192"))

Base = declarative_base()


class Speaker(Base):
    __tablename__ = "speakers"

    id = Column(BigInteger, primary_key=True)
    speaker_id = Column("speaker_key", Text, unique=True, nullable=False)
    display_name = Column(Text)
    embedding = Column(Vector(EMBEDDING_DIM), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=text("now()"))
    meta = Column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))


engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

classifier: Optional[EncoderClassifier] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="/tmp/speechbrain_model",
        run_opts={"device": "cpu"}
    )
    yield


app = FastAPI(title="Voice ID", lifespan=lifespan)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


def convert_to_wav(input_bytes: bytes) -> bytes:
    """Convert any audio/video format to 16kHz mono WAV using ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".input", delete=False) as input_file:
        input_file.write(input_bytes)
        input_path = input_file.name

    output_path = input_path + ".wav"

    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", input_path,
                "-ac", "1",           # mono
                "-ar", "16000",       # 16kHz
                "-vn",                # no video
                "-f", "wav",
                output_path
            ],
            capture_output=True,
            check=True
        )

        with open(output_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(input_path):
            os.unlink(input_path)
        if os.path.exists(output_path):
            os.unlink(output_path)


def extract_embedding(audio_bytes: bytes) -> np.ndarray:
    # Try to read directly first, fallback to ffmpeg conversion
    try:
        audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
    except Exception:
        # Convert with ffmpeg (handles video, mp3, etc.)
        wav_bytes = convert_to_wav(audio_bytes)
        audio_data, sample_rate = sf.read(io.BytesIO(wav_bytes))

    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)

    if sample_rate != 16000:
        import torchaudio.transforms as T
        resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
        audio_tensor = torch.tensor(audio_data, dtype=torch.float32)
        audio_data = resampler(audio_tensor).numpy()

    audio_tensor = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        embedding = classifier.encode_batch(audio_tensor)

    return embedding.squeeze().cpu().numpy()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": classifier is not None}


@app.post("/enroll")
def enroll(
    speaker_id: str = Query(...),
    display_name: Optional[str] = Query(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    _: str = Depends(verify_api_key)
):
    audio_bytes = file.file.read()

    try:
        embedding = extract_embedding(audio_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process audio: {str(e)}")

    existing = db.query(Speaker).filter(Speaker.speaker_id == speaker_id).first()

    if existing:
        existing.embedding = embedding.tolist()
        if display_name:
            existing.display_name = display_name
        db.commit()
        return {"message": "Speaker updated", "speaker_id": speaker_id}

    speaker = Speaker(
        speaker_id=speaker_id,
        display_name=display_name,
        embedding=embedding.tolist()
    )
    db.add(speaker)
    db.commit()

    return {"message": "Speaker enrolled", "speaker_id": speaker_id}


@app.post("/identify")
def identify(
    file: UploadFile = File(...),
    threshold: float = Query(0.80, ge=0.0, le=1.0),
    db: Session = Depends(get_db),
    _: str = Depends(verify_api_key)
):
    audio_bytes = file.file.read()

    try:
        embedding = extract_embedding(audio_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process audio: {str(e)}")

    embedding_list = embedding.tolist()

    result = db.execute(
        text("""
            SELECT speaker_key as speaker_id, display_name,
                   1 - (embedding <=> CAST(:embedding AS vector)) as similarity
            FROM speakers
            ORDER BY embedding <=> CAST(:embedding AS vector)
            LIMIT 1
        """),
        {"embedding": str(embedding_list)}
    ).fetchone()

    if result is None:
        return {"identified": False, "message": "No speakers enrolled"}

    speaker_id, display_name, similarity = result

    if similarity >= threshold:
        return {
            "identified": True,
            "speaker_id": speaker_id,
            "display_name": display_name,
            "similarity": round(float(similarity), 4)
        }

    return {
        "identified": False,
        "message": "No match above threshold",
        "best_match": {
            "speaker_id": speaker_id,
            "similarity": round(float(similarity), 4)
        }
    }


@app.get("/speakers")
def list_speakers(
    db: Session = Depends(get_db),
    _: str = Depends(verify_api_key)
):
    speakers = db.query(Speaker.speaker_id, Speaker.display_name, Speaker.created_at).all()
    return {
        "speakers": [
            {
                "speaker_id": s.speaker_id,
                "display_name": s.display_name,
                "created_at": s.created_at.isoformat() if s.created_at else None
            }
            for s in speakers
        ]
    }


@app.delete("/speakers/{speaker_id}")
def delete_speaker(
    speaker_id: str,
    db: Session = Depends(get_db),
    _: str = Depends(verify_api_key)
):
    speaker = db.query(Speaker).filter(Speaker.speaker_id == speaker_id).first()
    if not speaker:
        raise HTTPException(status_code=404, detail="Speaker not found")

    db.delete(speaker)
    db.commit()
    return {"message": "Speaker deleted", "speaker_id": speaker_id}
