"""FastAPI backend for Voice Studio. Run: uvicorn server:app --app-dir src --port 8000"""
import os
import re
import subprocess
import threading
import time
import uuid

import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from clone import generate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFS = os.path.join(ROOT, "refs")
OUT = os.path.join(ROOT, "out")
os.makedirs(REFS, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

app = FastAPI(title="Voice Studio")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
gpu_lock = threading.Lock()  # one generation at a time on a 4 GB GPU


def safe_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_-]", "_", name.strip())[:40]
    if not name:
        raise HTTPException(400, "Invalid name")
    return name


def voice_path(name: str) -> str:
    p = os.path.join(REFS, safe_name(name) + ".wav")
    if not os.path.isfile(p):
        raise HTTPException(404, f"Voice '{name}' not found")
    return p


@app.get("/api/status")
def status():
    try:
        import torch
        cuda = torch.cuda.is_available()
        return {"cuda": cuda, "gpu": torch.cuda.get_device_name(0) if cuda else None}
    except Exception:
        return {"cuda": False, "gpu": None}


@app.get("/api/voices")
def list_voices():
    voices = []
    for f in sorted(os.listdir(REFS)):
        if f.lower().endswith(".wav"):
            info = sf.info(os.path.join(REFS, f))
            voices.append({"name": f[:-4], "duration": round(info.duration, 1)})
    return voices


@app.post("/api/voices")
async def add_voice(name: str = Form(...), file: UploadFile = File(...)):
    """Upload any audio file; it is converted to mono 24 kHz WAV."""
    name = safe_name(name)
    tmp = os.path.join(OUT, f"upload_{uuid.uuid4().hex}")
    with open(tmp, "wb") as fh:
        fh.write(await file.read())
    dest = os.path.join(REFS, name + ".wav")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-ac", "1", "-ar", "24000", dest],
            check=True,
        )
    except subprocess.CalledProcessError:
        raise HTTPException(400, "Could not read that audio file")
    finally:
        os.remove(tmp)
    return {"name": name, "duration": round(sf.info(dest).duration, 1)}


@app.delete("/api/voices/{name}")
def delete_voice(name: str):
    os.remove(voice_path(name))
    return {"ok": True}


@app.get("/api/voices/{name}/audio")
def voice_audio(name: str):
    return FileResponse(voice_path(name), media_type="audio/wav")


class GenerateRequest(BaseModel):
    voice: str
    text: str = Field(min_length=1, max_length=5000)
    lang: str = "en"
    engine: str = "xtts"
    speed: float = Field(1.0, ge=0.5, le=2.0)
    gap: float = Field(0.25, ge=0.0, le=3.0)
    duration: float | None = Field(None, gt=0, le=600)


@app.post("/api/generate")
def generate_audio(req: GenerateRequest):
    ref = voice_path(req.voice)
    t0 = time.time()
    with gpu_lock:
        wav, sr = generate(req.text, ref, req.engine, req.lang, req.speed, req.gap, req.duration)
    fname = f"{safe_name(req.voice)}_{uuid.uuid4().hex[:8]}.wav"
    sf.write(os.path.join(OUT, fname), wav, sr)
    secs = len(wav) / sr
    elapsed = time.time() - t0
    return {"file": fname, "seconds": round(secs, 1), "elapsed": round(elapsed, 1),
            "rtf": round(elapsed / secs, 2)}


@app.get("/api/audio/{fname}")
def get_audio(fname: str):
    p = os.path.join(OUT, os.path.basename(fname))
    if not os.path.isfile(p):
        raise HTTPException(404)
    return FileResponse(p, media_type="audio/wav", filename=os.path.basename(fname))
