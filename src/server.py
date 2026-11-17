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

from analysis import analyze_voice, clean_reference
from clone import XTTS_LANGS, Cancelled, Options, generate

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
JOBS: dict[str, dict] = {}


def safe_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9 _-]", "_", name.strip())
    name = re.sub(r"\s+", " ", name)[:40].strip()
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
        return {"cuda": cuda, "gpu": torch.cuda.get_device_name(0) if cuda else None,
                "langs": XTTS_LANGS}
    except Exception:
        return {"cuda": False, "gpu": None, "langs": XTTS_LANGS}


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


class RenameRequest(BaseModel):
    name: str = Field(min_length=1, max_length=40)


@app.post("/api/voices/{name}/rename")
def rename_voice(name: str, req: RenameRequest):
    src = voice_path(name)
    new = safe_name(req.name)
    dest = os.path.join(REFS, new + ".wav")
    if os.path.abspath(src).lower() != os.path.abspath(dest).lower() and os.path.exists(dest):
        raise HTTPException(409, f"A voice named '{new}' already exists")
    os.replace(src, dest)
    return {"name": new, "duration": round(sf.info(dest).duration, 1)}


@app.get("/api/voices/{name}/analysis")
def voice_analysis(name: str):
    return analyze_voice(voice_path(name))


@app.post("/api/voices/{name}/clean")
def clean_voice(name: str):
    """Make a continuous speech-only copy of a voice as '<name> clean'."""
    src = voice_path(name)
    new = safe_name(f"{name} clean")
    dest = os.path.join(REFS, new + ".wav")
    try:
        secs = clean_reference(src, dest)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"name": new, "duration": round(secs, 1)}


@app.get("/api/voices/{name}/audio")
def voice_audio(name: str):
    return FileResponse(voice_path(name), media_type="audio/wav")


class GenerateRequest(BaseModel):
    voice: str
    text: str = Field(min_length=1, max_length=20000)
    lang: str = "en"
    engine: str = "xtts"
    speed: float = Field(1.0, ge=0.5, le=2.0)
    gap: float = Field(0.25, ge=0.0, le=3.0)
    para_gap: float = Field(0.6, ge=0.0, le=5.0)
    duration: float | None = Field(None, gt=0, le=1800)
    temperature: float = Field(0.75, ge=0.1, le=1.5)
    top_p: float = Field(0.85, gt=0, le=1.0)
    top_k: int = Field(50, ge=1, le=200)
    repetition_penalty: float = Field(10.0, ge=1.0, le=20.0)
    seed: int | None = None
    ref_seconds: int = Field(30, ge=6, le=30)
    pitch: float = Field(0.0, ge=-12, le=12)
    gain_db: float = Field(0.0, ge=-20, le=20)
    normalize: bool = False
    trim: bool = True
    casual: bool = False
    indian_text: bool = False
    tech_terms: bool = False
    smart_pace: bool = False
    pronunciations: list[tuple[str, str]] = []
    takes: int = Field(1, ge=1, le=4)
    format: str = Field("wav", pattern="^(wav|mp3)$")


def run_job(job_id: str, req: GenerateRequest):
    job = JOBS[job_id]
    t0 = time.time()
    try:
        with gpu_lock:
            if job["cancel"].is_set():
                raise Cancelled()
            job["status"] = "running"
            takes = []
            for i in range(req.takes):
                o = Options(
                    lang=req.lang, engine=req.engine, speed=req.speed, gap=req.gap, para_gap=req.para_gap,
                    duration=req.duration, temperature=req.temperature, top_p=req.top_p, top_k=req.top_k,
                    repetition_penalty=req.repetition_penalty,
                    seed=None if req.seed is None else req.seed + i, ref_seconds=req.ref_seconds,
                    pitch=req.pitch, gain_db=req.gain_db, normalize=req.normalize, trim=req.trim,
                    casual=req.casual, indian_text=req.indian_text, tech_terms=req.tech_terms,
                    smart_pace=req.smart_pace, pronunciations=req.pronunciations)

                def progress(done, total, i=i):
                    job["done"], job["total"] = i * total + done, req.takes * total

                wav, sr, seed = generate(req.text, voice_path, req.voice, o, progress, job["cancel"])
                base = f"{safe_name(req.voice)}_{uuid.uuid4().hex[:8]}"
                path = os.path.join(OUT, base + ".wav")
                sf.write(path, wav, sr)
                fname = base + ".wav"
                if req.format == "mp3":
                    fname = base + ".mp3"
                    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-b:a", "192k",
                                    os.path.join(OUT, fname)], check=True)
                    os.remove(path)
                takes.append({"file": fname, "seconds": round(len(wav) / sr, 1), "seed": seed})
            elapsed = time.time() - t0
            job["result"] = {"takes": takes, "elapsed": round(elapsed, 1)}
            job["status"] = "done"
    except Cancelled:
        job["status"] = "cancelled"
    except HTTPException as e:
        job["status"], job["error"] = "error", str(e.detail)
    except Exception as e:  # surface any engine error to the UI
        job["status"], job["error"] = "error", f"{type(e).__name__}: {e}"


@app.post("/api/generate")
def start_generate(req: GenerateRequest):
    voice_path(req.voice)  # validate early
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {"status": "queued", "done": 0, "total": 0, "result": None, "error": None,
                    "cancel": threading.Event()}
    threading.Thread(target=run_job, args=(job_id, req), daemon=True).start()
    return {"job": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    j = JOBS.get(job_id)
    if not j:
        raise HTTPException(404, "Unknown job")
    return {k: v for k, v in j.items() if k != "cancel"}


@app.post("/api/jobs/{job_id}/cancel")
def job_cancel(job_id: str):
    j = JOBS.get(job_id)
    if not j:
        raise HTTPException(404, "Unknown job")
    j["cancel"].set()
    return {"ok": True}


@app.get("/api/audio/{fname}")
def get_audio(fname: str):
    p = os.path.join(OUT, os.path.basename(fname))
    if not os.path.isfile(p):
        raise HTTPException(404)
    mt = "audio/mpeg" if p.endswith(".mp3") else "audio/wav"
    return FileResponse(p, media_type=mt, filename=os.path.basename(fname))
