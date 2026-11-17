"""Training-dataset builder for XTTS fine-tuning.

Layout:  datasets/<name>/raw/*.wav        uploaded or recorded audio (mono, 22.05 kHz)
         datasets/<name>/wavs/*.wav       3-11 s clips
         datasets/<name>/metadata_train.csv, metadata_eval.csv   (Coqui format: audio_file|text|speaker_name)
"""
import csv
import json
import os
import random
import shutil
import subprocess
import tempfile
import uuid
import zipfile

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = os.path.join(ROOT, "datasets")
SR = 22050
MIN_S, MAX_S = 3.0, 11.0


def ds_dir(name):
    return os.path.join(DATASETS, name)


def create(name):
    for sub in ("raw", "wavs"):
        os.makedirs(os.path.join(ds_dir(name), sub), exist_ok=True)


def add_audio(name, src_path):
    """Convert any audio file to mono 22.05 kHz WAV inside the dataset."""
    create(name)
    dest = os.path.join(ds_dir(name), "raw", f"{uuid.uuid4().hex[:10]}.wav")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src_path, "-ac", "1", "-ar", str(SR), dest],
                   check=True)
    return dest


def _split_points(y):
    """Speech intervals merged into clips of MIN_S..MAX_S, cut at pauses."""
    import librosa
    raw_spans = librosa.effects.split(y, top_db=35, frame_length=1024, hop_length=256)
    max_n, min_n = int(MAX_S * SR), int(MIN_S * SR)
    hop = int(0.02 * SR)

    # step 1: break any span longer than MAX_S at its quietest point (between 6 s and MAX_S)
    spans = []
    for a, b in raw_spans:
        a, b = int(a), int(b)
        while b - a > max_n:
            win = y[a + int(6 * SR): a + max_n]
            e = np.array([np.abs(win[i:i + hop]).mean() for i in range(0, len(win) - hop, hop)])
            cut = a + int(6 * SR) + int(e.argmin()) * hop
            spans.append((a, cut))
            a = cut
        spans.append((a, b))

    # step 2: merge neighbouring spans into clips up to MAX_S when the pause between them is short
    clips, cur = [], None
    for a, b in spans:
        if cur is not None and b - cur[0] <= max_n and a - cur[1] <= int(0.7 * SR):
            cur = (cur[0], b)
        else:
            if cur is not None:
                clips.append(cur)
            cur = (a, b)
    if cur is not None:
        clips.append(cur)

    pad = int(0.08 * SR)
    return [(max(a - pad, 0), min(b + pad, len(y))) for a, b in clips if b - a >= min_n]


def segment(name, progress=None):
    """Cut raw audio into clips. Returns list of clip paths."""
    d = ds_dir(name)
    shutil.rmtree(os.path.join(d, "wavs"), ignore_errors=True)
    os.makedirs(os.path.join(d, "wavs"), exist_ok=True)
    raws = sorted(f for f in os.listdir(os.path.join(d, "raw")) if f.endswith(".wav"))
    paths, n = [], 0
    for i, f in enumerate(raws):
        y, _ = sf.read(os.path.join(d, "raw", f), dtype="float32")
        for a, b in _split_points(y):
            seg = y[a:b]
            peak = float(np.abs(seg).max())
            if peak > 0:
                seg = seg * (0.9 / peak)
            p = os.path.join(d, "wavs", f"{n:05d}.wav")
            sf.write(p, seg, SR)
            paths.append(p)
            n += 1
        if progress:
            progress(i + 1, len(raws))
    return paths


def transcribe(clips, f5_python, worker, progress=None, cancel=None):
    """Transcribe clips with Whisper in the separate F5/Whisper environment. Returns list of texts."""
    with tempfile.TemporaryDirectory() as tmp:
        job = os.path.join(tmp, "job.json")
        out = os.path.join(tmp, "out.json")
        json.dump({"clips": clips, "out": out}, open(job, "w", encoding="utf-8"))
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        p = subprocess.Popen([f5_python, worker, "transcribe_many", job], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, encoding="utf-8", env=env)
        for line in p.stdout:
            if cancel is not None and cancel.is_set():
                p.kill()
                raise InterruptedError()
            if line.startswith("DONE") and progress:
                progress(int(line.split()[1]) + 1, len(clips))
        err = p.stderr.read()
        p.wait()
        if p.returncode != 0 or not os.path.isfile(out):
            raise RuntimeError(f"Transcription failed: {err[-500:]}")
        return json.load(open(out, encoding="utf-8"))


def write_metadata(name, clips, texts, speaker="speaker"):
    d = ds_dir(name)
    rows = [(os.path.relpath(c, d).replace("\\", "/"), " ".join(t.split()), speaker)
            for c, t in zip(clips, texts) if len(t.split()) >= 2]
    random.Random(1).shuffle(rows)
    n_eval = max(1, len(rows) // 10) if len(rows) >= 10 else 0
    for fname, part in (("metadata_eval.csv", rows[:n_eval]), ("metadata_train.csv", rows[n_eval:])):
        with open(os.path.join(d, fname), "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="|", lineterminator="\n", quoting=csv.QUOTE_NONE, escapechar="\\")
            w.writerow(["audio_file", "text", "speaker_name"])
            w.writerows(part)
    total = sum(sf.info(os.path.join(d, r[0])).duration for r in rows)
    stats = {"segments": len(rows), "minutes": round(total / 60, 1), "eval": n_eval}
    json.dump(stats, open(os.path.join(d, "stats.json"), "w"))
    return stats


def info(name):
    d = ds_dir(name)
    raw_s = 0.0
    rd = os.path.join(d, "raw")
    if os.path.isdir(rd):
        raw_s = sum(sf.info(os.path.join(rd, f)).duration for f in os.listdir(rd) if f.endswith(".wav"))
    stats = {}
    sp = os.path.join(d, "stats.json")
    if os.path.isfile(sp):
        stats = json.load(open(sp))
    return {"name": name, "raw_minutes": round(raw_s / 60, 1), "segments": stats.get("segments", 0),
            "minutes": stats.get("minutes", 0.0), "built": bool(stats)}


def list_datasets():
    if not os.path.isdir(DATASETS):
        return []
    return [info(n) for n in sorted(os.listdir(DATASETS)) if os.path.isdir(ds_dir(n))]


def export_zip(name):
    d = ds_dir(name)
    out = os.path.join(DATASETS, f"{name}.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        for f in ("metadata_train.csv", "metadata_eval.csv"):
            z.write(os.path.join(d, f), f)
        wd = os.path.join(d, "wavs")
        used = set()
        for f in ("metadata_train.csv", "metadata_eval.csv"):
            for row in csv.DictReader(open(os.path.join(d, f), encoding="utf-8"), delimiter="|"):
                used.add(row["audio_file"])
        for rel in sorted(used):
            z.write(os.path.join(d, rel), rel)
    return out


def delete(name):
    shutil.rmtree(ds_dir(name), ignore_errors=True)
    z = os.path.join(DATASETS, f"{name}.zip")
    if os.path.isfile(z):
        os.remove(z)
