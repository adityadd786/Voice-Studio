"""Voice cloning engine + CLI.

Example:
  python src/clone.py --ref refs/me.wav --lang en --speed 1.1 --casual \
      --text "Hello. [pause 0.6s] [speed 0.9] [pitch -1] Slower, deeper part." --out out/test.wav
Only clone voices you own or have consent to use.
"""
import argparse
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(__file__))
from pace import (TECH_TERMS, apply_casual, apply_indian_text, apply_pronunciations,  # noqa: E402
                  count_sentences, normalize_loudness, parse_script, pitch_shift, silence,
                  smart_adjust, speed_for_duration, split_sentences, time_stretch, trim_silence)

XTTS_LANGS = ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar",
              "zh-cn", "ja", "hu", "ko", "hi"]


class Cancelled(Exception):
    pass


@dataclass
class Options:
    lang: str = "en"
    engine: str = "xtts"
    speed: float = 1.0
    gap: float = 0.25            # between sentences
    para_gap: float = 0.6        # between paragraphs
    duration: float | None = None  # fit whole output to N seconds
    # expressiveness (XTTS sampling)
    temperature: float = 0.75
    top_p: float = 0.85
    top_k: int = 50
    repetition_penalty: float = 10.0
    seed: int | None = None
    ref_seconds: int = 30        # how much of the reference clip to analyse (6-30)
    # post-processing
    pitch: float = 0.0           # semitones
    gain_db: float = 0.0
    normalize: bool = False
    trim: bool = True
    # text shaping
    casual: bool = False
    indian_text: bool = False    # ₹ amounts in lakh/crore
    tech_terms: bool = False     # built-in acronym/tech dictionary
    smart_pace: bool = False     # teaching-style pauses and slower definitions
    pronunciations: list = field(default_factory=list)  # [(find, say), ...]


_XTTS = None
_LATENTS = {}


def load_xtts():
    global _XTTS
    if _XTTS is None:
        import torch
        from TTS.api import TTS
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _XTTS = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(dev)
    return _XTTS


def latents_for(ref, ref_seconds):
    """Voice analysis is slow; cache per reference file."""
    key = (ref, os.path.getmtime(ref), ref_seconds)
    if key not in _LATENTS:
        model = load_xtts().synthesizer.tts_model
        _LATENTS[key] = model.get_conditioning_latents(
            audio_path=[ref], gpt_cond_len=ref_seconds, gpt_cond_chunk_len=6, max_ref_length=ref_seconds)
    return _LATENTS[key]


def synth_xtts(text, ref, lang, speed, o: Options):
    model = load_xtts().synthesizer.tts_model
    gpt, spk = latents_for(ref, o.ref_seconds)
    out = model.inference(
        text, lang, gpt, spk, temperature=o.temperature, top_p=o.top_p, top_k=o.top_k,
        repetition_penalty=o.repetition_penalty, speed=speed, enable_text_splitting=False)
    return np.asarray(out["wav"], dtype=np.float32), 24000


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F5_PY = os.path.join(ROOT, ".venv-f5", "Scripts", "python.exe")
F5_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "f5_worker.py")
F5_CACHE = os.path.join(ROOT, "cache", "f5")


def f5_available():
    return os.path.isfile(F5_PY)


def _run_worker(args, **kw):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_DISABLE_SYMLINKS_WARNING="1")
    return subprocess.Popen([F5_PY, F5_WORKER, *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8", env=env, **kw)


def prepare_f5_ref(path):
    """F5 reads only ~12 s of reference and needs its exact transcript, so build a clean clip of
    at most 10 s and transcribe that clip. Cached per voice file."""
    from analysis import clean_reference
    os.makedirs(F5_CACHE, exist_ok=True)
    key = f"{os.path.splitext(os.path.basename(path))[0]}_{int(os.path.getmtime(path))}"
    clip, txt = os.path.join(F5_CACHE, key + ".wav"), os.path.join(F5_CACHE, key + ".txt")
    if not (os.path.isfile(clip) and os.path.isfile(txt)):
        clean_reference(path, clip, max_seconds=10, fit=True)
        p = _run_worker(["transcribe", clip, txt])
        _, err = p.communicate()
        if p.returncode != 0 or not os.path.isfile(txt):
            raise RuntimeError(f"Transcribing the reference failed: {err[-400:]}")
    return clip, open(txt, encoding="utf-8").read().strip()


def synth_f5_batch(plan, resolve_voice, default_voice, o, seed, progress, cancel):
    """Synthesize all sentences with F5-TTS in the separate environment. Returns ({plan_idx: wav}, sr)."""
    import json
    import tempfile
    if not f5_available():
        raise RuntimeError("F5-TTS is not installed (missing .venv-f5).")
    by_voice = {}
    for i, st in enumerate(plan):
        if st[0] == "sent":
            if st[3] != "en":
                raise ValueError(f"F5-TTS supports English only (found [{st[3]}]). Use the XTTS engine for other languages.")
            by_voice.setdefault(st[4] or default_voice, []).append(i)
    total = sum(len(v) for v in by_voice.values()) or 1
    audio, done, sr = {}, 0, 24000
    for voice, idxs in by_voice.items():
        clip, ref_text = prepare_f5_ref(resolve_voice(voice))
        with tempfile.TemporaryDirectory() as tmp:
            job = {"ref": clip, "ref_text": ref_text, "out_dir": tmp, "seed": seed, "nfe_step": 16,
                   "items": [{"text": plan[i][1], "speed": plan[i][2]} for i in idxs]}
            jp = os.path.join(tmp, "job.json")
            json.dump(job, open(jp, "w", encoding="utf-8"))
            p = _run_worker(["synth", jp])
            for line in p.stdout:
                if cancel is not None and cancel.is_set():
                    p.kill()
                    raise Cancelled()
                if line.startswith("DONE"):
                    done += 1
                    if progress:
                        progress(done, total)
            err = p.stderr.read()
            p.wait()
            if p.returncode != 0:
                raise RuntimeError(f"F5-TTS failed: {err[-600:]}")
            for n, i in enumerate(idxs):
                w, sr = sf.read(os.path.join(tmp, f"{n}.wav"), dtype="float32")
                audio[i] = w if w.ndim == 1 else w.mean(axis=1)
    return audio, sr


def generate(script, resolve_voice, default_voice, o: Options, progress=None, cancel=None, ref_text=""):
    """Returns (wav, sr, seed). resolve_voice(name)->path. progress(done,total)."""
    import torch
    seed = o.seed if o.seed is not None else random.randint(0, 2**31 - 1)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    items = parse_script(script, o.speed, o.lang, default_voice, o.pitch, o.para_gap)

    # 1) plan: ("pause", s) | ("sent", text, speed, lang, voice, pitch, extra_pause)
    plan = []
    for item in items:
        if item[0] == "pause":
            plan.append(item)
            continue
        _, text, spd, lang, voice, pitch = item
        if o.casual and lang == "en":
            text = apply_casual(text)
        if o.indian_text:
            text = apply_indian_text(text)
        # user entries win over the built-in dictionary
        text = apply_pronunciations(text, list(o.pronunciations) + (TECH_TERMS if o.tech_terms else []))
        for sent in split_sentences(text):
            extra, mult = smart_adjust(sent) if o.smart_pace else (0.0, 1.0)
            plan.append(("sent", sent, max(0.3, min(spd * mult, 3.0)), lang, voice, pitch, extra))

    # 2) synthesize
    total = max(sum(1 for s in plan if s[0] == "sent"), 1)
    sr, audio = 24000, {}
    if o.engine == "f5":
        audio, sr = synth_f5_batch(plan, resolve_voice, default_voice, o, seed, progress, cancel)
    else:
        done = 0
        for i, st in enumerate(plan):
            if st[0] != "sent":
                continue
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            _, sent, sspd, lang, voice, _p, _e = st
            audio[i], sr = synth_xtts(sent, resolve_voice(voice or default_voice), lang, sspd, o)
            done += 1
            if progress:
                progress(done, total)

    # 3) assemble
    pieces = []
    for i, st in enumerate(plan):
        if st[0] == "pause":
            pieces.append(silence(st[1], sr))
            continue
        w = audio[i]
        if o.trim:
            w = trim_silence(w, sr, keep=0.08)
        pieces.append(pitch_shift(w, sr, st[5]))
        pieces.append(silence(o.gap + st[6], sr))

    wav = np.concatenate(pieces) if pieces else np.zeros(1, np.float32)
    if o.duration:
        wav = time_stretch(wav, sr, speed_for_duration(wav, sr, o.duration))
    if o.normalize:
        wav = normalize_loudness(wav, sr)
    if o.gain_db:
        wav = np.clip(wav * (10 ** (o.gain_db / 20)), -1, 1)
    return wav, sr, seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["xtts", "f5"], default="xtts")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--ref-text", default="")
    ap.add_argument("--text", required=True)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--gap", type=float, default=0.25)
    ap.add_argument("--duration", type=float)
    ap.add_argument("--temperature", type=float, default=0.75)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--pitch", type=float, default=0.0)
    ap.add_argument("--casual", action="store_true")
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--out", default="out/output.wav")
    a = ap.parse_args()

    o = Options(lang=a.lang, engine=a.engine, speed=a.speed, gap=a.gap, duration=a.duration,
                temperature=a.temperature, seed=a.seed, pitch=a.pitch, casual=a.casual, normalize=a.normalize)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    t0 = time.time()
    wav, sr, seed = generate(a.text, lambda _n: a.ref, None, o, ref_text=a.ref_text)
    sf.write(a.out, wav, sr)
    dur = len(wav) / sr
    print(f"saved {a.out} | {dur:.1f}s audio in {time.time()-t0:.1f}s | seed {seed}")


if __name__ == "__main__":
    main()
