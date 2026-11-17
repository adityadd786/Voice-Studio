"""Voice cloning engine + CLI.

Example:
  python src/clone.py --ref refs/me.wav --lang en --speed 1.1 --casual \
      --text "Hello. [pause 0.6s] [speed 0.9] [pitch -1] Slower, deeper part." --out out/test.wav
Only clone voices you own or have consent to use.
"""
import argparse
import os
import random
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


def synth_f5(text, ref, speed, ref_text=""):
    from f5_tts.api import F5TTS
    global _F5
    try:
        _F5
    except NameError:
        _F5 = F5TTS()
    wav, sr, _ = _F5.infer(ref_file=ref, ref_text=ref_text, gen_text=text, speed=speed)
    return np.array(wav, dtype=np.float32), sr


def generate(script, resolve_voice, default_voice, o: Options, progress=None, cancel=None, ref_text=""):
    """Returns (wav, sr, seed). resolve_voice(name)->path. progress(done,total)."""
    import torch
    seed = o.seed if o.seed is not None else random.randint(0, 2**31 - 1)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    items = parse_script(script, o.speed, o.lang, default_voice, o.pitch, o.para_gap)
    total, done, sr, pieces = max(count_sentences(items), 1), 0, 24000, []
    for item in items:
        if item[0] == "pause":
            pieces.append(silence(item[1], sr))
            continue
        _, text, spd, lang, voice, pitch = item
        ref = resolve_voice(voice or default_voice)
        if o.casual and lang == "en":
            text = apply_casual(text)
        if o.indian_text:
            text = apply_indian_text(text)
        # user entries win over the built-in dictionary
        text = apply_pronunciations(text, list(o.pronunciations) + (TECH_TERMS if o.tech_terms else []))
        for sent in split_sentences(text):
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            extra, mult = smart_adjust(sent) if o.smart_pace else (0.0, 1.0)
            sspd = max(0.3, min(spd * mult, 3.0))
            if o.engine == "f5":
                w, sr = synth_f5(sent, ref, sspd, ref_text)
            else:
                w, sr = synth_xtts(sent, ref, lang, sspd, o)
            if o.trim:
                w = trim_silence(w, sr, keep=0.08)
            pieces.append(pitch_shift(w, sr, pitch))
            pieces.append(silence(o.gap + extra, sr))
            done += 1
            if progress:
                progress(done, total)

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
