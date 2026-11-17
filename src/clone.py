"""Voice cloning CLI.

Example:
  python src/clone.py --engine xtts --ref refs/me.wav --lang en --speed 1.1 \
      --text "Hello there. [pause 0.6s] [speed 0.9] This part is slower." --out out/test.wav
Only clone voices you own or have consent to use.
"""
import argparse
import os
import sys
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(__file__))
from pace import parse_script, split_sentences, silence, time_stretch, speed_for_duration  # noqa: E402

_XTTS = None


def load_xtts():
    global _XTTS
    if _XTTS is None:
        import torch
        from TTS.api import TTS
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _XTTS = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(dev)
    return _XTTS


def synth_xtts(text, ref, lang, speed):
    tts = load_xtts()
    wav = tts.tts(text=text, speaker_wav=ref, language=lang, speed=speed)
    return np.array(wav, dtype=np.float32), 24000


def synth_f5(text, ref, lang, speed, ref_text=""):
    from f5_tts.api import F5TTS
    global _F5
    try:
        _F5
    except NameError:
        _F5 = F5TTS()
    wav, sr, _ = _F5.infer(ref_file=ref, ref_text=ref_text, gen_text=text, speed=speed)
    return np.array(wav, dtype=np.float32), sr


def generate(text, ref, engine="xtts", lang="en", speed=1.0, gap=0.25,
             target_seconds=None, ref_text=""):
    pieces, sr = [], 24000
    for item in parse_script(text, speed):
        if item[0] == "pause":
            pieces.append(silence(item[1], sr))
            continue
        _, chunk_text, spd = item
        for sent in split_sentences(chunk_text):
            if engine == "xtts":
                w, sr = synth_xtts(sent, ref, lang, spd)
            else:
                w, sr = synth_f5(sent, ref, lang, spd, ref_text)
            pieces.append(w)
            pieces.append(silence(gap, sr))
    wav = np.concatenate(pieces) if pieces else np.zeros(1, np.float32)
    if target_seconds:
        wav = time_stretch(wav, sr, speed_for_duration(wav, sr, target_seconds))
    return wav, sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["xtts", "f5"], default="xtts")
    ap.add_argument("--ref", required=True, help="reference voice wav")
    ap.add_argument("--ref-text", default="", help="transcript of ref (F5)")
    ap.add_argument("--text", required=True)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--gap", type=float, default=0.25, help="seconds between sentences")
    ap.add_argument("--duration", type=float, help="force total length in seconds")
    ap.add_argument("--out", default="out/output.wav")
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    t0 = time.time()
    wav, sr = generate(a.text, a.ref, a.engine, a.lang, a.speed, a.gap, a.duration, a.ref_text)
    sf.write(a.out, wav, sr)
    dur = len(wav) / sr
    print(f"saved {a.out} | {dur:.1f}s audio in {time.time()-t0:.1f}s (RTF {(time.time()-t0)/dur:.2f})")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"peak VRAM {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    except Exception:
        pass


if __name__ == "__main__":
    main()
