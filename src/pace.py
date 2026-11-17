"""Script parsing, text shaping and audio post-processing helpers.

Inline markers (case-insensitive):
  [pause 0.5s]     silence
  [speed 0.9]      pace for following text
  [pitch +2]       pitch shift in semitones for following text
  [lang hi]        language for following text
  [voice name]     switch cloned voice for following text
Blank lines are paragraph breaks (longer pause).
"""
import re
import subprocess

import numpy as np

MARKER = re.compile(r"\[(pause|speed|pitch|lang|voice)\s+([^\]]+?)\]", re.I)

# "Casual speech": written forms that XTTS reads the way people actually say them.
CASUAL = [
    (r"\bgoing to\b", "gonna"), (r"\bwant to\b", "wanna"), (r"\bgot to\b", "gotta"),
    (r"\bkind of\b", "kinda"), (r"\bsort of\b", "sorta"), (r"\blot of\b", "lotta"),
    (r"\bI don't know\b", "I dunno"), (r"\bgive me\b", "gimme"), (r"\blet me\b", "lemme"),
    (r"\bout of\b", "outta"), (r"\bdo you\b", "d'you"), (r"\byes\b", "yeah"),
    (r"\bhave to\b", "hafta"), (r"\bthem\b", "'em"),
]


def apply_casual(text):
    for pat, rep in CASUAL:
        text = re.sub(pat, rep, text, flags=re.I)
    return text


def apply_pronunciations(text, pairs):
    """pairs: list of (find, say). Whole-word, case-insensitive."""
    for find, say in pairs:
        if find.strip():
            text = re.sub(rf"(?<!\w){re.escape(find.strip())}(?!\w)", say, text, flags=re.I)
    return text


def parse_script(text, default_speed=1.0, default_lang="en", default_voice=None,
                 default_pitch=0.0, para_gap=0.6):
    """Return items: ('text', str, speed, lang, voice, pitch) | ('pause', seconds)."""
    items = []
    state = {"speed": default_speed, "lang": default_lang, "voice": default_voice, "pitch": default_pitch}
    pos = 0

    def emit_text(chunk):
        # blank lines become paragraph pauses
        paras = re.split(r"\n\s*\n", chunk)
        for i, para in enumerate(paras):
            para = " ".join(para.split())
            if para:
                items.append(("text", para, state["speed"], state["lang"], state["voice"], state["pitch"]))
            if i < len(paras) - 1 and para and para_gap > 0:
                items.append(("pause", para_gap))

    for m in MARKER.finditer(text):
        emit_text(text[pos:m.start()])
        kind, val = m.group(1).lower(), m.group(2).strip()
        try:
            if kind == "pause":
                items.append(("pause", float(val.rstrip("sS"))))
            elif kind == "speed":
                state["speed"] = max(0.3, min(float(val), 3.0))
            elif kind == "pitch":
                state["pitch"] = max(-12.0, min(float(val), 12.0))
            elif kind == "lang":
                state["lang"] = val.lower()
            elif kind == "voice":
                state["voice"] = val
        except ValueError:
            pass  # ignore malformed marker
        pos = m.end()
    emit_text(text[pos:])
    return items


def split_sentences(text, max_chars=220):
    """Split into sentence-sized chunks to keep VRAM low."""
    parts = re.split(r"(?<=[.!?।。！？])\s+", text.strip())
    out, cur = [], ""
    for p in parts:
        if cur and len(cur) + len(p) + 1 > max_chars:
            out.append(cur)
            cur = p
        else:
            cur = f"{cur} {p}".strip()
    if cur:
        out.append(cur)
    return out


def count_sentences(items):
    return sum(len(split_sentences(i[1])) for i in items if i[0] == "text")


def silence(seconds, sr):
    return np.zeros(int(seconds * sr), dtype=np.float32)


def _ffmpeg_filter(wav, sr, flt):
    pcm = (np.clip(wav, -1, 1) * 32767).astype("<i2").tobytes()
    cmd = ["ffmpeg", "-loglevel", "error", "-f", "s16le", "-ar", str(sr), "-ac", "1",
           "-i", "-", "-filter:a", flt, "-f", "s16le", "-"]
    out = subprocess.run(cmd, input=pcm, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype="<i2").astype(np.float32) / 32767


def time_stretch(wav, sr, factor):
    """Pitch-preserving stretch. factor>1 = faster."""
    if abs(factor - 1.0) < 0.01:
        return wav
    filters, f = [], factor
    while f > 2.0:
        filters.append("atempo=2.0"); f /= 2.0
    while f < 0.5:
        filters.append("atempo=0.5"); f /= 0.5
    filters.append(f"atempo={f:.4f}")
    return _ffmpeg_filter(wav, sr, ",".join(filters))


def pitch_shift(wav, sr, semitones):
    """Shift pitch in semitones, keeping duration."""
    if abs(semitones) < 0.05 or len(wav) == 0:
        return wav
    r = 2 ** (semitones / 12)
    return _ffmpeg_filter(wav, sr, f"asetrate={int(sr * r)},aresample={sr},atempo={1 / r:.5f}")


def normalize_loudness(wav, sr):
    # loudnorm upsamples to 192 kHz internally; resample back or duration is wrong
    return _ffmpeg_filter(wav, sr, f"loudnorm=I=-16:TP=-1.5:LRA=11,aresample={sr}")


def trim_silence(wav, sr, threshold_db=-45, keep=0.15):
    """Trim leading/trailing silence, keeping a short margin."""
    amp = 10 ** (threshold_db / 20)
    idx = np.where(np.abs(wav) > amp)[0]
    if len(idx) == 0:
        return wav
    k = int(keep * sr)
    return wav[max(idx[0] - k, 0): idx[-1] + k]


def speed_for_duration(wav, sr, target_seconds):
    return (len(wav) / sr) / target_seconds
