"""Text parsing and pace control helpers.

Supported inline markers in input text:
  [pause 0.5s]   insert silence
  [speed 0.9]    change speed for following text (until next marker)
"""
import re
import subprocess
import numpy as np

MARKER = re.compile(r"\[(pause|speed)\s+([0-9.]+)s?\]", re.I)


def parse_script(text, default_speed=1.0):
    """Return list of ('text', str, speed) / ('pause', seconds) items."""
    items, speed, pos = [], default_speed, 0
    for m in MARKER.finditer(text):
        chunk = text[pos:m.start()].strip()
        if chunk:
            items.append(("text", chunk, speed))
        kind, val = m.group(1).lower(), float(m.group(2))
        if kind == "pause":
            items.append(("pause", val))
        else:
            speed = val
        pos = m.end()
    tail = text[pos:].strip()
    if tail:
        items.append(("text", tail, speed))
    return items


def split_sentences(text, max_chars=220):
    """Split into sentence-sized chunks to keep VRAM low."""
    parts = re.split(r"(?<=[.!?।])\s+", text.strip())
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


def silence(seconds, sr):
    return np.zeros(int(seconds * sr), dtype=np.float32)


def time_stretch(wav, sr, factor):
    """Pitch-preserving stretch via ffmpeg atempo. factor>1 = faster."""
    if abs(factor - 1.0) < 0.01:
        return wav
    filters, f = [], factor
    while f > 2.0:
        filters.append("atempo=2.0"); f /= 2.0
    while f < 0.5:
        filters.append("atempo=0.5"); f /= 0.5
    filters.append(f"atempo={f:.4f}")
    pcm = (np.clip(wav, -1, 1) * 32767).astype("<i2").tobytes()
    cmd = ["ffmpeg", "-loglevel", "error", "-f", "s16le", "-ar", str(sr), "-ac", "1",
           "-i", "-", "-filter:a", ",".join(filters), "-f", "s16le", "-"]
    out = subprocess.run(cmd, input=pcm, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype="<i2").astype(np.float32) / 32767


def speed_for_duration(wav, sr, target_seconds):
    """Speed factor needed so wav fits target_seconds."""
    return (len(wav) / sr) / target_seconds
