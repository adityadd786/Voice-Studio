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


# Tech / course terms that TTS usually mangles. Used when "Tech & course terms" is on.
TECH_TERMS = [
    ("API", "A P I"), ("APIs", "A P Is"), ("SQL", "sequel"), ("NoSQL", "no sequel"), ("AWS", "A W S"),
    ("CPU", "C P U"), ("GPU", "G P U"), ("RAM", "ram"), ("UI", "U I"), ("UX", "U X"),
    ("HTML", "H T M L"), ("CSS", "C S S"), ("JSON", "jay-son"), ("URL", "U R L"), ("HTTP", "H T T P"),
    ("HTTPS", "H T T P S"), ("IDE", "I D E"), ("AI", "A I"), ("ML", "M L"), ("LLM", "L L M"),
    ("OTP", "O T P"), ("PDF", "P D F"), ("FAQ", "F A Q"), ("GST", "G S T"), ("PAN", "pan"),
    ("UPI", "U P I"), ("KYC", "K Y C"), ("IIT", "I I T"), ("CEO", "C E O"), ("ROI", "R O I"),
    ("GitHub", "git hub"), ("Linux", "linnux"), ("Kubernetes", "koo-ber-net-eez"), ("nginx", "engine X"),
    ("IdentityIQ", "Identity I Q"), ("Tomcat", "tom cat"), ("e.g.", "for example"), ("i.e.", "that is"),
    ("vs", "versus"), ("etc.", "et cetera"),
]

_AMOUNT = re.compile(
    r"(?:₹|Rs\.?|INR)\s?(\d(?:[\d,]*\d)?(?:\.\d+)?)(?:\s?(lakhs?|crores?|cr|thousand|k)\b)?", re.I)


def _amount(m):
    from num2words import num2words
    num = float(m.group(1).replace(",", ""))
    unit = (m.group(2) or "").lower()
    unit_word = {"lakh": "lakh", "lakhs": "lakh", "crore": "crore", "crores": "crore", "cr": "crore",
                 "thousand": "thousand", "k": "thousand"}.get(unit, "")
    n = int(num) if num == int(num) else num
    return f"{num2words(n, lang='en_IN')} {unit_word} rupees".replace("  ", " ")


def apply_indian_text(text):
    """Read ₹/Rs amounts the Indian way: ₹5,00,000 -> five lakh rupees."""
    return _AMOUNT.sub(_amount, text)


_CUES = re.compile(
    r"\b(for example|for instance|in other words|is called|means|definition|remember|important|"
    r"note that|step \d+|first|second|third|finally|key point)\b", re.I)


def smart_adjust(sentence):
    """Teaching-style pacing: returns (extra_pause_seconds, speed_multiplier)."""
    extra, mult = 0.0, 1.0
    s = sentence.strip()
    if s.endswith("?"):
        extra = 0.35          # let a question land
    elif s.endswith(":"):
        extra = 0.3
    if "..." in s or "…" in s:
        extra = max(extra, 0.4)
    if _CUES.search(s):
        mult = 0.93           # slow slightly for definitions and key points
    if len(s.split()) <= 3:
        extra = max(extra, 0.12)
    return extra, mult


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


def split_sentences(text, max_chars=220, min_chars=10):
    """One chunk per sentence so pauses and pacing apply per sentence.
    Very short sentences ("Okay.") are joined to the next one (tiny inputs make XTTS glitch);
    sentences over max_chars are split at commas to keep VRAM low."""
    parts = [p for p in re.split(r"(?<=[.!?।。！？])\s+", text.strip()) if p]
    sized = []
    for p in parts:
        while len(p) > max_chars:
            cut = p.rfind(",", 0, max_chars)
            if cut < max_chars // 3:
                cut = p.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            sized.append(p[:cut + 1].strip())
            p = p[cut + 1:].strip()
        if p:
            sized.append(p)
    out, carry = [], ""
    for p in sized:
        p = f"{carry} {p}".strip() if carry else p
        carry = ""
        if len(p) < min_chars:
            carry = p
        else:
            out.append(p)
    if carry:
        if out:
            out[-1] = f"{out[-1]} {carry}"
        else:
            out.append(carry)
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
