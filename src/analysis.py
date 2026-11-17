"""Acoustic analysis of a reference voice clip (no ML models needed)."""
import numpy as np
import soundfile as sf


def _db(x, floor=1e-8):
    return 20 * np.log10(np.maximum(x, floor))


def clean_reference(src, dest, max_seconds=30, gap=0.15, top_db=38):
    """Cut digital silence and long pauses so the clip is continuous speech.
    Keeps natural short pauses (<= `gap` seconds), stops at `max_seconds`. Returns seconds kept."""
    import librosa

    y, sr = sf.read(src, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    spans = librosa.effects.split(y, top_db=top_db, frame_length=1024, hop_length=256)
    pad = int(0.05 * sr)
    # faint room-tone instead of exact zeros, which models and detectors treat as an artefact
    joiner = (np.random.default_rng(0).standard_normal(int(gap * sr)) * 3e-4).astype("float32")
    pieces, total = [], 0
    for a, b in spans:
        seg = y[max(a - pad, 0): min(b + pad, len(y))]
        if len(seg) < int(0.12 * sr):   # ignore clicks and breaths
            continue
        pieces += [seg, joiner]
        total += len(seg) + len(joiner)
        if total >= max_seconds * sr:
            break
    if not pieces:
        raise ValueError("No speech found in this clip")
    out = np.concatenate(pieces)[: int(max_seconds * sr)]
    peak = float(np.abs(out).max())
    if peak > 0:
        out = out * (0.7 / peak)        # about -3 dBFS
    sf.write(dest, out, sr)
    return len(out) / sr


def analyze_voice(path):
    import librosa

    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr > 16000:  # analysis does not need more; also much faster
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr_orig, sr = sr, 16000
    else:
        sr_orig = sr
    duration = len(y) / sr

    hop, win = int(0.01 * sr), int(0.03 * sr)
    frames = librosa.util.frame(y, frame_length=win, hop_length=hop)
    rms = np.sqrt((frames ** 2).mean(axis=0))
    rms_db = _db(rms)

    # Frames that are exact digital silence (noise gate / editing) say nothing about the
    # room noise, so measure noise and level on the remaining frames only.
    digital_silence = rms_db < -90
    live = rms_db[~digital_silence]
    if len(live) < 10:
        live = rms_db
    noise_floor = float(np.percentile(live, 10))
    speech_level = float(np.percentile(live, 90))
    snr = speech_level - noise_floor
    voiced = (rms_db > (noise_floor + 12)) & ~digital_silence

    # pauses = silent runs longer than 300 ms
    pauses, run = 0, 0
    for v in voiced:
        if not v:
            run += 1
        else:
            if run * 0.01 >= 0.3:
                pauses += 1
            run = 0

    # pitch on voiced speech only (yin is fast; pyin is more accurate but slow on CPU)
    f0 = librosa.yin(y, fmin=65, fmax=400, sr=sr, frame_length=1024, hop_length=hop)
    n = min(len(f0), len(voiced))
    f0v = f0[:n][voiced[:n]]
    f0v = f0v[(f0v > 70) & (f0v < 390)]
    pitch = {}
    if len(f0v) > 20:
        pitch = {
            "median_hz": round(float(np.median(f0v))),
            "low_hz": round(float(np.percentile(f0v, 10))),
            "high_hz": round(float(np.percentile(f0v, 90))),
        }
        med = pitch["median_hz"]
        pitch["register"] = "low (typical male)" if med < 135 else "mid" if med < 175 else "high (typical female)"
        pitch["variation"] = round(float(np.std(12 * np.log2(f0v / np.median(f0v)))), 1)  # semitones

    report = {
        "duration_s": round(duration, 1),
        "sample_rate": sr_orig,
        "digital_silence_pct": round(float(digital_silence.mean() * 100)),
        "peak_db": round(float(_db(np.abs(y).max())), 1),
        "loudness_db": round(speech_level, 1),
        "noise_floor_db": round(noise_floor, 1),
        "snr_db": round(float(snr), 1),
        "clipping_pct": round(float((np.abs(y) > 0.99).mean() * 100), 3),
        "speech_pct": round(float(voiced.mean() * 100)),
        "long_pauses": pauses,
        "pitch": pitch,
    }

    tips = []
    if duration < 8:
        tips.append("Clip is short. 15-30 seconds gives a closer clone.")
    elif duration > 40:
        tips.append("Only the first 30 seconds are analysed by XTTS. Put your best, clearest 15-30 seconds first.")
    if snr < 25:
        tips.append("Background noise is fairly high. Record in a quieter room or denoise the clip; noise is copied into the clone.")
    if report["clipping_pct"] > 0.05:
        tips.append("The recording is clipping (too loud). Record again with the mic further away.")
    if report["peak_db"] < -20:
        tips.append("The recording is very quiet. Increase mic level or normalise it.")
    if report["digital_silence_pct"] > 10:
        tips.append(
            f"{report['digital_silence_pct']}% of the clip is exact digital silence (a noise gate or editing). "
            "Cloning models copy that choppiness. Use a continuous, un-gated recording or cut the silent parts out.")
    if report["speech_pct"] < 60:
        tips.append("Lots of silence in the clip. Trim pauses so more of it is speech.")
    if not tips:
        tips.append("Recording quality looks good for cloning.")
    report["tips"] = tips
    return report
