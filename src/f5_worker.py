"""F5-TTS worker. Runs inside the separate .venv-f5 environment (different dependency set from XTTS).

  python f5_worker.py transcribe REF.wav OUT.txt
  python f5_worker.py synth JOB.json
JOB.json: {"ref": wav, "ref_text": str, "out_dir": dir, "seed": int, "nfe_step": int,
           "items": [{"text": str, "speed": float}, ...]}
Prints "DONE <index>" after each item so the caller can show progress.
"""
import json
import os
import sys


def transcribe(ref, out):
    from faster_whisper import WhisperModel
    model = WhisperModel("small", device="cpu", compute_type="int8")
    segs, _ = model.transcribe(ref, language="en", beam_size=5)
    text = " ".join(s.text.strip() for s in segs).strip()
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print("TEXT", text, flush=True)


def synth(job_path):
    import soundfile as sf
    import torch
    from f5_tts.api import F5TTS

    job = json.load(open(job_path, encoding="utf-8"))
    tts = F5TTS(model="F5TTS_v1_Base", device="cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(job["out_dir"], exist_ok=True)
    for i, it in enumerate(job["items"]):
        wav, sr, _ = tts.infer(
            ref_file=job["ref"], ref_text=job["ref_text"], gen_text=it["text"], speed=it["speed"],
            nfe_step=job.get("nfe_step", 32), seed=job.get("seed"), remove_silence=False)
        sf.write(os.path.join(job["out_dir"], f"{i}.wav"), wav, sr)
        print("DONE", i, flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    if sys.argv[1] == "transcribe":
        transcribe(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "synth":
        synth(sys.argv[2])
