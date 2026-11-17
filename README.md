# Voice Studio

Local voice cloning (text-to-speech) with pace control, built for a 4 GB GPU.

## Setup (Windows, Python 3.11)
```
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt six
```
ffmpeg must be on PATH.

## Run the web app (shadcn/ui)
Two terminals:
```
# 1) API (from project root, venv active)
$env:COQUI_TOS_AGREED="1"   # PowerShell
uvicorn server:app --app-dir src --port 8010

# 2) Frontend
cd web
pnpm install
pnpm dev
```
Open http://localhost:5173. Use pnpm, not npm: the shadcn CLI fails under npm 11.

## Fine-tuning your own voice (accent and style)
A single reference clip copies your timbre but not your accent well. Fine-tuning does. Training needs a
16 GB GPU, so it runs in free Google Colab; the app prepares the data and imports the result.

1. **Fine-tune tab -> Collect:** upload recordings or record the on-screen sentences. Aim for 10-30 minutes of
   clean speech (no noise gate, no noise cancelling).
2. **Build dataset:** cuts 3-11 s clips and transcribes them locally with Whisper (needs the `.venv-f5` environment).
3. **Download** the dataset zip and `training/xtts_finetune_colab.ipynb`, open the notebook in Colab with a T4 GPU, run it.
4. **Import** the resulting `finetuned_xtts.zip`, then pick it under Settings -> Voice & expression -> Model.

`training/train_xtts.py` has a `--dry-run` mode that checks the setup without training. Regenerate the notebook with
`python training/make_notebook.py`. Datasets, models and zips are git-ignored because they contain your voice.

## Other engine: F5-TTS
`.venv-f5` is a separate environment (`pip install f5-tts faster-whisper` with CUDA PyTorch). F5 follows accent more
closely but is very slow on a 4 GB GPU (about 10x slower than real time).

## CLI usage
```
python src/clone.py --engine xtts --ref refs/me.wav --lang en --speed 1.0 \
  --text "Hello. [pause 0.6s] [speed 0.85] Slower part. [speed 1.15] Faster part." \
  --out out/test.wav
```
- `[pause 0.5s]` inserts silence, `[speed 0.9]` changes pace for the text after it.
- `[pitch +2]` shifts pitch (semitones), `[lang hi]` switches language, `[voice name]` switches voice.
- `--duration N` stretches the result to N seconds (pitch preserved).
- Put reference clips (10-30 s, one clean speaker) in `refs/`. They are git-ignored.

## Notes
- XTTS-v2 uses the Coqui CPML license (non-commercial). Set `COQUI_TOS_AGREED=1` to skip the prompt.
- Measured on GTX 1650 Ti: about 2 GB peak VRAM, roughly 2x slower than real time.
- Only clone voices you own or have consent to use.


