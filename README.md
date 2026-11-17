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


