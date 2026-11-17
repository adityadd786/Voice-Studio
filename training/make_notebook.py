"""Generate xtts_finetune_colab.ipynb from train_xtts.py so the two never drift apart.
Run:  python training/make_notebook.py
"""
import json
import os

here = os.path.dirname(os.path.abspath(__file__))
script = open(os.path.join(here, "train_xtts.py"), encoding="utf-8").read()


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


cells = [
    md("""
# Fine-tune XTTS-v2 on your voice (Voice Studio)

1. **Runtime -> Change runtime type -> T4 GPU** (free).
2. Run the cells in order.
3. When asked, upload the dataset zip you downloaded from Voice Studio's *Fine-tune* tab.
4. Download `finetuned_xtts.zip` at the end and import it in Voice Studio.

Only train on a voice you own or have permission to use. Aim for 10-30 minutes of clean speech.
"""),
    code("!nvidia-smi"),
    code("!pip install -q coqui-tts huggingface_hub"),
    code("""
from google.colab import files
import os, zipfile
up = files.upload()                      # choose your dataset zip
name = next(iter(up))
os.makedirs('/content/dataset', exist_ok=True)
zipfile.ZipFile(name).extractall('/content/dataset')
print(os.listdir('/content/dataset'))
"""),
    code("%%writefile train_xtts.py\n" + script),
    code("!python train_xtts.py --data /content/dataset --out /content/run --epochs 10"),
    md("""
## Get the model back
`finetuned_xtts.zip` is about 1.9 GB. Downloading through the browser can be slow; copying it to Google Drive
first is more reliable.
"""),
    code("""
from google.colab import files
files.download('/content/run/finetuned_xtts.zip')
"""),
    code("""
# Alternative: copy to Google Drive, then download from drive.google.com
from google.colab import drive
drive.mount('/content/drive')
!cp /content/run/finetuned_xtts.zip /content/drive/MyDrive/
"""),
]

nb = {
    "cells": cells,
    "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                 "kernelspec": {"display_name": "Python 3", "name": "python3"}},
    "nbformat": 4, "nbformat_minor": 5,
}
out = os.path.join(here, "xtts_finetune_colab.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("wrote", out)
