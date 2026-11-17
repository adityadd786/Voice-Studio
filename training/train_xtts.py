"""Fine-tune XTTS-v2 on your own voice dataset. Meant for a 16 GB GPU (e.g. a free Colab T4).

Input : a folder with metadata_train.csv, metadata_eval.csv and wavs/ (made by Voice Studio's
        Fine-tune tab: "audio_file|text|speaker_name").
Output: <out>/export/{model.pth, config.json, vocab.json, speakers_xtts.pth} and <out>/finetuned_xtts.zip
        Import the zip in Voice Studio (Fine-tune tab -> Import model).

  python train_xtts.py --data /content/dataset --out /content/run --epochs 8
Only train on voices you own or have permission to use.
"""
import argparse
import os
import shutil
import zipfile

BASE_REPO = "coqui/XTTS-v2"
BASE_FILES = ["model.pth", "dvae.pth", "mel_stats.pth", "vocab.json", "config.json", "speakers_xtts.pth"]


def count_rows(path):
    if not os.path.isfile(path):
        return 0
    with open(path, encoding="utf-8") as f:
        return max(sum(1 for _ in f) - 1, 0)  # minus header


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset folder (metadata_*.csv + wavs/)")
    ap.add_argument("--out", default="run", help="output folder")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=2, help="per-step batch size (2 fits a 16 GB T4)")
    ap.add_argument("--grad-accum", type=int, default=2,
                    help="gradient accumulation steps (small = more updates, right for small datasets)")
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--test-text", default="This is a test of my fine tuned voice. How does it sound?")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the model, load the dataset and export, but skip training (checks the setup)")
    a = ap.parse_args()

    from huggingface_hub import hf_hub_download
    from trainer import Trainer, TrainerArgs
    from TTS.config.shared_configs import BaseDatasetConfig
    from TTS.tts.datasets import load_tts_samples
    from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig
    from TTS.tts.models.xtts import XttsAudioConfig

    n_train = count_rows(os.path.join(a.data, "metadata_train.csv"))
    n_eval = count_rows(os.path.join(a.data, "metadata_eval.csv"))
    print(f"dataset: {n_train} train clips, {n_eval} eval clips")
    if n_train < 20:
        print("WARNING: very little data. Aim for 10-30 minutes (150+ clips) or the model will just memorise it.")

    base = os.path.join(a.out, "base")
    os.makedirs(base, exist_ok=True)
    for f in BASE_FILES:
        if not os.path.isfile(os.path.join(base, f)):
            print("downloading", f)
            shutil.copy(hf_hub_download(BASE_REPO, f), os.path.join(base, f))

    ds = BaseDatasetConfig(
        formatter="coqui", dataset_name="voice", path=a.data, language=a.lang,
        meta_file_train="metadata_train.csv", meta_file_val="metadata_eval.csv" if n_eval else "")

    args = GPTArgs(
        max_conditioning_length=132300, min_conditioning_length=66150, debug_loading_failures=False,
        max_wav_length=255995, max_text_length=200,
        mel_norm_file=os.path.join(base, "mel_stats.pth"), dvae_checkpoint=os.path.join(base, "dvae.pth"),
        xtts_checkpoint=os.path.join(base, "model.pth"), tokenizer_file=os.path.join(base, "vocab.json"),
        gpt_num_audio_tokens=1026, gpt_start_audio_token=1024, gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True, gpt_use_perceiver_resampler=True)

    first_clip = os.path.join(a.data, open(os.path.join(a.data, "metadata_train.csv"), encoding="utf-8")
                              .read().splitlines()[1].split("|")[0])
    config = GPTTrainerConfig(
        datasets=[ds], epochs=a.epochs, output_path=a.out, model_args=args, run_name="xtts_ft", project_name="voice_studio",
        audio=XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000),
        batch_size=a.batch, batch_group_size=48, eval_batch_size=a.batch, num_loader_workers=2,
        eval_split_max_size=256, print_step=25, plot_step=100, log_model_step=1000,
        save_step=100000, save_n_checkpoints=1, save_checkpoints=False, run_eval=bool(n_eval),
        optimizer="AdamW", optimizer_wd_only_on_weights=True,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2}, lr=a.lr,
        lr_scheduler="MultiStepLR", lr_scheduler_params={"milestones": [10**9], "gamma": 0.5, "last_epoch": -1},
        test_sentences=[{"text": a.test_text, "speaker_wav": first_clip, "language": a.lang}])

    model = GPTTrainer(config)
    kw = dict(eval_split=bool(n_eval), eval_split_max_size=config.eval_split_max_size,
              eval_split_size=config.eval_split_size)
    try:                                   # coqui-tts >= 0.28: takes the config
        train, evals = load_tts_samples(config, **kw)
    except TypeError:                      # older versions: takes the dataset list
        train, evals = load_tts_samples([ds], **kw)
    print(f"loaded {len(train)} train / {len(evals or [])} eval samples")
    if a.dry_run:
        print("dry run: skipping training")
    else:
        trainer = Trainer(
            TrainerArgs(restore_path=None, skip_train_epoch=False, start_with_eval=False,
                        grad_accum_steps=a.grad_accum),
            config, output_path=a.out, model=model, train_samples=train, eval_samples=evals)
        trainer.fit()

    # Export the trained weights straight from memory (works whether or not checkpoints were written).
    export = os.path.join(a.out, "export")
    os.makedirs(export, exist_ok=True)
    import torch
    sd = {k: v.cpu() for k, v in model.xtts.state_dict().items()
          if not (k.startswith("dvae") or "torch_mel_spectrogram" in k)}
    torch.save({"model": sd}, os.path.join(export, "model.pth"))
    for f in ("config.json", "vocab.json", "speakers_xtts.pth"):
        shutil.copy(os.path.join(base, f), os.path.join(export, f))
    zpath = os.path.join(a.out, "finetuned_xtts.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
        for f in sorted(os.listdir(export)):
            z.write(os.path.join(export, f), f)
    print("DONE. Download:", zpath, f"({os.path.getsize(zpath) / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
