# Endometrioma detection

Supporting repository for ovarian endometrioma detection using vision transformers on publicly available data. This work is currently under revew at npj Women's Health.

## Usage

Shared code lives in `src/endometriosis`, and experiments are run with the scripts in `scripts/` (from the repo root):

```bash
# Convert the raw MMOTU 2D data into a binary HuggingFace dataset
uv run scripts/make_hf_dataset.py --root /media/hdd/mmotu/2d --out /media/hdd/mmotu/hf_dataset

# Train all models (checkpoints, test outputs and plots go to outputs/checkpoints/<run name>)
scripts/run_experiments.sh

# Or train a single model
uv run scripts/train.py --model omnirad --weighted-loss

# Save test set outputs from a checkpoint for the paper plots
uv run scripts/predict.py --model vit --checkpoint outputs/checkpoints/google/vit-base-patch16-224-finetuned-binary-mmotu/checkpoint-40 --out notebooks/outputs/vit-outputs.pkl
```

The SHAP analysis and paper plots are in `notebooks/`.