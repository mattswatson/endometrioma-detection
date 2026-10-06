#!/usr/bin/env bash
# The binary MMOTU training runs, with the settings used for the paper.
set -euo pipefail
cd "$(dirname "$0")/.."

uv run scripts/train.py --model vit --epochs 5
uv run scripts/train.py --model vit --weighted-loss
uv run scripts/train.py --model omnirad
uv run scripts/train.py --model omnirad --weighted-loss
