"""
Run a trained checkpoint on the MMOTU test set and save its outputs for the paper plots.
"""
import argparse

import torch
from datasets import load_dataset

from endometriosis.data import DEFAULT_DATASET_PATH, with_image_transform
from endometriosis.inference import predict_test_set, save_outputs
from endometriosis.models import MODELS, load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS.keys(), required=True)
    parser.add_argument("--checkpoint", required=True, 
                        help="Trainer checkpoint directory, e.g. outputs/checkpoints/<run>/checkpoint-40")
    parser.add_argument("--out", required=True, 
                        help="Pickle file to write, e.g. notebooks/outputs/vit-outputs.pkl")
    parser.add_argument("--data", default=DEFAULT_DATASET_PATH,
                        help="Path to the HuggingFace MMOTU dataset")
    args = parser.parse_args()

    dataset = load_dataset(args.data)
    loaded = load_model(args.model, checkpoint=args.checkpoint)
    test_ds = with_image_transform(dataset["test"], loaded.val_transforms)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded.model.to(device)

    outputs = predict_test_set(loaded.model, test_ds)
    print(f"Model outputs shape: {outputs['logits'].shape}")
    print(f"Accuracy: {(outputs['predictions'] == outputs['labels']).float().mean().item():.4f}")

    save_outputs(outputs, args.out)


if __name__ == "__main__":
    main()
