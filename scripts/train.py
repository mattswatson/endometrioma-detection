"""
Train a binary classification model on the MMOTU dataset, optionally with a weighted cross entropy loss.

Checkpoints, test set outputs (for the paper plots) and evaluation plots are written to --output-dir.
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import wandb
from datasets import load_dataset
from sklearn.metrics import PrecisionRecallDisplay, confusion_matrix
from transformers import TrainingArguments

from endometriosis.data import DEFAULT_DATASET_PATH, ID2LABEL, with_image_transform
from endometriosis.inference import outputs_from_logits, save_outputs
from endometriosis.models import MODELS, load_model
from endometriosis.training import WeightedCETrainer, balanced_class_weights, collate_fn, compute_metrics


def plot_results(labels, predictions, title, output_dir):
    """Save a confusion matrix and precision-recall curve for the test set"""
    cm = confusion_matrix(labels, predictions.argmax(axis=1))
    g = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=ID2LABEL.values(), 
                    yticklabels=ID2LABEL.values())
    g.set_xlabel("Predicted label")
    g.set_ylabel("True label")
    g.set_title(f"Endometriosis Confusion Matrix\n({title})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    disp = PrecisionRecallDisplay.from_predictions(labels, predictions[:, 1])
    disp.ax_.set_title(f"Precision-Recall curve\n({title})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "precision_recall.png"))
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS.keys(), required=True)
    parser.add_argument("--weighted-loss", action="store_true", 
                        help="Weight the cross entropy loss by class frequency")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--data", default=DEFAULT_DATASET_PATH, 
                        help="Path to the HuggingFace MMOTU dataset")
    parser.add_argument("--output-dir", help="Defaults to outputs/checkpoints/<run name>")
    args = parser.parse_args()

    model_name = MODELS[args.model]
    run_name = f"{model_name}-finetuned-binary{'-weightedCE' if args.weighted_loss else ''}-mmotu"
    output_dir = args.output_dir or os.path.join("outputs", "checkpoints", run_name)

    dataset = load_dataset(args.data)
    loaded = load_model(args.model)

    # And set up our train and test datasets (for now, no val)
    train_ds = with_image_transform(dataset["train"], loaded.train_transforms)
    test_ds = with_image_transform(dataset["test"], loaded.val_transforms)

    wandb.init(project="mmotu-binary-classification", name=run_name)

    training_args = TrainingArguments(
        output_dir,
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=32,
        num_train_epochs=args.epochs,
        warmup_steps=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
        report_to="wandb"
    )

    class_weights = None
    if args.weighted_loss:
        class_weights = balanced_class_weights(dataset["train"]["label"])
        print(f"Class weights: {class_weights}")

    trainer = WeightedCETrainer(
        model=loaded.model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
        processing_class=loaded.image_processor,
        loss_weights=class_weights
    )

    trainer.train()

    eval_results = trainer.evaluate()
    print(eval_results)

    # Get the predictions and true labels for the test set from the best model
    predictions, labels, _ = trainer.predict(test_ds) # type: ignore
    save_outputs(outputs_from_logits(torch.from_numpy(predictions), torch.from_numpy(labels)), os.path.join(output_dir, "test-outputs.pkl"))

    title = f"{model_name}{', WeightedCE' if args.weighted_loss else ''}, Binary MMOTU US"
    plot_results(np.array(labels), predictions, title, output_dir)

    wandb.finish()


if __name__ == "__main__":
    main()
