"""
Pieces shared by training runs: metrics, collation and a Trainer with an (optionally) weighted loss.
"""
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, average_precision_score, 
                             roc_auc_score)
from sklearn.utils.class_weight import compute_class_weight
from transformers import Trainer


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    ap = average_precision_score(labels, pred.predictions[:, 1])
    auroc = roc_auc_score(labels, pred.predictions[:, 1])
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "average_precision": ap, "auroc": auroc}


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "labels": torch.tensor([item["label"] for item in batch]),
    }


def balanced_class_weights(labels) -> torch.Tensor:
    """Compute loss weights based on class distribution"""
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels)
    return torch.tensor(class_weights, dtype=torch.float)


class WeightedCETrainer(Trainer):
    """A Trainer that uses a cross entropy loss, weighted if loss_weights is given"""
    def __init__(self, *args, loss_weights: torch.Tensor | None=None, **kwargs):
        super().__init__(*args, **kwargs)

        if loss_weights is not None:
            self.loss_fn = torch.nn.CrossEntropyLoss(weight=loss_weights.to(self.model.device)) # type: ignore
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = self.loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss
