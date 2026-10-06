"""
Running a trained model on the test set and saving the outputs used by the paper plots.
"""
import pickle

import torch
from torch.utils.data import DataLoader

from endometriosis.training import collate_fn


def predict(model, image_tensor):
    """Given an image, get the model's prediction"""
    model.eval()
    with torch.no_grad():
        outputs = model(pixel_values=image_tensor.unsqueeze(0).to(model.device))  # Add batch dimension
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        predicted_label = torch.argmax(probabilities, dim=1).item()

    return predicted_label, probabilities.squeeze().tolist()


def outputs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    return {
        "logits": logits,
        "probabilities": probabilities,
        "predictions": torch.argmax(probabilities, dim=1),
        "labels": labels,
    }


def predict_test_set(model, test_ds, batch_size: int=32) -> dict:
    """Run the model on the whole test set and get model outputs"""
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            outputs = model(pixel_values=batch["pixel_values"].to(model.device))
            all_logits.append(outputs.logits.cpu())
            all_labels.append(batch["labels"])

    return outputs_from_logits(torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0))


def save_outputs(outputs: dict, path: str):
    """Save model outputs to a file for plotting later on"""
    with open(path, "wb") as f:
        pickle.dump(outputs, f)
