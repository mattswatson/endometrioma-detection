"""
Model loading for the backbones we use. Timm models wrapped so that they work
with the HuggingFace Trainer and return the same transforms.
"""
import os
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import timm
import torch.nn as nn
from safetensors.torch import load_file
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers.modeling_outputs import ImageClassifierOutput

from endometriosis.data import ID2LABEL, LABEL2ID

MODELS = {
    "vit": "google/vit-base-patch16-224",
    "omnirad": "Snarcy/OmniRad-small",
}


class TimmForImageClassification(nn.Module):
    """A wrapper for timm models which give the output in the format expected by the HF Trainer"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values=None, labels=None):
        logits = self.model(pixel_values)
        return ImageClassifierOutput(logits=logits)

    @property
    def device(self):
        return next(self.parameters()).device


@dataclass
class LoadedModel:
    model: nn.Module
    train_transforms: Callable
    val_transforms: Callable

    # Normalisation used by the transforms, needed to undo it when plotting images
    mean: Sequence[float]
    std: Sequence[float]

    # Only set for HuggingFace models; passed to the Trainer so it is saved alongside checkpoints
    image_processor: Optional[Any] = None


def _hf_transforms(image_processor):
    """Create transform functions, with augmentation for training"""
    normalize = transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    if "height" in image_processor.size:
        size = (image_processor.size["height"], image_processor.size["width"])
        crop_size = size
    elif "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
        crop_size = (size, size)

    train_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize,
        ]
    )

    return train_transforms, val_transforms


def load_model(model_key: str, checkpoint: Optional[str]=None) -> LoadedModel:
    """
    Load a binary classification model.

    Args:
        model_key: One of MODELS.
        checkpoint: Optional Trainer checkpoint directory to load fine-tuned weights from. If not given, the
            pretrained backbone is loaded with a fresh 2-class head.
    """
    model_name = MODELS[model_key]

    if model_key == "vit":
        path = checkpoint or model_name
        model = AutoModelForImageClassification.from_pretrained(
            path, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID, ignore_mismatched_sizes=True
        )
        image_processor = AutoImageProcessor.from_pretrained(path)
        train_transforms, val_transforms = _hf_transforms(image_processor)

        return LoadedModel(
            model, train_transforms, val_transforms, image_processor.image_mean, 
            image_processor.image_std, image_processor
        )

    timm_model = timm.create_model(f"hf-hub:{model_name}", pretrained=True, num_classes=2)
    data_config = timm.data.resolve_data_config(timm_model.pretrained_cfg) # type: ignore
    train_transforms = timm.data.create_transform(**data_config, is_training=True) # type: ignore
    val_transforms = timm.data.create_transform(**data_config, is_training=False) # type: ignore

    model = TimmForImageClassification(timm_model)
    if checkpoint is not None:
        model.load_state_dict(load_file(os.path.join(checkpoint, "model.safetensors")))

    return LoadedModel(model, train_transforms, val_transforms, data_config["mean"], data_config["std"])
