"""
MMOTU dataset handling: converting the raw 2D MMOTU data to a binary HuggingFace dataset, and loading it.
"""
import numbers
import os
import shutil
from typing import Callable, Optional, List, Tuple, Dict, Any, Iterator
from PIL import Image
import torch
import pandas as pd
from torch.utils.data import Dataset
from datasets import load_dataset
from tqdm import tqdm

import torchvision.transforms as T

DEFAULT_DATASET_PATH = "/media/hdd/mmotu/hf_dataset"

ID2LABEL = {0: "non-endometriosis", 1: "endometriosis"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


class MMOTUDataset(Dataset):
    """
    PyTorch dataset for 2D MMOTU.
    - Automatically supports:
      1) labels CSV: root/{train|val}_cls.txt containing 'image,label' (image path relative to root)
    - Returns dict {'image': Tensor, 'label': int} by default.
    - Call .to_hf_dataset() to get a Hugging Face Dataset
    """

    IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")

    def __init__(
        self,
        root: str,
        transform: Optional[Callable]=None,
        target_transform: Optional[Callable]=None,
        preload: bool=False,
        split: str="all"
    ):
        """
        Args:
            root: Root directory of the dataset.
            transform: Optional transform to apply to images.
            target_transform: Optional transform to apply to labels.
            preload: If True, load all images into memory at initialization (faster access, more memory usage).
            split: Which split to use ('train', 'test', or 'all').
        """
        self.root = root
        self.transform = transform or T.ToTensor()
        self.target_transform = target_transform
        self.preload = preload
        self.split = split

        self.samples: List[Tuple[str, int, str]] = []
        self.class_to_idx: Dict[str, int] = {}
        self._cache: List[Tuple[Any, int, str]] = []  # used if preload

        self._build_index()

        if self.preload:
            self._preload_images()

    def _is_image(self, fname: str) -> bool:
        return fname.lower().endswith(self.IMG_EXTS)

    def _build_index(self):
        # Assume we have train_cls.txt and val_cls.txt listing image IDs and labels
        if self.split == "train":
            label_filenames = [os.path.join(self.root, "train_cls.txt")]
        elif self.split == "test":
            label_filenames = [os.path.join(self.root, "val_cls.txt")]
        elif self.split == "all":
            # We want both train and test
            label_filenames = [
                os.path.join(self.root, "train_cls.txt"),
                os.path.join(self.root, "val_cls.txt")
            ]
        else:
            raise ValueError(f"Invalid split: {self.split}. Must be 'train', 'test', or 'all'.")

        for label_filename in label_filenames:
            current_split = "train" if "train" in label_filename else "test"

            if os.path.isfile(label_filename):
                df = pd.read_csv(label_filename, header=None, names=["image", "label"], sep="  ", 
                                 engine="python")
            else:
                raise FileNotFoundError(f"Label file not found: {label_filename}")

            # Run the add_sample function for each row in the dataframe
            df.apply(lambda row: self._add_sample(row["image"], row["label"], current_split), axis=1) # type: ignore

    def _add_sample(self, img_name: str, label_raw, split: str):
        # Make label to int. Integral (not int) so numpy integers keep their original class index
        # Note: this is quite fragile and very specific to this dataset
        # May want to come back and refactor this
        if isinstance(label_raw, numbers.Integral):
            label_idx = int(label_raw)
            label_name = str(label_raw)
        else:
            label_name = str(label_raw)
            if label_name not in self.class_to_idx:
                self.class_to_idx[label_name] = len(self.class_to_idx)
            label_idx = self.class_to_idx[label_name]

        # Add "images" to image path
        img_path = os.path.join(self.root, "images", img_name)

        self.samples.append((img_path, label_idx, split))

    def _preload_images(self):
        self._cache = []
        for p, lbl, split in self.samples:
            img = Image.open(p).convert("RGB")
            img_t = self.transform(img) if self.transform else img
            lbl_t = self.target_transform(lbl) if self.target_transform else lbl
            self._cache.append((img_t, lbl_t, split))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.preload:
            img_t, lbl, split = self._cache[idx]
            return {"image": img_t, "label": lbl}

        path, label, split = self.samples[idx]
        img = Image.open(path).convert("RGB")

        img_t = self.transform(img) if self.transform else img
        lbl_t = self.target_transform(label) if self.target_transform else label

        return {"image": img_t, "label": lbl_t}

    def get_class_names(self) -> List[str]:
        # return class names sorted by index
        names = [""] * len(self.class_to_idx)
        for n, i in self.class_to_idx.items():
            names[i] = n
        return names

    def _generator(self) -> Iterator[Dict[str, Any]]:
        for i in range(len(self)):
            item = self[i]
            # convert tensors to lists for HF compatibility
            img = item["image"]
            if torch.is_tensor(img):
                img = img.numpy().tolist()
            yield {"image": img, "label": int(item["label"])}

    def to_hf_dataset(self, huggingface_path: str, overwrite: bool=True):
        """
        Convert to Hugging Face ImageFolder dataset and save to disk.
        """
        # Create the folder, warn if it exists
        if os.path.exists(huggingface_path):
            if overwrite:
                print(f"Warning: Overwriting existing Hugging Face dataset at {huggingface_path}")
                shutil.rmtree(huggingface_path)
            else:
                print(f"Hugging Face dataset already exists at {huggingface_path}\nUse overwrite=True to replace it.")
                return

        # Put all of the images in this folder and get ready to create metadata.csv
        rows = []
        for path, label, split in tqdm(self.samples, desc="Converting to Hugging Face format"):
            split_folder = os.path.join(huggingface_path, split)
            os.makedirs(split_folder, exist_ok=True)

            shutil.copy(path, split_folder)
            img_name = os.path.basename(path)
            rows.append({"file_name": img_name, "label": label, "split": split})

        # Save metadata.csv for each split
        whole_df = pd.DataFrame(rows)
        for split in whole_df["split"].unique():
            split_df = whole_df[whole_df["split"] == split].drop(columns=["split"])
            split_df.to_csv(os.path.join(huggingface_path, split, "metadata.csv"), index=False)

        # Return a huggingface Dataset object pointing to this folder
        return load_dataset(huggingface_path)

    def to_binary_label(self):
        """
        Convert labels to binary (0 vs 1) based on the chocolate cyst class
        Modifies the dataset in-place.
        """
        # We know class 0 is chocolate cyst from the paper
        label_idx = 0
        for i in range(len(self.samples)):
            path, lbl, split = self.samples[i]
            binary_lbl = 1 if lbl == label_idx else 0
            self.samples[i] = (path, binary_lbl, split)

        # If we have preloaded, also update the cache
        if self.preload:
            for i in range(len(self._cache)):
                img, lbl, split = self._cache[i]
                binary_lbl = 1 if lbl == label_idx else 0
                self._cache[i] = (img, binary_lbl, split)

        return self  # allow chaining


def with_image_transform(split, transform):
    """Lazily apply transform to each image in a HF dataset split, adding a pixel_values column."""
    def preprocess(example_batch):
        example_batch["pixel_values"] = [transform(image.convert("RGB")) for image in example_batch["image"]]
        return example_batch

    return split.with_transform(preprocess)
