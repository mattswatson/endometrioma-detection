"""
Convert the raw 2D MMOTU data into a binary (endometriosis vs not) HuggingFace ImageFolder dataset.
"""
import argparse

from endometriosis.data import DEFAULT_DATASET_PATH, MMOTUDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/media/hdd/mmotu/2d", 
                        help="Root directory of the raw MMOTU 2D data")
    parser.add_argument("--out", default=DEFAULT_DATASET_PATH, 
                        help="Where to write the HuggingFace dataset")
    parser.add_argument("--no-overwrite", action="store_true", 
                        help="Don't replace an existing dataset at --out")
    args = parser.parse_args()

    dataset = MMOTUDataset(root=args.root, preload=False, split="all").to_binary_label()
    hf_dataset = dataset.to_hf_dataset(args.out, overwrite=not args.no_overwrite)
    print(hf_dataset)


if __name__ == "__main__":
    main()
