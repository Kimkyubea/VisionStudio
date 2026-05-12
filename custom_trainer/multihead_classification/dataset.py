# -*- coding:utf-8 -*-

import os

import torch

from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .utils import build_transform


VALID_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]


class MultiHeadDataset(Dataset):
    def __init__(self, image_paths, labels_by_head, transform=None):
        self.image_paths = image_paths
        self.labels_by_head = labels_by_head
        self.head_names = list(labels_by_head.keys())
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        target = {
            head_name: torch.tensor(self.labels_by_head[head_name][idx], dtype=torch.long)
            for head_name in self.head_names
        }
        return image, target


def is_image_file(path):
    return os.path.splitext(path)[1].lower() in VALID_IMAGE_EXTENSIONS


def map_label_value(raw_label, head):
    mapped_label = int(raw_label)
    if mapped_label == -1:
        return mapped_label
    if mapped_label < 0 or mapped_label >= head["num_classes"]:
        raise ValueError("Label out of range for head '{}'".format(head["name"]))
    return mapped_label


def parse_sample_label_file(label_path, heads, encoding="utf-8"):
    with open(label_path, "r", encoding=encoding) as file:
        tokens = file.readline().strip().split()

    if len(tokens) != len(heads):
        raise ValueError(
            "Label count mismatch in '{}': expected {}, got {}".format(label_path, len(heads), len(tokens))
        )

    parsed_targets = {}
    for idx, head in enumerate(heads):
        parsed_targets[head["name"]] = map_label_value(tokens[idx], head)

    return parsed_targets


def collect_samples(image_dir, heads, label_dir=None, label_ext=".txt", label_encoding="utf-8"):
    image_paths = []
    labels_by_head = {head["name"]: [] for head in heads}

    for image_name in tqdm(sorted(os.listdir(image_dir))):
        image_path = os.path.join(image_dir, image_name)
        if not os.path.isfile(image_path) or not is_image_file(image_path):
            continue

        stem = os.path.splitext(image_name)[0]
        current_label_dir = label_dir or image_dir
        label_path = os.path.join(current_label_dir, stem + label_ext)
        if not os.path.exists(label_path):
            continue

        try:
            parsed_targets = parse_sample_label_file(label_path, heads, encoding=label_encoding)
        except ValueError:
            continue

        image_paths.append(image_path)
        for head_name, mapped_label in parsed_targets.items():
            labels_by_head[head_name].append(mapped_label)

    return image_paths, labels_by_head


def create_dataloader(cfg, metadata, dataset_paths, split="train", batch_size=None, shuffle=None):
    image_dir = dataset_paths.get("{}_image_dir".format(split))
    label_dir = dataset_paths.get("{}_label_dir".format(split))

    if not image_dir:
        return None, [], {}

    image_paths, labels_by_head = collect_samples(
        image_dir=image_dir,
        heads=metadata["heads"],
        label_dir=label_dir,
        label_ext=cfg.get("label_ext", ".txt"),
        label_encoding=cfg.get("label_encoding", "utf-8"),
    )

    if not image_paths:
        if split == "train":
            raise ValueError("No training samples were found. Check dataset yaml train path and inferred labels path.")
        return None, [], {}

    dataset = MultiHeadDataset(
        image_paths=image_paths,
        labels_by_head=labels_by_head,
        transform=build_transform(
            metadata["input_size"],
            mean=metadata.get("transform", {}).get("mean"),
            std=metadata.get("transform", {}).get("std"),
        ),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size if batch_size is not None else cfg.get("batch", 16)),
        shuffle=bool(cfg.get("shuffle", True) if shuffle is None else shuffle),
        num_workers=int(cfg.get("workers", 0)),
    )

    return loader, image_paths, labels_by_head
