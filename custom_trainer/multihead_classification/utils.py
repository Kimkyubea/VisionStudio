# -*- coding:utf-8 -*-

import os

import torch
import yaml

from torchvision import transforms

from .model import load_checkpoint as load_model_checkpoint
from .model import normalize_heads


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data or {}


def save_yaml(path, data):
    with open(path, "w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, allow_unicode=True, sort_keys=False)


def resolve_path(path_value, base_dir=None):
    if path_value is None:
        return None
    if os.path.isabs(path_value):
        return os.path.normpath(path_value)
    if base_dir:
        return os.path.normpath(os.path.join(base_dir, path_value))
    return os.path.normpath(path_value)


def normalize_device(device):
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == -1:
        return "cpu"
    if device == 0:
        return "cuda:0"
    if isinstance(device, int):
        return "cuda:{}".format(device)
    return device


def build_transform(input_size, mean=None, std=None):
    mean = mean or [0.485, 0.456, 0.406]
    std = std or [0.229, 0.224, 0.225]
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def infer_label_dir(image_dir):
    normalized = os.path.normpath(image_dir)
    parts = normalized.split(os.sep)
    for idx, part in enumerate(parts):
        if part.lower() == "images":
            replaced = list(parts)
            replaced[idx] = "labels"
            return os.sep.join(replaced)
    return normalized


def load_dataset_paths(cfg):
    dataset_path = cfg.get("dataset")
    if not dataset_path:
        return {
            "train_image_dir": cfg.get("train_image_dir"),
            "train_label_dir": cfg.get("train_label_dir"),
            "val_image_dir": cfg.get("val_image_dir"),
            "val_label_dir": cfg.get("val_label_dir"),
        }

    dataset_path = resolve_path(dataset_path, cfg.get("config_dir"))
    dataset_cfg = load_yaml(dataset_path)
    dataset_base_dir = os.path.dirname(dataset_path)

    train_image_dir = resolve_path(dataset_cfg.get("train"), dataset_base_dir)
    val_image_dir = resolve_path(dataset_cfg.get("val"), dataset_base_dir)

    return {
        "train_image_dir": train_image_dir,
        "train_label_dir": infer_label_dir(train_image_dir) if train_image_dir else None,
        "val_image_dir": val_image_dir,
        "val_label_dir": infer_label_dir(val_image_dir) if val_image_dir else None,
    }


def load_checkpoint(model_path, device=None):
    device = normalize_device(device)
    return load_model_checkpoint(model_path, device, metadata_builder=normalize_heads)
