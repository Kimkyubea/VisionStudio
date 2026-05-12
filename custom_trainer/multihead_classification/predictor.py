# -*- coding: utf-8 -*-

import os
import random

import cv2
import numpy as np
import torch

from PIL import Image

from .utils import build_transform, load_checkpoint, normalize_device


VALID_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]


def is_image_file(path):
    return os.path.splitext(path)[1].lower() in VALID_EXTS


def collect_images(root_dir, shuffle=False):
    image_list = []
    for name in os.listdir(root_dir):
        path = os.path.join(root_dir, name)
        if os.path.isfile(path) and is_image_file(path):
            image_list.append(path)

    if shuffle:
        random.shuffle(image_list)
    else:
        image_list.sort()

    return image_list


def imread_unicode(path):
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def resolve_thresholds(metadata, thresholds=None, default_threshold=None):
    resolved = {}
    metadata_thresholds = metadata.get("predict_thresholds", {})

    for head in metadata["heads"]:
        head_name = head["name"]
        threshold = None
        if thresholds and head_name in thresholds:
            threshold = thresholds[head_name]
        elif head_name in metadata_thresholds:
            threshold = metadata_thresholds[head_name]
        elif default_threshold is not None:
            threshold = default_threshold

        resolved[head_name] = None if threshold is None else float(threshold)

    return resolved


def load_model(model_path, device=None):
    device = normalize_device(device)
    model, metadata, checkpoint = load_checkpoint(model_path, device=device)
    return model, metadata, checkpoint, device


def to_pil_image(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, (str, os.PathLike)):
        return Image.open(image).convert("RGB")

    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return Image.fromarray(image).convert("RGB")
        if image.ndim == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb_image).convert("RGB")
        raise ValueError("Unsupported numpy image shape: {}".format(image.shape))

    raise TypeError("Unsupported image input type: {}".format(type(image)))


def predict(model, metadata, image, device, thresholds=None, default_threshold=None):
    transform_cfg = metadata.get("transform", {})
    transform = build_transform(
        int(metadata.get("input_size", 224)),
        mean=transform_cfg.get("mean"),
        std=transform_cfg.get("std"),
    )
    pil_img = to_pil_image(image)
    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    threshold_by_head = resolve_thresholds(metadata, thresholds=thresholds, default_threshold=default_threshold)

    with torch.no_grad():
        logits_by_head = model(input_tensor)

    predictions = {}
    for head in sorted(metadata["heads"], key=lambda item: item.get("index", 0)):
        head_name = head["name"]
        probabilities = torch.softmax(logits_by_head[head_name], dim=1)
        prob_values = probabilities.squeeze(0).detach().cpu().tolist()
        pred_idx = int(torch.argmax(probabilities, dim=1))
        pred_conf = float(torch.max(probabilities))

        class_names = head.get("class_names") or []
        pred_name = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
        applied_threshold = threshold_by_head.get(head_name)
        is_unknown = applied_threshold is not None and pred_conf < applied_threshold
        display_name = "unknown" if is_unknown else pred_name

        predictions[head_name] = {
            "index": pred_idx,
            "name": display_name,
            "raw_name": pred_name,
            "confidence": pred_conf,
            "threshold": applied_threshold,
            "is_unknown": is_unknown,
            "scores": [
                {
                    "index": class_idx,
                    "name": class_names[class_idx] if class_idx < len(class_names) else str(class_idx),
                    "confidence": float(class_conf),
                }
                for class_idx, class_conf in enumerate(prob_values)
            ],
        }

    return predictions


def draw_predictions(image, predictions):
    if isinstance(image, (str, os.PathLike)):
        canvas = imread_unicode(image)
    elif isinstance(image, np.ndarray):
        canvas = image.copy()
    else:
        canvas = None

    if canvas is None:
        return None

    y = 30
    for head_name, result in predictions.items():
        text = "{}: {} ({:.2f})".format(head_name, result["name"], result["confidence"])
        color = (0, 165, 255) if result.get("is_unknown") else (0, 255, 0)
        cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        y += 30

    return canvas


class MultiHeadClassificationPredictor:
    def __init__(self, cfg):
        self.model, self.metadata, _, self.device = load_model(
            cfg["model_path"],
            device=cfg.get("device"),
        )
        self.default_threshold = cfg.get("threshold")
        self.thresholds = cfg.get("thresholds")

    def predict(self, image):
        return predict(
            self.model,
            self.metadata,
            image,
            self.device,
            thresholds=self.thresholds,
            default_threshold=self.default_threshold,
        )

    def draw(self, image, predictions=None):
        if predictions is None:
            predictions = self.predict(image)
        return draw_predictions(image, predictions)
