# -*- coding: utf-8 -*-

import timm
import torch
import torch.nn as nn


DEFAULT_HEADS = [
    {
        "name": "kind",
        "label_key": "typeID",
        "num_classes": 8,
        "class_names": ["sedan", "suv", "van", "hatchback", "pickup", "bus", "truck", "estate"],
    },
    {
        "name": "color",
        "label_key": "colorID",
        "num_classes": 10,
        "class_names": ["yellow", "orange", "green", "gray", "red", "blue", "white", "golden", "brown", "black"],
    },
]


def normalize_backbone_config(cfg):
    raw_backbone = cfg.get("backbone")
    if isinstance(raw_backbone, str):
        return {
            "name": raw_backbone,
            "pretrained": bool(cfg.get("pretrained", True)),
            "mean": cfg.get("mean"),
            "std": cfg.get("std"),
        }

    if isinstance(raw_backbone, dict):
        return {
            "name": str(raw_backbone.get("name", cfg.get("backbone_name", "convnext_tiny"))),
            "pretrained": bool(raw_backbone.get("pretrained", cfg.get("pretrained", True))),
            "mean": raw_backbone.get("mean", cfg.get("mean")),
            "std": raw_backbone.get("std", cfg.get("std")),
        }

    return {
        "name": str(cfg.get("backbone_name", "convnext_tiny")),
        "pretrained": bool(cfg.get("pretrained", True)),
        "mean": cfg.get("mean"),
        "std": cfg.get("std"),
    }


def normalize_heads(cfg):
    raw_heads = cfg.get("heads")
    if not raw_heads:
        raw_heads = [
            {
                "name": "kind",
                "label_key": "typeID",
                "num_classes": int(cfg.get("num_type_classes", 8)),
                "class_names": cfg.get("type_names") or DEFAULT_HEADS[0]["class_names"],
            },
            {
                "name": "color",
                "label_key": "colorID",
                "num_classes": int(cfg.get("num_color_classes", 10)),
                "class_names": cfg.get("color_names") or DEFAULT_HEADS[1]["class_names"],
            },
        ]

    heads = []
    for idx, head in enumerate(raw_heads):
        name = str(head["name"])
        label_key = str(head.get("label_key", name))
        class_names = [str(name_item) for name_item in head.get("class_names", [])]
        num_classes = int(head.get("num_classes", len(class_names)))

        if num_classes <= 0:
            raise ValueError("num_classes must be positive for head '{}'".format(name))
        if class_names and len(class_names) != num_classes:
            raise ValueError("class_names length must match num_classes for head '{}'".format(name))

        heads.append(
            {
                "index": idx,
                "name": name,
                "label_key": label_key,
                "num_classes": num_classes,
                "class_names": class_names,
                "loss_weight": float(head.get("loss_weight", 1.0)),
                "label_smoothing": float(head.get("label_smoothing", cfg.get("label_smoothing", 0.0))),
                "auto_class_weight": bool(head.get("auto_class_weight", cfg.get("auto_class_weight", False))),
                "class_weight_strategy": str(head.get("class_weight_strategy", cfg.get("class_weight_strategy", "inverse"))),
                "class_weights": head.get("class_weights"),
            }
        )

    return heads


def build_metadata(cfg):
    backbone_cfg = normalize_backbone_config(cfg)
    transform_cfg = resolve_transform_config(backbone_cfg, cfg.get("imgsz", 224))
    return {
        "framework": cfg.get("framework", "custom_multihead"),
        "task": cfg.get("task", "classification"),
        "backbone": backbone_cfg,
        "backbone_name": backbone_cfg["name"],
        "pretrained": backbone_cfg["pretrained"],
        "input_size": int(cfg.get("imgsz", 224)),
        "transform": transform_cfg,
        "heads": normalize_heads(cfg),
        "predict_thresholds": cfg.get("thresholds", {}),
    }


def resolve_transform_config(backbone_cfg, input_size):
    mean = backbone_cfg.get("mean")
    std = backbone_cfg.get("std")

    if mean is None or std is None:
        try:
            model = timm.create_model(backbone_cfg["name"], pretrained=False, num_classes=0)
            pretrained_cfg = getattr(model, "pretrained_cfg", {}) or {}
            mean = pretrained_cfg.get("mean", mean)
            std = pretrained_cfg.get("std", std)
        except Exception:
            pass

    mean = mean or (0.485, 0.456, 0.406)
    std = std or (0.229, 0.224, 0.225)

    return {
        "input_size": int(input_size),
        "mean": [float(value) for value in mean],
        "std": [float(value) for value in std],
    }


class MultiHeadModel(nn.Module):
    def __init__(self, heads, backbone_name="convnext_tiny", pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features
        self.heads = nn.ModuleDict(
            {
                head["name"]: nn.Linear(feat_dim, int(head["num_classes"]))
                for head in heads
            }
        )

    def forward(self, x):
        feat = self.backbone(x)
        return {head_name: head_layer(feat) for head_name, head_layer in self.heads.items()}


def load_checkpoint(model_path, device, metadata_builder=normalize_heads):
    checkpoint = torch.load(model_path, map_location=device)
    metadata = checkpoint.get("metadata", {})

    if not metadata.get("heads"):
        metadata = {
            "backbone": {
                "name": metadata.get("backbone_name", "convnext_tiny"),
                "pretrained": False,
            },
            "backbone_name": metadata.get("backbone_name", "convnext_tiny"),
            "pretrained": False,
            "input_size": metadata.get("input_size", 224),
            "transform": metadata.get(
                "transform",
                {
                    "input_size": metadata.get("input_size", 224),
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                },
            ),
            "heads": metadata_builder(metadata),
            "predict_thresholds": metadata.get("thresholds", {}),
        }
    elif not metadata.get("backbone"):
        metadata["backbone"] = {
            "name": metadata.get("backbone_name", "convnext_tiny"),
            "pretrained": metadata.get("pretrained", False),
        }
        metadata["backbone_name"] = metadata["backbone"]["name"]
    if not metadata.get("transform"):
        metadata["transform"] = {
            "input_size": metadata.get("input_size", 224),
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        }

    model = MultiHeadModel(
        heads=metadata["heads"],
        backbone_name=metadata["backbone"]["name"],
        pretrained=False,
    )

    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, metadata, checkpoint
