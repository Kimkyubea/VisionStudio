# -*- coding: utf-8 -*-

import argparse
import copy
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .dataset import create_dataloader
from .model import MultiHeadModel, build_metadata
from .utils import load_dataset_paths, load_yaml, normalize_device, save_yaml


class ModelEMA:
    def __init__(self, model, decay=0.9998, tau=2000):
        self.ema = copy.deepcopy(model).eval()
        for parameter in self.ema.parameters():
            parameter.requires_grad_(False)
        self.decay = float(decay)
        self.tau = float(tau)
        self.updates = 0

    def _decay_value(self):
        return self.decay * (1.0 - math.exp(-self.updates / max(self.tau, 1.0)))

    def update(self, model):
        self.updates += 1
        decay = self._decay_value()

        model_state = model.state_dict()
        ema_state = self.ema.state_dict()
        for key, value in ema_state.items():
            if not torch.is_floating_point(value):
                value.copy_(model_state[key])
                continue
            value.mul_(decay).add_(model_state[key].detach(), alpha=1.0 - decay)


def build_lr_lambda(total_steps, warmup_steps, min_lr_ratio):
    total_steps = max(int(total_steps), 1)
    warmup_steps = max(int(warmup_steps), 0)
    min_lr_ratio = float(min_lr_ratio)

    def lr_lambda(step_index):
        step_index = min(int(step_index), total_steps)
        if warmup_steps > 0 and step_index < warmup_steps:
            return max((step_index + 1) / warmup_steps, 1e-6)
        if total_steps <= warmup_steps:
            return 1.0

        progress = (step_index - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return lr_lambda


def _normalize_manual_class_weights(head, device):
    class_weights = head.get("class_weights")
    if class_weights is None:
        return None

    if len(class_weights) != int(head["num_classes"]):
        raise ValueError(
            "class_weights length must match num_classes for head '{}'".format(head["name"])
        )

    return torch.tensor([float(weight) for weight in class_weights], dtype=torch.float32, device=device)


def _compute_auto_class_weights(head, labels_by_head, device):
    head_name = head["name"]
    labels = labels_by_head.get(head_name, [])
    valid_labels = [int(label) for label in labels if int(label) != -1]

    if not valid_labels:
        return torch.ones(int(head["num_classes"]), dtype=torch.float32, device=device)

    counts = torch.bincount(torch.tensor(valid_labels, dtype=torch.long), minlength=int(head["num_classes"])).float()
    strategy = str(head.get("class_weight_strategy", "inverse")).lower()

    weights = torch.ones_like(counts)
    positive_mask = counts > 0
    if strategy == "inverse":
        weights[positive_mask] = 1.0 / counts[positive_mask]
    elif strategy == "sqrt_inverse":
        weights[positive_mask] = 1.0 / torch.sqrt(counts[positive_mask])
    elif strategy == "log_inverse":
        weights[positive_mask] = 1.0 / torch.log(counts[positive_mask] + 1.0)
    else:
        raise ValueError(
            "Unsupported class_weight_strategy '{}' for head '{}'".format(strategy, head_name)
        )

    positive_weights = weights[positive_mask]
    if positive_weights.numel() > 0:
        weights[positive_mask] = positive_weights / positive_weights.mean().clamp(min=1e-6)
    return weights.to(device)


def build_class_weights(heads, labels_by_head, device):
    class_weights_by_head = {}
    for head in heads:
        manual_weights = _normalize_manual_class_weights(head, device)
        if manual_weights is not None:
            class_weights_by_head[head["name"]] = manual_weights
            continue

        if bool(head.get("auto_class_weight", False)):
            class_weights_by_head[head["name"]] = _compute_auto_class_weights(head, labels_by_head, device)
        else:
            class_weights_by_head[head["name"]] = None

    return class_weights_by_head


def build_criteria(heads, class_weights_by_head):
    return {
        head["name"]: nn.CrossEntropyLoss(
            weight=class_weights_by_head.get(head["name"]),
            ignore_index=-1,
            label_smoothing=float(head.get("label_smoothing", 0.0)),
        )
        for head in heads
    }


def evaluate(model, loader, criteria, heads, device):
    model.eval()
    total_loss = 0.0
    total_count = 0
    correct_by_head = {head["name"]: 0 for head in heads}
    valid_count_by_head = {head["name"]: 0 for head in heads}

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = {name: value.to(device) for name, value in targets.items()}

            logits_by_head = model(images)
            total_batch_loss = 0.0

            for head in heads:
                head_name = head["name"]
                head_weight = float(head.get("loss_weight", 1.0))
                head_loss = criteria[head_name](logits_by_head[head_name], targets[head_name])
                total_batch_loss = total_batch_loss + (head_loss * head_weight)

                predictions = logits_by_head[head_name].argmax(dim=1)
                valid_mask = targets[head_name] != -1
                valid_count_by_head[head_name] += valid_mask.sum().item()
                if valid_mask.any():
                    correct_by_head[head_name] += (predictions[valid_mask] == targets[head_name][valid_mask]).sum().item()

            batch_size = images.size(0)
            total_loss += total_batch_loss.item() * batch_size
            total_count += batch_size

    summary = {"loss": total_loss / max(total_count, 1)}
    for head in heads:
        head_name = head["name"]
        summary["{}_accuracy".format(head_name)] = correct_by_head[head_name] / max(valid_count_by_head[head_name], 1)

    return summary


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(cfg):
    seed = int(cfg.get("seed", 0))
    set_seed(seed)
    print(f"[INFO]: Random seed set to {seed}")

    metadata = build_metadata(cfg)
    device = normalize_device(cfg.get("device"))
    dataset_paths = load_dataset_paths(cfg)

    print("[INFO]: Train dataset loader is being created")
    loader, image_paths, train_labels_by_head = create_dataloader(cfg, metadata, dataset_paths, split="train", shuffle=True)
    print("[DONE]: Train dataset has been loaded")

    print("[INFO]: Validation dataset loader is being created")
    val_loader, val_image_paths, _ = create_dataloader(
        cfg,
        metadata,
        dataset_paths,
        split="val",
        batch_size=cfg.get("val_batch", cfg.get("batch", 16)),
        shuffle=False,
    )
    print("[DONE]: Validation dataset has been loaded")

    model = MultiHeadModel(
        heads=metadata["heads"],
        backbone_name=metadata["backbone"]["name"],
        pretrained=bool(metadata["backbone"]["pretrained"]),
    ).to(device)

    class_weights_by_head = build_class_weights(metadata["heads"], train_labels_by_head, device)
    criteria = build_criteria(metadata["heads"], class_weights_by_head)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("lr", 1e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )

    epochs = int(cfg.get("epochs", 10))
    steps_per_epoch = max(len(loader), 1)
    total_steps = epochs * steps_per_epoch
    warmup_epochs = float(cfg.get("warmup_epochs", 3.0))
    warmup_steps = int(cfg.get("warmup_steps", warmup_epochs * steps_per_epoch))
    min_lr_ratio = float(cfg.get("min_lr_ratio", 0.05))
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=build_lr_lambda(total_steps=total_steps, warmup_steps=warmup_steps, min_lr_ratio=min_lr_ratio),
    )
    ema = None
    if bool(cfg.get("use_ema", True)):
        ema = ModelEMA(
            model,
            decay=float(cfg.get("ema_decay", 0.9998)),
            tau=float(cfg.get("ema_tau", 2000)),
        )
    log_interval = int(cfg.get("log_interval", 10))
    project_dir = cfg.get("project_dir", "outputs")
    project_name = cfg.get("project_name", "multihead_classification")
    save_dir = os.path.join(project_dir, project_name)
    os.makedirs(save_dir, exist_ok=True)

    best_loss = float("inf")
    history = []
    best_path = os.path.join(save_dir, "best.pth")
    last_path = os.path.join(save_dir, "last.pth")

    print("[INFO]: Start custom multi-head training")
    print("[INFO]: backbone   = {}".format(metadata["backbone"]["name"]))
    print("[INFO]: pretrained = {}".format(bool(metadata["backbone"]["pretrained"])))
    print("[INFO]: train image dir = {}".format(dataset_paths.get("train_image_dir")))
    print("[INFO]: train label dir = {}".format(dataset_paths.get("train_label_dir")))
    print("[INFO]: val image dir   = {}".format(dataset_paths.get("val_image_dir")))
    print("[INFO]: val label dir   = {}".format(dataset_paths.get("val_label_dir")))
    print("[INFO]: train samples = {}".format(len(image_paths)))
    print("[INFO]: val samples   = {}".format(len(val_image_paths)))
    print("[INFO]: warmup steps = {}".format(warmup_steps))
    print("[INFO]: min lr ratio = {}".format(min_lr_ratio))
    print("[INFO]: ema enabled  = {}".format(bool(ema is not None)))
    print("[INFO]: heads      = {}".format(", ".join(head["name"] for head in metadata["heads"])))
    for head in metadata["heads"]:
        head_name = head["name"]
        weights = class_weights_by_head.get(head_name)
        if weights is None:
            print("[INFO]: class weights [{}] = disabled".format(head_name))
        else:
            weight_text = ", ".join("{:.3f}".format(float(weight)) for weight in weights.detach().cpu())
            print("[INFO]: class weights [{}] = {}".format(head_name, weight_text))
    print("[INFO]: save dir   = {}".format(save_dir))

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_count = 0
        correct_by_head = {head["name"]: 0 for head in metadata["heads"]}
        valid_count_by_head = {head["name"]: 0 for head in metadata["heads"]}

        for step, (images, targets) in enumerate(loader, start=1):
            images = images.to(device)
            targets = {name: value.to(device) for name, value in targets.items()}

            optimizer.zero_grad()
            logits_by_head = model(images)

            loss_by_head = {}
            total_batch_loss = 0.0
            for head in metadata["heads"]:
                head_name = head["name"]
                head_weight = float(head.get("loss_weight", 1.0))
                head_loss = criteria[head_name](logits_by_head[head_name], targets[head_name])
                loss_by_head[head_name] = head_loss
                total_batch_loss = total_batch_loss + (head_loss * head_weight)

            total_batch_loss.backward()
            optimizer.step()
            if ema is not None:
                ema.update(model)
            scheduler.step()

            batch_size = images.size(0)
            total_loss += total_batch_loss.item() * batch_size
            total_count += batch_size

            for head in metadata["heads"]:
                head_name = head["name"]
                predictions = logits_by_head[head_name].argmax(dim=1)
                valid_mask = targets[head_name] != -1
                valid_count_by_head[head_name] += valid_mask.sum().item()
                if valid_mask.any():
                    correct_by_head[head_name] += (predictions[valid_mask] == targets[head_name][valid_mask]).sum().item()

            if step % log_interval == 0 or step == len(loader):
                loss_text = " ".join(
                    "{}={:.4f}(x{:.2f})".format(
                        head_name,
                        loss_by_head[head_name].item(),
                        next(head["loss_weight"] for head in metadata["heads"] if head["name"] == head_name),
                    )
                    for head_name in loss_by_head
                )
                print(
                    "[Epoch {}/{}][Step {}/{}] total={:.4f} {}".format(
                        epoch + 1,
                        epochs,
                        step,
                        len(loader),
                        total_batch_loss.item(),
                        loss_text,
                    )
                )

        avg_loss = total_loss / max(total_count, 1)
        epoch_summary = {
            "epoch": epoch + 1,
            "train_loss": float(avg_loss),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }

        for head in metadata["heads"]:
            head_name = head["name"]
            epoch_summary["train_{}_accuracy".format(head_name)] = correct_by_head[head_name] / max(valid_count_by_head[head_name], 1)

        eval_model = ema.ema if ema is not None else model
        if val_loader is not None:
            val_summary = evaluate(eval_model, val_loader, criteria, metadata["heads"], device)
            epoch_summary["val_loss"] = float(val_summary["loss"])
            for head in metadata["heads"]:
                head_name = head["name"]
                epoch_summary["val_{}_accuracy".format(head_name)] = val_summary["{}_accuracy".format(head_name)]

        history.append(epoch_summary)

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": epoch_summary.get("val_loss", avg_loss),
            "metadata": metadata,
            "history": history,
        }
        if ema is not None:
            checkpoint["ema_state_dict"] = ema.ema.state_dict()
            checkpoint["ema_updates"] = ema.updates

        torch.save(checkpoint, last_path)

        train_metric_text = " ".join(
            "train_{}_acc={:.4f}".format(head["name"], epoch_summary["train_{}_accuracy".format(head["name"])])
            for head in metadata["heads"]
        )
        message = "[INFO]: epoch {} summary train_loss={:.4f} lr={:.6f} {}".format(
            epoch + 1,
            avg_loss,
            epoch_summary["lr"],
            train_metric_text,
        )
        if val_loader is not None:
            val_metric_text = " ".join(
                "val_{}_acc={:.4f}".format(head["name"], epoch_summary["val_{}_accuracy".format(head["name"])])
                for head in metadata["heads"]
            )
            message = "{} val_loss={:.4f} {}".format(message, epoch_summary["val_loss"], val_metric_text)
        print(message)

        monitor_loss = epoch_summary.get("val_loss", avg_loss)
        if monitor_loss < best_loss:
            best_loss = monitor_loss
            best_checkpoint = copy.deepcopy(checkpoint)
            torch.save(best_checkpoint, best_path)
            print("[INFO]: Saved best checkpoint to {}".format(best_path))

    save_yaml(
        os.path.join(save_dir, "args.yaml"),
        {
            "train_config": cfg,
            "metadata": metadata,
            "best_loss": float(best_loss),
            "weights": {"best": best_path, "last": last_path},
            "optimizer": {"lr": float(cfg.get("lr", 1e-4)), "warmup_steps": warmup_steps, "min_lr_ratio": min_lr_ratio},
            "ema": {"enabled": bool(ema is not None), "decay": float(cfg.get("ema_decay", 0.9998)), "tau": float(cfg.get("ema_tau", 2000))},
            "class_weights": {
                head_name: None if weights is None else [float(weight) for weight in weights.detach().cpu().tolist()]
                for head_name, weights in class_weights_by_head.items()
            },
        },
    )

    return {
        "save_dir": save_dir,
        "best_path": best_path,
        "last_path": last_path,
        "best_loss": float(best_loss),
        "history": history,
        "metadata": metadata,
    }


def train_from_config(cfg):
    return train(cfg)


def main():
    parser = argparse.ArgumentParser(description="Custom multi-head classification trainer")
    parser.add_argument("--config", required=True, help="Path to yaml config")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    cfg["config_dir"] = os.path.dirname(os.path.abspath(args.config))
    result = train_from_config(cfg)

    print("[INFO]: Training complete")
    print("[INFO]: save_dir  = {}".format(result["save_dir"]))
    print("[INFO]: best_path = {}".format(result["best_path"]))
    print("[INFO]: last_path = {}".format(result["last_path"]))


if __name__ == "__main__":
    main()
