# -*- coding:utf-8 -*-

import os
from ultralytics import YOLO, settings
settings.update({"mlflow": False})

class UltralyticsTrainer():
    def __init__(self, config, logger=None):
        self.config = config
        self.model = YOLO(self.config["model"])

    def train(self):
        prj_dir = self.config.get("project_dir", "runs")
        prj_name = self.config.get("project_name", "train")
        prj_pull = os.path.join(prj_dir, prj_name)

        train_args = {
            "data": self.config["dataset"],
            "epochs": self.config.get("epochs", 50),
            "imgsz": self.config.get("imgsz", 640),
            "batch": self.config.get("batch", 4),
            "device": self.config.get("device", 0),
            "workers": self.config.get("workers", 4),
            "cache": self.config.get("cache", False),
            "freeze": self.config.get("freeze", 0),

            "save_dir": prj_pull
        }

        extra_args = self.config.get("yolo_args", {})

        for key in extra_args:
            train_args[key] = extra_args[key]

        print("[INFO] train args ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ")
        for k, v in train_args.items():
            print(f"{k}: {v}")

        self.model.train(**train_args)

class RFDETRTrainer():
    def __init__(self, config, logger=None):
        self.config = config
