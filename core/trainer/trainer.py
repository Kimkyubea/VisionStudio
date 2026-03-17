# -*- coding:utf-8 -*-

from ultralytics import YOLO

class UltralyticsTrainer():
    def __init__(self, config, logger=None):
        self.config = config
        self.model = YOLO(self.config["model"])

    def train(self):
        self.model.train(
            data    = self.config["dataset"], # Yolo yaml file path
            epochs  = self.config.get("epochs", 50),
            imgsz   = self.config.get("imgsz", 640),
            batch   = self.config.get("batch", 4),
            device  = self.config.get("device", 0),
            project = self.config.get("project", "proejct_name"),
            workers = self.config.get("workers", 4),
            cache   = self.config.get("cache", False),

            freeze  = self.config.get("freeze", 5)
        )

        # TODO # 
        # logging #

class RFDETRTrainer():
    def __init__(self, config, logger=None):
        self.config = config
