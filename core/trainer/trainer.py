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
            "data"   : self.config["dataset"],
            "epochs" : self.config.get("epochs", 50),
            "imgsz"  : self.config.get("imgsz", 640),
            "batch"  : self.config.get("batch", 4),
            "device" : self.config.get("device", 0),
            "workers": self.config.get("workers", 4),
            "cache"  : self.config.get("cache", False),
            "freeze" : self.config.get("freeze", 0),

            "save_dir": prj_pull
        }

        extra_args = self.config.get("extra_args", {})

        for key in extra_args:
            train_args[key] = extra_args[key]

        print("[INFO] train args ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ")
        for k, v in train_args.items():
            print(f"{k}: {v}")

        self.model.train(**train_args)

class RFDETRTrainer():
    def __init__(self, config, logger=None):
        self.config = config

        model_size = config['model_size']
        model_path = config.get('model_path', None)
        num_class  = config.get('nc', None)

        self.model = self.build_model(model_size, model_path=model_path, num_class=num_class)

    def build_model(self, model_size, model_path=None, num_class=None):
        size = model_size.strip().lower()

        kwargs = {}

        if num_class is not None: kwargs["num_classes"] = num_class
        if model_path is not None: kwargs["pretrain_weights"] = model_path

        if size == "nano":
            from rfdetr import RFDETRNano
            return RFDETRNano(**kwargs)

        elif size == "small":
            from rfdetr import RFDETRSmall
            return RFDETRSmall(**kwargs)

        elif size == "medium":
            from rfdetr import RFDETRMedium
            return RFDETRMedium(**kwargs)

        elif size == "large":
            from rfdetr import RFDETRLarge
            return RFDETRLarge(**kwargs)

        elif size == "xlarge":
            from rfdetr import RFDETRXLarge
            return RFDETRXLarge(**kwargs)

        elif size == "2xlarge":
            from rfdetr import RFDETR2XLarge
            return RFDETR2XLarge(**kwargs)

        elif size == "base":
            from rfdetr import RFDETRBase
            return RFDETRBase(**kwargs)

        else:
            raise ValueError("Unknown model_size: %s" % model_size)

    def train(self):
        prj_dir = self.config.get("project_dir", "runs")
        prj_name = self.config.get("project_name", "train")
        prj_pull = os.path.join(prj_dir, prj_name)

        train_args = {
            "dataset_dir"     : self.config["dataset"],
            "epochs"          : self.config.get("epochs", 50),
            "short_size"      : self.config.get("imgsz", 640),
            "batch_size"      : self.config.get("batch", 4),
            "grad_accum_steps": self.config.get("grad_accum_steps", 1),
            "num_workers"     : self.config.get("workers", 2),
            "device"          : self.config.get("device", "cuda"),
            "lr"              : self.config.get("lr", 1e-4),

            "output_dir": prj_pull,
        }

        extra_args = self.config.get("extra_args", {})

        for key in extra_args:
            train_args[key] = extra_args[key]

        for k, v in train_args.items():
            print("%s: %s" % (k, str(v)))

        self.model.train(**train_args)