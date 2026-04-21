# -*- coding:utf-8 -*-

import os
import cv2
import yaml
import numpy as np

from ultralytics import YOLO, settings
settings.update({"mlflow": False})

class UltralyticsDetectionTrainer():
    def __init__(self, config, logger=None):
        self.config = config
        self.model = YOLO(self.config["model"])

    def train(self):
        run_root = os.path.join(
            self.config.get("project_dir", "runs"),
            self.config.get("project_name", "train_detection"),
        )
        artifact_name = self.config.get("artifact_dir_name", "train_artifacts")

        train_args = {
            "data"   : self.config["dataset"],
            "epochs" : self.config.get("epochs", 50),
            "imgsz"  : self.config.get("imgsz", 640),
            "batch"  : self.config.get("batch", 4),
            "device" : self.config.get("device", 0),
            "workers": self.config.get("workers", 4),
            "cache"  : self.config.get("cache", False),
            "freeze" : self.config.get("freeze", 0),
            # "project": run_root,
            # "name"   : artifact_name,
            "save_dir" : os.path.join(run_root, artifact_name)
        }

        extra_args = self.config.get("extra_args", {})

        for key in extra_args:
            train_args[key] = extra_args[key]

        print("[INFO] train args ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ")
        for k, v in train_args.items():
            print(f"{k}: {v}")

        self.model.train(**train_args)

class UltralyticsClassificationTrainer():
    def __init__(self, config, logger=None):
        self.config = config
        self.model = YOLO(self.config["model"])

    def train(self):
        run_root = os.path.join(
            self.config.get("project_dir", "runs"),
            self.config.get("project_name", "train_classification"),
        )
        artifact_name = self.config.get("artifact_dir_name", "train_artifacts")

        train_args = {
            "data"   : self.config["dataset"],
            "epochs" : self.config.get("epochs", 50),
            "imgsz"  : self.config.get("imgsz", 224),
            "batch"  : self.config.get("batch", 4),
            "device" : self.config.get("device", 0),
            "workers": self.config.get("workers", 4),
            "cache"  : self.config.get("cache", False),

            "project": run_root,
            "name"   : artifact_name
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

    def _save_yaml(self, path, data):
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump( data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

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

        config_obj = self.model.get_train_config(**train_args).model_dump()
        save_args_path = os.path.join(prj_pull, "args.yaml")
        self._save_yaml(save_args_path, config_obj)

        _original_imread = cv2.imread

        def _safe_imread(path, flags=cv2.IMREAD_COLOR):
            try:
                data = np.fromfile(path, dtype=np.uint8)
                if data.size == 0: return None
                return cv2.imdecode(data, flags)
            except Exception: return None

        cv2.imread = _safe_imread

        try:
            self.model.train(**train_args)
        finally: cv2.imread = _original_imread

class CoDETRTrainer:
    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger

    def train(self):
        print("[INFO] Co-DETR training is not implemented yet")
        print("[INFO] Placeholder for future MMDetection integration")

        return


#################################################
##############  Customized trainer ##############
#################################################

class MultiHeadClassificationTrainer:
    def __init__ (self, config, logger=None):
        self.config = config
        self.logger = logger

    def train(self):
        from custom_trainer.multihead_classification.trainer import train

        result = train(self.config)

        print("[INFO]: Custom multi-head classification training DONE")
        print("[INFO]: save_dir  = {}".format(result['save_dir']))
        print("[INFO]: best_path = {}".format(result['bast_path']))
        print("[INFO]: last_path = {}".format(result['last_path']))


class MultiHeadDetectionTrainer:
    def __init__ (self, config, logger=None):
        self.config = config
        self.logger = logger

    def train(self):
        from custom_trainer.multihead_detection.trainer import train

        result = train(self.config)

        print("[INFO]: Custom multi-head detection training DONE")
        print("[INFO]: save_dir  = {}".format(result['save_dir']))
        print("[INFO]: best_path = {}".format(result['best_path']))
        print("[INFO]: last_path = {}".format(result['last_path']))
