# -*- coding:utf-8 -*-

import os
import torch

from ultralytics import YOLO

class UltralyticsExportor:
    def __init__(self, config, logger=None):
        self.config = config
        self.model_path = self.config.get("model_path", "")
        if self.model_path == "": raise Exception("[ERROR]: Model path is BLANK")

        self.model = YOLO(self.model_path)

    def export(self):
        batch_cfg = self.config.get("batch", 1)

        if isinstance(batch_cfg, int): batch_list = [batch_cfg]
        else: batch_list = batch_cfg

        save_dir = self.config.get("export_dir", os.path.dirname(self.model_path))

        for batch_size in batch_list:
            export_args = {
                "format": "onnx",
                "imgsz": self.config.get("img_sz", 640),
                "batch": batch_size,
                "opset": self.config.get("opset", 12),
            }

            extra_args = self.config.get("extra_args", {})

            for key in extra_args:
                export_args[key] = extra_args[key]

            print("[INFO]: export args batch size: {} ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ".format(batch_size))
            for k, v in export_args.items():
                print(f"{k}: {v}")

            out_path = self.model.export(**export_args)

            new_name = "model_bsz{}.onnx".format(batch_size)
            new_path = os.path.join(save_dir, new_name)

            if os.path.exists(out_path): 
                os.rename(out_path, new_path)
                print("[INFO]: Export Done {}".format(new_path))

            else: print("[WARN]: Export output not found {}".format(out_path))

class RFDETRExporter:
    def __init__(self, config, logger=None):
        self.config = config 
        self.model_path = self.config.get("model_path", "")
        if self.model_path == "": raise Exception("[ERROR]: Model path is BLANK")
        model_size = self.config.get("model_size", "")
        if model_size == "": raise Exception("[ERROR]: Model size is BLANK")

        num_class  = self.config.get("nc", None)

        self.model = self.build_model(model_size, model_path=self.model_path, num_class=num_class)

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

    def export(self):
        batch_cfg = self.config.get("batch", 1)

        if isinstance(batch_cfg, int): batch_list = [batch_cfg]
        else: batch_list = batch_cfg

        save_dir = self.config.get("export_dir", os.path.dirname(self.model_path))
        # img_sz = self.config.get("img_sz", 560)
        simplify = self.config.get("simplify", False)

        for batch_size in batch_list:
            export_args = {
                "output_dir": save_dir,
                # "shape": (img_sz, img_sz),
                "opset_version": self.config.get("opset", 17),
                "simplify": simplify,
            }

            extra_args = self.config.get("extra_args", {})
            for key in extra_args:
                export_args[key] = extra_args[key]

            print("[INFO]: export args batch size: {} ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ".format(batch_size))
            for k, v in export_args.items():
                print(f"{k}: {v}")

            self.model.export(**export_args)
            out_path = os.path.join(save_dir, "inference_model.onnx")

            new_name = "model_bsz{}.onnx".format(batch_size)
            new_path = os.path.join(save_dir, new_name)

            if os.path.exists(out_path): 
                os.rename(out_path, new_path)
                print("[INFO]: Export Done {}".format(new_path))

            else: print("[WARN]: Export output not found {}".format(out_path))
