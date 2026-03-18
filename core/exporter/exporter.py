# -*- coding:utf-8 -*-

import os
from ultralytics import YOLO

class UltralyticsExportor:
    def __init__(self, config, logger=None):
        self.config = config
        self.model_path = self.config.get("model_path", "")
        if self.model_path == "": raise Exception("[ERROR]: Model path is BLANK")

        self.model = YOLO(self.config['model_path'])

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

            print("[INFO] export args batch size: {} ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ".format(batch_size))
            for k, v in export_args.items():
                print(f"{k}: {v}")

            out_path = self.model.export(**export_args)

            new_name = "model_bsz{}.onnx".format(batch_size)
            new_path = os.path.join(save_dir, new_name)

            if os.path.exists(out_path): 
                os.rename(out_path, new_path)
                print("[INFO]: Export Done {}".format(new_path))

            else: print("[WARN]: Export output not found {}".format(out_path))

class RFDETRExporter: pass


        
