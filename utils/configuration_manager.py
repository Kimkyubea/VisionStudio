# -*- coding:utf-8 -*-

import copy
import os
import yaml

class VisionConfigManager:
    def __init__(self, cfg):
        self.cfg = copy.deepcopy(cfg)
        data_src_yaml = self.cfg.get("dataset", "")
        if data_src_yaml != "": self.data_cfg = self.load_config(self.cfg.get("dataset", ""))

    @classmethod
    def from_file(cls, cfg_path):
        return cls(cls.load_config(cfg_path))

    @staticmethod
    def load_config(cfg_path):
        if cfg_path == "":
            raise Exception("[ERROR]: Config file path is BLANK")

        with open(cfg_path, "r", encoding="utf-8") as cf:
            cfg = yaml.safe_load(cf)

        return cfg

    @staticmethod
    def dump_config(dst_path, cfg):
        dst_dir = os.path.dirname(dst_path)
        if dst_dir:
            os.makedirs(dst_dir, exist_ok=True)

        with open(dst_path, "w", encoding="utf-8") as df:
            yaml.dump(cfg, df, indent=4, default_flow_style=False, sort_keys=False)

    @property
    def framework(self):
        return self.cfg.get("framework", "")

    @property
    def project_dir(self):
        return self.cfg["project_dir"]

    @property
    def project_name(self):
        return self.cfg["project_name"]

    @property
    def work_dir(self):
        return os.path.abspath(os.path.join(self.project_dir, self.project_name))

    @property
    def dataset_dir(self):
        return os.path.abspath(os.path.join(self.work_dir, "dataset"))

    @property
    def artifact_dir_name(self):
        return self.cfg.get("artifact_dir_name", "train_artifacts")

    @property
    def artifact_dir(self):
        return os.path.abspath(os.path.join(self.work_dir, self.artifact_dir_name))

    def get_runtime_config(self):
        return copy.deepcopy(self.cfg)

    def build_hooked_train_config(self):
        hooked_cfg = copy.deepcopy(self.cfg)
        hooked_data_cfg = copy.deepcopy(self.data_cfg)

        hooked_data_cfg["train"] = os.path.join("train", "images")
        hooked_data_cfg["val"] = os.path.join("valid", "images")

        hooked_data_config_file = os.path.abspath(os.path.join(self.dataset_dir, "data.yaml"))
        self.dump_config(hooked_data_config_file, hooked_data_cfg)

        if hooked_cfg["framework"] in ["ultralytics", "custom_multihead"]:
            hooked_cfg["dataset"] = hooked_data_config_file
        elif hooked_cfg["framework"] == "rfdetr":
            hooked_cfg["dataset"] = os.path.abspath(self.dataset_dir)
        else:
            raise Exception("[ERROR]: Unsupported framework: {}".format(hooked_cfg["framework"]))

        return hooked_cfg

    def dump_runtime_config(self, cfg, file_name="VS_train_cfgs.yaml"):
        dump_path = os.path.abspath(os.path.join(self.work_dir, file_name))
        self.dump_config(dump_path, cfg)
        return dump_path

def load_config(cfg_path):
    return VisionConfigManager.load_config(cfg_path)

def dump_config(dst_path, cfg):
    VisionConfigManager.dump_config(dst_path, cfg)
