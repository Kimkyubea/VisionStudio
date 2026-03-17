# -*- coding:utf-8 -*-

import yaml

def load_config(cfg_path):
    with open(cfg_path, 'r', encoding="utf-8") as cf:
        cfg = yaml.safe_load(cf)

    return cfg
