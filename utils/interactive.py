# -*- coding:utf-8 -*-

import os
from InquirerPy import inquirer

def select_command():
    cmd = inquirer.select(
        message="Select Command",
        choices=[
            "train",
            "evaluate",
            "visualize",
            "export",
            "log_eval",
            "log_release"
        ],
    ).execute()

    return cmd

def select_config(config_dir="configs"):
    if not os.path.exists(config_dir):
        print("[ERROR] Config directory not found:", config_dir)
        return ""

    files = []

    for f in os.listdir(config_dir):
        if f.endswith(".yaml") or f.endswith(".yml"):
            files.append(os.path.join(config_dir, f))

    if len(files) == 0:
        print("[ERROR] No config files found")
        return ""

    cfg = inquirer.select(
        message="Select Config File",
        choices=files
    ).execute()

    return cfg