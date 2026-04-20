# -*- coding:utf-8 -*-

import os
from InquirerPy import inquirer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

    config_dir = os.path.join(BASE_DIR, config_dir)

    print("[INFO] config_dir:", config_dir)

    if not os.path.exists(config_dir):
        print("[ERROR] Config directory not found:", config_dir)
        return ""

    files = []

    # 🔥 하위 폴더까지 전부 탐색
    for root, dirs, filenames in os.walk(config_dir):
        for f in filenames:
            if f.lower().endswith(".yaml") or f.lower().endswith(".yml"):
                full_path = os.path.join(root, f)

                # 👉 보기 좋게 상대경로로 변환
                rel_path = os.path.relpath(full_path, BASE_DIR)

                files.append(rel_path)

    if len(files) == 0:
        print("[ERROR] No config files found")
        return ""

    # 정렬 (가독성)
    files.sort()

    cfg = inquirer.fuzzy(
        message="Search Config File",
        choices=files,
    ).execute()

    return cfg