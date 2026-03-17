# -*- coding:utf-8 -*-

import os, sys

from vs_cli.train import run_train
from vs_cli.visualize import run_visualize
from vs_cli.evaluate import run_evaluate
from utils.configuration_manager import load_config

def main():
    if len(sys.argv) < 3: 
        print("[ERROR]: Check ther entered arguments.")
        return

    cmd = sys.argv[1]
    config_path = sys.argv[2]
    print("[INFO]: Loading configurations from {} ... ".format(config_path))
    cfg = load_config(config_path)
    print("[INFO]: Configuration loading DONE")

    if cmd == "train":
        run_train(cfg)
        return

    elif cmd == "visualize":
        run_visualize(cfg)
        return

    elif cmd == "evaluate":
        run_evaluate(cfg)
        return

if __name__ == "__main__":
    main()
