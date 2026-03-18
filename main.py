# -*- coding:utf-8 -*-

import os, sys

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
        from vs_cli.train import run_train
        run_train(cfg)
        return

    elif cmd == "visualize":
        from vs_cli.visualize import run_visualize
        run_visualize(cfg)
        return

    elif cmd == "evaluate":
        from vs_cli.evaluate import run_evaluate
        run_evaluate(cfg)
        return

    elif cmd == "log_eval":
        from vs_cli.log_mlflow import log_evaluation
        log_evaluation(cfg)

    # elif cmd == "log_release_note":
    #     from vs_cli.log_mlflow import log_release_note

    # elif cmd == "upload_modle":
    #     from vs_cli.log_mlflow import upload_model

if __name__ == "__main__":
    main()
