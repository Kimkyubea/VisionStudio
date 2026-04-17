# -*- coding:utf-8 -*-

import os, sys

from utils.configuration_manager import VisionConfigManager
from utils.dataset_linker import build_dataset_tasks, link_dataset_tasks, bulk_unlink, write_link_manifest
from utils.interactive import select_command, select_config

def main(cmd, config_path):
    print("[INFO]: Loading configurations from {} ... ".format(config_path))
    cfg_manager = VisionConfigManager.from_file(config_path)
    print("[INFO]: Configuration loading DONE")

    cfg = cfg_manager.get_runtime_config()

    if cmd == "train":
        from vs_cli.train import run_train

        dataset_tasks = build_dataset_tasks(cfg_manager.cfg, cfg_manager.data_cfg)
        link_dataset_tasks(dataset_tasks)
        write_link_manifest(cfg_manager.cfg, dataset_tasks)

        cfg = cfg_manager.build_hooked_train_config()
        cfg_manager.dump_runtime_config(cfg)

        run_train(cfg)

        bulk_unlink(cfg_manager.dataset_dir)

    elif cmd == "visualize":
        from vs_cli.visualize import run_visualize
        run_visualize(cfg)

    elif cmd == "evaluate":
        from vs_cli.evaluate import run_evaluate
        run_evaluate(cfg)

    elif cmd == "export":
        from vs_cli.export import run_export
        run_export(cfg)

    elif cmd == "log_eval":
        from vs_cli.log_mlflow import log_evaluation
        log_evaluation(cfg)

    elif cmd == "log_release":
        from vs_cli.log_mlflow import log_release_note
        log_release_note(cfg)

    elif cmd == "upload_model":
        from vs_cli.log_mlflow import upload_model
        upload_model(cfg)

    elif cmd == "log_model":
        from vs_cli.log_mlflow import log_model
        log_model(cfg)

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        cmd = sys.argv[1]
        config_path = sys.argv[2]

    else:
        print("[INFO] Enter Interactive Mode")
        cmd = select_command()
        config_path = select_config()

    main(cmd, config_path)