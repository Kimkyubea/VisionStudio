# -*- coding:utf-8 -*-

import os, sys
import argparse

from utils.configuration_manager import load_config

def get_parser():

    parser = argparse.ArgumentParser(
        description="VisionStudio CLI"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # -------------------------------
    # train
    parser_train = subparsers.add_parser("train", help="Train model")
    parser_train.add_argument("config", help="Path to config file")

    # -------------------------------
    # visualize
    parser_vis = subparsers.add_parser("visualize", help="Visualize results")
    parser_vis.add_argument("config", help="Path to config file")

    # -------------------------------
    # evaluate
    parser_eval = subparsers.add_parser("evaluate", help="Evaluate model")
    parser_eval.add_argument("config", help="Path to config file")

    # -------------------------------
    # export
    parser_export = subparsers.add_parser("export", help="Export model")
    parser_export.add_argument("config", help="Path to config file")

    # -------------------------------
    # log_eval
    parser_log_eval = subparsers.add_parser("log_eval", help="Log evaluation to MLflow")
    parser_log_eval.add_argument("config", help="Path to config file")

    # -------------------------------
    # log_release
    parser_log_rel = subparsers.add_parser("log_release", help="Log release note to MLflow")
    parser_log_rel.add_argument("config", help="Path to config file")

    # -------------------------------
    # upload_model
    parser_upload = subparsers.add_parser("upload_model", help="Upload model to MLflow")
    parser_upload.add_argument("config", help="Path to config file")

    # -------------------------------
    # log_model
    parser_log_model = subparsers.add_parser("log_model", help="Log a model artifact to MLflow")
    parser_log_model.add_argument("config", help="Path to config file")
    return parser


def main(args):

    if args.command is None:
        print("[ERROR]: Please input command")
        return

    print("[INFO]: Loading configurations from {} ... ".format(args.config))
    cfg = load_config(args.config)
    print("[INFO]: Configuration loading DONE")

    # -------------------------------
    if args.command == "train":
        from vs_cli.train import run_train
        run_train(cfg)

    elif args.command == "visualize":
        from vs_cli.visualize import run_visualize
        run_visualize(cfg)

    elif args.command == "evaluate":
        from vs_cli.evaluate import run_evaluate
        run_evaluate(cfg)

    elif args.command == "export":
        from vs_cli.export import run_export
        run_export(cfg)

    elif args.command == "log_eval":
        from vs_cli.log_mlflow import log_evaluation
        log_evaluation(cfg)

    elif args.command == "log_release":
        from vs_cli.log_mlflow import log_release_note
        log_release_note(cfg)

    elif args.command == "upload_model":
        from vs_cli.log_mlflow import upload_model
        upload_model(cfg)

    elif args.command == "log_model":
        from vs_cli.log_mlflow import log_model
        log_model(cfg)


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
