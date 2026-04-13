# -*- coding:utf-8 -*-

import os, sys

from core.logger.logger import VSMLflowLogger

def log_evaluation(cfg):
    print("[INFO]: Start logging to mlflow ...")

    logger = VSMLflowLogger(cfg)
    logger.log_eval_result()
    log_cfg_option = cfg.get('log_config', False)
    if log_cfg_option: logger.log_train_cfg()

    print("[INFO]: End logging")

def log_release_note(cfg):
    logger = VSMLflowLogger(cfg)
    release_d = cfg.get("release", {})
    logger.log_release_note(release_d)

def upload_model(cfg):
    logger = VSMLflowLogger(cfg)
    model_artifacts = cfg.get("model_artifacts", [])
    logger.upload_models(model_artifacts)

def log_model(cfg):
    logger = VSMLflowLogger(cfg)
    registered_models = cfg.get("registered_models", [])
    logger.register_onnx_models(registered_models)