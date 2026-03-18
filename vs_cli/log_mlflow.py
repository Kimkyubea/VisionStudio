# -*- coding:utf-8 -*-

import os, sys
import mlflow
import yaml
import json 

from core.logger.logger import VSMLflowLogger

def log_evaluation(cfg):
    print("[INFO]: Start logging to mlflow ...")

    logger = VSMLflowLogger(cfg)
    logger.log_eval_result()
    logger.log_train_cfg()

    print("[INFO]: End logging")

def log_release_note(cfg): pass
def upload_model(cfg): pass