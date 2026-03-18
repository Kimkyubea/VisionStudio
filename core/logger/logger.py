# -*- coding:utf-8 -*-

import os, sys
import mlflow

from datetime import datetime
from mlflow.tracking import MlflowClient

from utils.common import load_json, load_yaml

class VSMLflowLogger:
    def __init__(self, config):
        self.config = config

        self.trk_uri = self.config.get("tracking_uri", "")
        self.exp_name = self.config.get("experiment_name", "")
        self.run_name = self.config.get("run_name", "")
        self.eval_ds_key = self.config.get("eval_ds_key", "")
        self.work_dir = self.config.get("work_dir", ".")
        self.eval_file = os.path.join(self.work_dir, f'{self.config.get("result_name", "evaluation_result")}.json')
        self.train_cfg_file = os.path.join(
            self.work_dir, 
            f'{self.config.get("cfg_file_name", "args")}.{self.config.get("cfg_file_ext", "yaml")}'
        )

        if self.trk_uri     == "": raise Exception("[ERROR]: Tracking URI is BLANK")
        if self.exp_name    == "": raise Exception("[ERROR]: Experiment name is BLANK")
        if self.run_name    == "": raise Exception("[ERROR]: Run name is BLANK")
        if self.eval_ds_key == "": raise Exception("[ERROR]: Evaluation dataset key is BLANK")

        self.mlclient = MlflowClient(tracking_uri=self.trk_uri)

        self.ensure_run_id()

    def find_run_id_by_name_and_tag(self, tag_key):
        filter_str = "attributes.run_name = '{}' and tags.{} = '{}'".format(self.run_name, tag_key, self.eval_ds_key)

        runs = self.mlclient.search_runs(
            experiment_ids = [self.exp_id],
            filter_string = filter_str,
            order_by=["attributes.start_time DESC"],
            max_results = 1
        )

        if runs is None or len(runs) == 0: return ""

        return runs[0].info.run_id

    def ensure_run_id(self): 
        mlflow.set_tracking_uri(self.trk_uri)
        exp = self.mlclient.get_experiment_by_name(self.exp_name)
        if exp is None: 
            self.exp_id = self.mlclient.create_experiment(self.exp_name)
            print("[INFO] Created new experiment:", self.exp_name, "experiment_id:", self.exp_id)
        else: 
            self.exp_id = exp.experiment_id
            print("[INFO] Found existing experiment:", self.exp_name, "experiment_id:", self.exp_id)

        self.run_id = self.find_run_id_by_name_and_tag("eval_dataset")

        if self.run_id is not None and len(self.run_id) > 0:
            print("[INFO] Found existing run:", self.run_name, "run_id:", self.run_id)
            return
        
        with mlflow.start_run(run_name=self.run_name, experiment_id=self.exp_id) as r:
            mlflow.set_tag("eval_dataset", self.eval_ds_key)
            self.run_id = r.info.run_id
            print("[INFO] Created new run:", self.run_name, "run_id:", self.run_id)

    def log_eval_result(self):
        eval_met = load_json(self.eval_file)

        with mlflow.start_run(run_id = self.run_id):
            for k, v in eval_met.items():
                mlflow.log_metric(k, float(v))

        print("[DONE] Eval metrics logged to run_id:", self.run_id)

    def log_train_cfg(self):
        args_d = load_yaml(self.train_cfg_file)

        with mlflow.start_run(run_id = self.run_id):
            mlflow.log_dict(args_d, artifact_file="train_configs/train_cfg.json")

        print("[DONE] Train config logged to run_id:", self.run_id)

    def log_release_note(self, release_info):
        def _build_release_md(release_info):
            date   = release_info.get("date", "")
            notes  = release_info.get("notes", [])
            author = release_info.get("author", "FODICS")

            s = ""
            s += "## Release Note\n\n"

            if date: s += "** Date **\n{}\n\n".format(date)
            if author: s+= "** Author **\n{}\n\n".format(author)

            s += "** Notes **\n"

            for sen in notes:
                s += "- {}\n".format(sen)

            s += "\n"

            return s

        with mlflow.start_run(run_id = self.run_id):
            md_text = _build_release_md(release_info)
            mlflow.set_tag("released", "true")
            mlflow.set_tag("release_date", release_info.get("date", ""))

            mlflow.log_text(md_text, artifact_file="release_note/RELEASE.md")

        print("[DONE] Release note artifact added to run_id:", self.run_id)

    def upload_models(self, model_arts):
        with mlflow.start_run(run_id = self.run_id):
            for target_model in model_arts:
                file_name = os.path.basename(target_model)
                print('[INFO]: Uploading {} ...'.format(file_name))

                mlflow.log_artifact(target_model, artifact_path="model")

            up_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            mlflow.set_tag("model_uploaded_time", up_time)

        print("[DONE] Model artifacts uploaded to run: {}".format(self.run_name))




            
