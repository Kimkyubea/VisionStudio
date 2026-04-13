# -*- coding:utf-8 -*-

import os, sys
import onnx
import glob
import mlflow
import numpy as np

from datetime import datetime
from onnx import TensorProto
from mlflow.models import ModelSignature
from mlflow.tracking import MlflowClient
from mlflow.types.schema import Schema, TensorSpec

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
        self.sample_dir = os.path.join(self.work_dir, self.config.get("sample_dir", "eval_samples"))
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
        if os.path.exists(self.sample_dir): self.upload_inf_results()

    def upload_inf_results(self):
        smp_files = sorted(glob.glob('{}/*'.format(self.sample_dir)))
        with mlflow.start_run(run_id = self.run_id):
            for smp_file in smp_files:
                file_name = os.path.basename(smp_file)
                print('[INFO]: Uploading inference result sample image {}'.format(file_name))

                mlflow.log_artifact(smp_file, artifact_path="samples")

        print('[DONE]: Inference sample files uploaded to run: {}'.format(self.run_name))

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

    def register_onnx_models(self, model_entries):
        if model_entries is None or len(model_entries) == 0:
            raise Exception("[ERROR]: registered_models is EMPTY")

        with mlflow.start_run(run_id=self.run_id):
            for idx, model_entry in enumerate(model_entries):
                if not isinstance(model_entry, dict):
                    raise Exception("[ERROR]: registered_models[{}] must be dict".format(idx))

                model_path = model_entry.get("model_path", "")
                registered_model_name = model_entry.get("registered_model_name", "")
                alias = model_entry.get("alias", "")
                description = model_entry.get("description", "")
                await_registration_for = model_entry.get("await_registration_for", 300)
                model_tags = model_entry.get("tags", {})
                arti_path = model_entry.get("arti_path", "")

                if model_path == "":
                    raise Exception("[ERROR]: model_path is BLANK in registered_models[{}]".format(idx))
                if not os.path.exists(model_path):
                    raise Exception("[ERROR]: Model path does not exist: {}".format(model_path))
                if not model_path.lower().endswith(".onnx"):
                    raise Exception("[ERROR]: Only ONNX model registration is supported: {}".format(model_path))
                if registered_model_name == "":
                    raise Exception("[ERROR]: registered_model_name is BLANK in registered_models[{}]".format(idx))

                print("[INFO]: Registering ONNX model {} -> {}".format(model_path, registered_model_name))
                onnx_model = onnx.load(model_path)
                model_signature = self._build_onnx_signature(onnx_model)
                model_info = mlflow.onnx.log_model(
                    onnx_model=onnx_model,
                    # artifact_path=artifact_path,
                    registered_model_name=registered_model_name,
                    signature=model_signature,
                    await_registration_for=await_registration_for,
                    tags=model_tags,
                    name=arti_path
                )

                latest_version = self._find_latest_model_version(registered_model_name)
                if latest_version is None:
                    print("[WARN]: Registered model version lookup failed for {}".format(registered_model_name))
                    continue
 
                version = latest_version.version
                shortcut_tags = self._build_model_version_shortcuts(latest_version)
 
                version_description = self._build_model_version_description(description, shortcut_tags)
                if version_description:
                    self.mlclient.update_model_version(
                        name=registered_model_name,
                        version=version,
                        description=version_description
                    )

                if alias:
                    self.mlclient.set_registered_model_alias(
                        name=registered_model_name,
                        alias=alias,
                        version=version
                    )

                self.mlclient.set_model_version_tag(
                    name=registered_model_name,
                    version=version,
                    key="experiment_name",
                    value=self.exp_name
                )

                self.mlclient.set_model_version_tag(
                    name=registered_model_name,
                    version=version,
                    key="run_name",
                    value=self.run_name
                )

                for tag_key, tag_value in shortcut_tags.items():
                    self.mlclient.set_model_version_tag(
                        name=registered_model_name,
                        version=version,
                        key=tag_key,
                        value=tag_value
                    )
 
                print("[DONE] Registered model {} version {}".format(registered_model_name, version))
                if model_info is not None and getattr(model_info, "model_uri", None):
                    print("[INFO]: Model URI = {}".format(model_info.model_uri))

    def _build_model_version_shortcuts(self, model_version):
        run_id = getattr(model_version, "run_id", "") or self.run_id
        source = getattr(model_version, "source", "") or ""
        base_url = self.trk_uri.rstrip("/")

        shortcuts = {}

        if self.exp_name:
            shortcuts["experiment_name"] = self.exp_name
        if self.run_name:
            shortcuts["run_name"] = self.run_name

        if base_url.startswith("http://") or base_url.startswith("https://"):
            shortcuts["experiment_url"] = "{}/#/experiments/{}".format(base_url, self.exp_id)
            shortcuts["run_url"] = "{}/#/experiments/{}/runs/{}".format(base_url, self.exp_id, run_id)

        if source:
            shortcuts["model_uri"] = source

        return shortcuts

    def _build_model_version_description(self, description, shortcut_tags):
        lines = []
        if description:
            lines.append(description.strip())

        shortcut_keys = [
            ("Experiment", "experiment_url"),
            ("Run", "run_url"),
        ]
        shortcut_lines = []
        for label, key in shortcut_keys:
            value = shortcut_tags.get(key, "")
            if value:
                shortcut_lines.append("- {}: {}".format(label, value))

        if shortcut_lines:
            if lines:
                lines.append("")
            lines.append("### Shortcuts")
            lines.extend(shortcut_lines)

        return "\n".join(lines).strip()

    def _find_latest_model_version(self, registered_model_name):
        versions = self.mlclient.search_model_versions(
            filter_string="name = '{}'".format(registered_model_name),
            order_by=["version_number DESC"],
            max_results=1
        )

        if versions is None or len(versions) == 0:
            return None

        return versions[0]

    def _build_onnx_signature(self, onnx_model):
        input_specs = self._build_tensor_specs(onnx_model.graph.input)
        output_specs = self._build_tensor_specs(onnx_model.graph.output)
        return ModelSignature(inputs=Schema(input_specs), outputs=Schema(output_specs))

    def _build_tensor_specs(self, value_infos):
        tensor_specs = []
        for value_info in value_infos:
            tensor_type = value_info.type.tensor_type
            np_dtype = self._onnx_elem_type_to_numpy_dtype(tensor_type.elem_type)
            shape = []
            for dim in tensor_type.shape.dim:
                if dim.HasField("dim_value"):
                    shape.append(int(dim.dim_value))
                else:
                    shape.append(-1)
            tensor_specs.append(TensorSpec(np_dtype, tuple(shape), value_info.name))
        return tensor_specs

    def _onnx_elem_type_to_numpy_dtype(self, elem_type):
        type_map = {
            TensorProto.FLOAT: np.dtype(np.float32),
            TensorProto.UINT8: np.dtype(np.uint8),
            TensorProto.INT8: np.dtype(np.int8),
            TensorProto.UINT16: np.dtype(np.uint16),
            TensorProto.INT16: np.dtype(np.int16),
            TensorProto.INT32: np.dtype(np.int32),
            TensorProto.INT64: np.dtype(np.int64),
            TensorProto.BOOL: np.dtype(np.bool_),
            TensorProto.FLOAT16: np.dtype(np.float16),
            TensorProto.DOUBLE: np.dtype(np.float64),
            TensorProto.UINT32: np.dtype(np.uint32),
            TensorProto.UINT64: np.dtype(np.uint64),
        }
        if elem_type not in type_map:
            raise Exception("[ERROR]: Unsupported ONNX tensor elem_type: {}".format(elem_type))
        return type_map[elem_type]
