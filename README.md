# VisionStudio

<div align="center">
  <img src="./docs/logo_wo_bg.png" alt="VisionStudio Logo" width="300">
</div>

VisionStudio is a CLI-based platform for Vision AI workflows. It provides a unified interface for training, evaluation, visualization, export, and experiment tracking across multiple model backends.

## Overview

VisionStudio is designed around a config-driven CLI:

```text
VisionStudio = Training + Evaluation + Visualization + Export + Tracking
```

Currently supported integrations include:
- Ultralytics detection
- Ultralytics classification
- RF-DETR detection
- Custom multi-head classification

## Architecture

<div align="left">
  <img src="./docs/VS_architecture_v0.png" alt="VisionStudio Architecture" width="750">
</div>

VisionStudio keeps CLI entrypoints and framework-specific implementations separated.

- `main.py`: top-level CLI
- `vs_cli/`: command entrypoints such as `train`, `visualize`, `evaluate`
- `core/`: trainer, predictor, visualizer, exporter, logging modules
- `custom_trainer/`: custom model packages and trainer implementations
- `utils/`: common helpers

## Directory Structure

<div align="left">
  <img src="./docs/directory_structure.png" alt="VisionStudio Directory Structure" width="750">
</div>

```text
core/            Core training, inference, evaluation, export, and logging modules
vs_cli/          CLI command entry points
custom_trainer/  Custom model packages and trainers
utils/           Common utilities
docs/            Documentation and architecture assets
env/             Environment setup files
```

## Installation

```powershell
conda env create --file environment.yml
conda activate <your_env_name>
pip install -r requirements.txt
```

## CLI Usage

```powershell
python main.py --help
```

## Commands

| Command | Description |
| --- | --- |
| `train` | Train a model |
| `evaluate` | Evaluate a model |
| `visualize` | Visualize prediction results |
| `export` | Export a model |
| `log_eval` | Log evaluation results to MLflow |
| `log_release` | Log release notes to MLflow |
| `upload_model` | Upload model artifacts to MLflow |
| `log_model` | Log model metadata to MLflow |

## Train

```powershell
python main.py train train.yaml
```

Example Ultralytics detection config:

```yaml
framework: ultralytics
task: detection

model: base_models/yolo/yolo11n.pt
dataset: test/data.yaml

epochs: 50
imgsz: 640
batch: 4

project_dir: outputs/project_vision_01
project_name: exp01

yolo_args:
  lr0: 0.0001
  momentum: 0.937
  hsv_h: 0.015
  hsv_s: 0.7
```

## Visualize

```powershell
python main.py visualize visualize.yaml
```

Example detection config:

```yaml
framework: ultralytics
task: detection

model_path: outputs/project_vision_01/exp01/weights/best.pt
src_path: test/images/val

nc: 1
img_sz: 640
conf_threshold: 0.5
nms_threshold: 0.3
```

Example custom multi-head classification config:

```yaml
framework: custom_multihead
task: classification

model_path: outputs/multihead/exp01/best.pth
src_path: D:/dataset/valid/images

device: 0
shuffle: false

threshold: 0.6
thresholds:
  kind: 0.6
  color: 0.5

font_scale: 0.7
font_thickness: 2
line_height: 30
text_origin: [10, 30]
```

Notes:
- the command is `visualize`
- for custom multi-head classification, the predictor loads the model once during initialization
- `predict()` receives an image directly

## Evaluate

```powershell
python main.py evaluate eval.yaml
```

Example config:

```yaml
framework: ultralytics
task: detection

image_dir: test/images/val
label_dir: test/labels/val
class_file: test/class_names.txt

model_path: outputs/project_vision_01/exp01/weights/best.pt
nc: 1

img_sz: 640
conf_threshold: 0.001
nms_threshold: 0.6
dst_dir: outputs/project_vision_01/exp01/eval
result_name: evaluation_result
```

## Export

```powershell
python main.py export export.yaml
```

Example config:

```yaml
framework: ultralytics
model_path: outputs/project_vision_01/exp01/weights/best.pt

img_sz: 640
batch: [1, 4, 8, 16]
opset: 12

export_dir: outputs/project_vision_01/exp01/weights
```

## MLflow Logging

VisionStudio supports experiment tracking and model artifact management through MLflow.

```powershell
python main.py log_eval logging.yaml
python main.py log_release logging.yaml
python main.py upload_model logging.yaml
python main.py log_model logging.yaml
```

## Custom Multi-Head Classification

VisionStudio now supports a custom multi-head classification workflow through the `custom_multihead` framework key.

Training config example:

```yaml
framework: custom_multihead
task: classification

dataset: D:/dataset/data.yaml

epochs: 30
imgsz: 224
batch: 16
val_batch: 16
workers: 2
device: 0

backbone_name: convnext_tiny
pretrained: true
lr: 0.0001
weight_decay: 0.0001

project_dir: outputs/multihead
project_name: exp01

heads:
  - name: kind
    num_classes: 14
    class_names:
      - passenger_car
      - truck
      - taxi
      - bus
      - suv
      - van
      - fire_engine
      - police_car
      - ambulance
      - motorcycle
      - bicycle
      - kick_scooter
      - wheelchair
      - forklift

  - name: color
    num_classes: 10
    class_names:
      - yellow
      - orange
      - green
      - gray
      - red
      - blue
      - white
      - golden
      - brown
      - black

thresholds:
  kind: 0.6
  color: 0.5
```

Dataset yaml example:

```yaml
train: D:/dataset/train/images
val: D:/dataset/valid/images
```

Label format:

```text
train/images/image1.jpg
train/labels/image1.txt
```

`image1.txt`

```text
0 4
```

Each token is a class index and the token order must match the `heads` order.

Outputs:
- `best.pth`
- `last.pth`
- `args.yaml`

Detailed package usage is documented in [custom_trainer/multihead_classification/README.md](/C:/Users/User/Desktop/workspace/MLOps/VisionStudio/custom_trainer/multihead_classification/README.md).

## Extensibility

VisionStudio is designed so that framework-specific logic stays behind thin adapters.

This makes it easier to add:
- new trainers
- new predictors
- new visualization flows
- custom model packages under `custom_trainer/`

## Design Philosophy

Different frameworks may have different APIs, but their workflows should be managed consistently.

## Conclusion

VisionStudio aims to be an integrated experiment management platform for Vision AI model development, while remaining open to custom extensions such as multi-head classification.
