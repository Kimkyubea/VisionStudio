# Multihead Classification

`multihead_classification` is a custom multi-head classification package for VisionStudio.

It supports:
- config-driven training
- per-image multi-head labels stored in `.txt`
- validation during training
- checkpoint-based inference
- VisionStudio `train` / `visualize` integration through thin adapters

## Package Layout

- `trainer.py`: training loop and validation
- `dataset.py`: dataset parsing and dataloader creation
- `model.py`: model definition, head metadata, checkpoint restore
- `utils.py`: yaml, path, device, transform, dataset path helpers
- `predictor.py`: inference helpers and predictor class

## Label Format

Each image must have a label file with the same stem.

Example:

```text
train/images/image1.jpg
train/labels/image1.txt
```

`image1.txt`

```text
0 4
```

Each token is a class index and the order must match the order of `heads` in the config.

Example:

```yaml
heads:
  - name: kind
    num_classes: 14
  - name: color
    num_classes: 10
```

In this case:
- first token -> `kind`
- second token -> `color`

## Dataset YAML

Training follows a dataset yaml format similar to Ultralytics.

Example `data.yaml`:

```yaml
train: D:/dataset/train/images
val: D:/dataset/valid/images
```

Label directories are inferred automatically by replacing `images` with `labels`.

So the trainer will look for:

```text
D:/dataset/train/labels
D:/dataset/valid/labels
```

## Training Config Example

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

backbone:
  name: convnext_tiny
  pretrained: true
  # mean/std are read from timm pretrained_cfg when omitted.
  # mean: [0.485, 0.456, 0.406]
  # std: [0.229, 0.224, 0.225]
lr: 0.0001
weight_decay: 0.0001
warmup_epochs: 3.0
min_lr_ratio: 0.05
use_ema: true
ema_decay: 0.9998
ema_tau: 2000
label_smoothing: 0.0
auto_class_weight: false
class_weight_strategy: inverse

project_dir: outputs/multihead
project_name: exp01

heads:
  - name: kind
    num_classes: 14
    loss_weight: 1.0
    label_smoothing: 0.05
    auto_class_weight: true
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
    loss_weight: 0.7
    label_smoothing: 0.1
    class_weights: [1.0, 1.0, 1.1, 1.0, 1.0, 1.0, 0.9, 1.2, 1.1, 1.0]
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

## Training

Standalone:

```powershell
python classification_multi_head.py --config D:\path\to\train_config.yaml
```

VisionStudio:

```powershell
python main.py train D:\path\to\train_config.yaml
```

Outputs:
- `best.pth`
- `last.pth`
- `args.yaml`

## Inference

The predictor loads the model once in `__init__()` and receives an image in `predict()`.

Supported input types:
- image path
- `PIL.Image`
- `numpy.ndarray` image

The returned result is a dictionary keyed by head name.

Example shape:

```python
{
    "kind": {
        "index": 3,
        "name": "bus",
        "raw_name": "bus",
        "confidence": 0.87,
        "threshold": 0.6,
        "is_unknown": False,
        "scores": [...],
    },
    "color": {
        "index": 4,
        "name": "red",
        "raw_name": "red",
        "confidence": 0.64,
        "threshold": 0.5,
        "is_unknown": False,
        "scores": [...],
    },
}
```

## Visualization Config Example

```yaml
framework: custom_multihead
task: classification

model_path: D:/path/to/best.pth
src_path: D:/path/to/images

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

VisionStudio visualization:

```powershell
python main.py visualize D:\path\to\visualize_config.yaml
```

Note:
- the command is `visualize`, not `visualization`
- threshold keys must match the configured head names

## Notes

- Avoid using reserved module names such as `type` for head names. Use names like `kind` or `vehicle_type`.
- If `val` is missing in `data.yaml`, validation is skipped.
- If a prediction confidence is below threshold, that head is returned as `unknown`.
- Training supports optional warmup, cosine LR decay, and EMA for more stable optimization.
- Label smoothing can be set globally or per head, and head-specific values override the global setting.
- Class imbalance can be handled per head with either automatic class weights from the train split or manually provided `class_weights`.
- Image preprocessing applies resize, tensor conversion, and normalization. Mean/std are read from the selected timm backbone when available, with ImageNet defaults as fallback.
