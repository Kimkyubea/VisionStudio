# -*- coding:utf-8 -*-

import cv2
import random

from utils.common import get_files, imread_unicode


def _load_ultralytics_detection_predictor():
    from core.predictor.predictor import UltralyticsDetectionPredictor
    return UltralyticsDetectionPredictor


def _load_rfdetr_detection_predictor():
    from core.predictor.predictor import RFDETRDetectionPredictor
    return RFDETRDetectionPredictor


def _load_multihead_classification_predictor():
    from core.predictor.predictor import MultiHeadClassificationPredictor
    return MultiHeadClassificationPredictor


PREDICTOR_REGISTRY = {
    ("ultralytics", "detection"): _load_ultralytics_detection_predictor,
    ("rfdetr", "detection"): _load_rfdetr_detection_predictor,
    ("custom_multihead", "classification"): _load_multihead_classification_predictor,
}


def create_predictor(cfg):
    frw = cfg["framework"]
    task = cfg["task"]

    key = (frw, task)
    loader = PREDICTOR_REGISTRY.get(key)
    if loader is None:
        raise Exception("[ERROR]: Unsupported framework / task : {}".format(key))

    predictor_cls = loader()
    return predictor_cls(cfg)


def create_visualizer(cfg):
    task = cfg.get("task", "detection")

    if task == "detection":
        from core.visualizer.visualizer import DetectionVisualizer
        return DetectionVisualizer(cfg)

    if task == "classification":
        from core.visualizer.visualizer import MultiHeadClassificationVisualizer
        return MultiHeadClassificationVisualizer(cfg)

    raise Exception("[ERROR]: Unsupported visualization task : {}".format(task))


def run_visualize(cfg):
    visualizer = create_visualizer(cfg)
    predictor = create_predictor(cfg)

    src_path = cfg["src_path"]
    shuffle = cfg.get("shuffle", False)

    files = get_files(src_path)
    if shuffle:
        random.shuffle(files)

    for img_path in files:
        image = imread_unicode(img_path)
        if image is None:
            continue

        results = predictor.predict(image)
        vis = visualizer.draw(image, results)

        cv2.imshow("VS Visualization", vis)
        if cv2.waitKey(0) & 0xFF == ord("q"):
            break
