# -*- coding:utf-8 -*-

import cv2
import random
import os

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
    window_name = "VS Visualization"

    def _is_stream_source(path):
        path_l = path.lower()
        return path_l.startswith("rtsp://") or path_l.startswith("rtmp://") or path_l.startswith("http://") or path_l.startswith("https://")

    def _is_video_file(path):
        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".mpg", ".mpeg", ".m4v", ".ts"}
        return os.path.isfile(path) and os.path.splitext(path)[1].lower() in video_exts

    def _show_frame(image, wait_ms):
        results = predictor.predict(image)
        vis = visualizer.draw(image, results)
        cv2.imshow(window_name, vis)
        return cv2.waitKey(wait_ms) & 0xFF

    if os.path.isdir(src_path):
        files = get_files(src_path)
        if shuffle:
            random.shuffle(files)

        for img_path in files:
            image = imread_unicode(img_path)
            if image is None: continue

            if _show_frame(image, 0) == ord("q"): break

    elif _is_video_file(src_path) or _is_stream_source(src_path):
        cap = cv2.VideoCapture(src_path)
        if not cap.isOpened():
            raise Exception("[ERROR]: Failed to open source: {}".format(src_path))

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None: break

                if _show_frame(frame, 1) == ord("q"): break
        finally:
            cap.release()
    else:
        raise Exception("[ERROR]: Unsupported src_path: {}".format(src_path))

    cv2.destroyAllWindows()
