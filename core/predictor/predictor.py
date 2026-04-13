# -*- coding:utf-8 -*-

import numpy as np

from ultralytics import YOLO

from custom_trainer.multihead_classification.predictor import MultiHeadClassificationPredictor as CustomMultiHeadClassificationPredictor


class UltralyticsDetectionPredictor:
    def __init__(self, cfg):
        model_path = cfg["model_path"]
        self.model = YOLO(model_path)

        self.imgsz = cfg.get("img_sz", 640)
        self.conf_thd = cfg.get("conf_threshold", 0.5)
        self.iou_thd = cfg.get("nms_threshold", 0.3)

    def predict(self, image):
        result = self.model(image, verbose=False, imgsz=self.imgsz, conf=self.conf_thd, iou=self.iou_thd)[0]

        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        output = []

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            conf = scores[i]
            cls = classes[i]

            output.append([conf, cls, x1, y1, x2, y2])

        return output


class RFDETRDetectionPredictor:
    def __init__(self, cfg):
        model_size = cfg["model_size"]
        model_path = cfg["model_path"]
        num_class = cfg["nc"]

        self.imgsz = cfg.get("img_sz", 640)
        self.conf_thd = cfg.get("conf_threshold", 0.5)
        self.iou_thd = cfg.get("nms_threshold", 0.3)

        self.model = self.build_model(model_size, model_path=model_path, num_class=num_class)

    def build_model(self, model_size, model_path=None, num_class=None):
        size = model_size.strip().lower()

        kwargs = {}

        if num_class is not None:
            kwargs["num_classes"] = num_class
        if model_path is not None:
            kwargs["pretrain_weights"] = model_path

        if size == "nano":
            from rfdetr import RFDETRNano
            return RFDETRNano(**kwargs)
        if size == "small":
            from rfdetr import RFDETRSmall
            return RFDETRSmall(**kwargs)
        if size == "medium":
            from rfdetr import RFDETRMedium
            return RFDETRMedium(**kwargs)
        if size == "large":
            from rfdetr import RFDETRLarge
            return RFDETRLarge(**kwargs)
        if size == "xlarge":
            from rfdetr import RFDETRXLarge
            return RFDETRXLarge(**kwargs)
        if size == "2xlarge":
            from rfdetr import RFDETR2XLarge
            return RFDETR2XLarge(**kwargs)
        if size == "base":
            from rfdetr import RFDETRBase
            return RFDETRBase(**kwargs)

        raise ValueError("Unknown model_size: %s" % model_size)

    def predict(self, image):
        result = self.model.predict(image, threshold=self.conf_thd)

        boxes = result.xyxy
        scores = result.confidence.reshape((-1, 1))
        classes = result.class_id.reshape((-1, 1))

        output = np.concatenate([scores, classes, boxes], axis=-1)

        return output


class MultiHeadClassificationPredictor(CustomMultiHeadClassificationPredictor):
    pass
