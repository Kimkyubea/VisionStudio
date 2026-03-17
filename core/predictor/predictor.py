# -*- coding:utf-8 -*-

from ultralytics import YOLO

class YOLOPredictor:
    def __init__(self, cfg):
        model_path = cfg['model_path']
        self.model = YOLO(model_path)

        self.imgsz = cfg.get('img_sz', 640)
        self.conf_thd = cfg.get('conf_threshold', 0.5)
        self.iou_thd = cfg.get('nms_threshold', 0.3)

    def predict(self, image):
        result = self.model(image, verbose=False, imgsz=self.imgsz, conf=self.conf_thd, iou=self.iou_thd)[0]

        boxes   = result.boxes.xyxy.cpu().numpy()
        scores  = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        output = []

        for i in range(len(boxes)):
            x1,y1,x2,y2 = boxes[i]
            conf = scores[i]
            cls = classes[i]

            output.append([conf, cls, x1, y1, x2, y2])

        return output

class RFDETRPredictor: pass