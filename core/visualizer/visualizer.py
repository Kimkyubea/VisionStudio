# -*- coding:utf-8 -*-

import cv2
import random

class DetectionVisualizer:
    def __init__(self, cfg):
        self.cat_cnt = cfg.get('nc', 1)

        self.colors = {
            i: (random.randint(50,255), random.randint(50,255), random.randint(50,255))
            for i in range(self.cat_cnt)
        }

    def draw(self, img, boxes, scale=0.35, t=2):
        for box in boxes:
            conf  = box[0]
            cls   = box[1]
            cords = list(map(int, box[2:]))

            label = "%d: %.2f" % (cls, conf)

            img = cv2.rectangle(img, (cords[0], cords[1]), (cords[2], cords[3]), self.colors[cls], t)
            img = cv2.putText(img, label, (cords[0], cords[1]-5), cv2.FONT_HERSHEY_SIMPLEX, scale, self.colors[cls], 1, cv2.LINE_AA)

        return img

class SegmentationVisualizer: pass
class ClassificationVisualizer: pass
class PoseVisualizer: pass
class ObbVisualizer: pass
