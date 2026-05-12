# -*- coding:utf-8 -*-

import os
import cv2
import random


class DetectionVisualizer:
    def __init__(self, cfg):
        self.cat_cnt = cfg.get("nc", 1)

        self.colors = {
            i: (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
            for i in range(self.cat_cnt)
        }

        self.save_dir = cfg.get("save_dir", "custom_trainer/test/asan_test")
        self.count = 0

    def draw(self, img, boxes, scale=0.35, t=2):
        canvas = img.copy()
        for box in boxes:
            conf = box[0]
            cls = int(box[1])
            cords = list(map(int, box[2:]))

            label = "%d: %.2f" % (cls, conf)

            canvas = cv2.rectangle(canvas, (cords[0], cords[1]), (cords[2], cords[3]), self.colors[cls], t)
            canvas = cv2.putText(canvas, label, (cords[0], cords[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, scale, self.colors[cls], 1, cv2.LINE_AA)

        return canvas

    def vehicle_cropNsave(self, img, boxes):
        for i, box in enumerate(boxes):
            conf = box[0]
            cls = int(box[1])
            cords = list(map(int, box[2:]))

            if cls == 1:
                x1, y1, x2, y2 = cords
                crop_img = img[max(0, y1):y2, max(0, x1):x2]

                if crop_img.size == 0:
                    continue

                save_path = os.path.join(self.save_dir, "crop_{}_{}.jpg".format(cls, self.count))
                cv2.imwrite(save_path, crop_img)
                self.count += 1
                print("Saved: {}".format(save_path))

        return self.count


class MultiHeadClassificationVisualizer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.scale = cfg.get("font_scale", 0.7)
        self.thickness = cfg.get("font_thickness", 2)
        self.line_height = cfg.get("line_height", 30)
        self.origin = tuple(cfg.get("text_origin", (10, 30)))

    def draw(self, img, results):
        canvas = img.copy()
        x, y = self.origin

        for head_name, result in results.items():
            text = "{}: {} ({:.2f})".format(head_name, result["name"], result["confidence"])
            color = (0, 165, 255) if result.get("is_unknown") else (0, 255, 0)
            cv2.putText(canvas, text, (x, y), self.font, self.scale, color, self.thickness, cv2.LINE_AA)
            y += self.line_height

        return canvas


class SegmentationVisualizer:
    pass


class ClassificationVisualizer:
    pass


class PoseVisualizer:
    pass


class ObbVisualizer:
    pass
