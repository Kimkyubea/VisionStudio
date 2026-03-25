# -*- coding:utf-8 -*-

import os, sys
import cv2

from utils.common import get_files, imread_unicode

def create_prdictor(cfg):
    frw = cfg['framework']
    if frw == "ultralytics":
        from core.predictor.predictor import YOLOPredictor
        predictor = YOLOPredictor(cfg)
        print("[INFO]: Create vision predictor ULTRALYTICS")

    elif frw == "rfdetr": 
        from core.predictor.predictor import RFDETRPredictor
        predictor = RFDETRPredictor(cfg)
        print("[INFO]: Create vision predictor RFDETR")

    return predictor

def create_visualizer(cfg):
    task = cfg.get('task', 'detection')
    if task == 'detection': 
        from core.visualizer.visualizer import DetectionVisualizer
        visualizer = DetectionVisualizer(cfg)

    return visualizer

def run_visualize(cfg):
    visualizer = create_visualizer(cfg)
    predictor = create_prdictor(cfg)

    src_path = cfg['src_path']

    files = get_files(src_path)
    for img_path in files:
        _img = imread_unicode(img_path)
        if _img is None: continue

        boxes = predictor.predict(_img)
        vis = visualizer.draw(_img, boxes)

        cv2.imshow('VS Visualization', vis)
        if cv2.waitKey(0) & 0xFF == ord('q'): break