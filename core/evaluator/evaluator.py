# -*- coding:utf-8 -*-

import os, sys
sys.path.append('core/evaluator/cocoapi/PythonAPI')
import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

class DetectionEvaluator:
    def __init__(self, config):
        self.config = config

    def evaluate_coco(self, gt_json, pred_json):
        # Loads COCO Ground Truth and Predictions to calculate metrics.
        coco_gt = COCO(gt_json)
        coco_dt = coco_gt.loadRes(pred_json)

        cocoEval = COCOeval(coco_gt, coco_dt, "bbox")
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        precision = cocoEval._export_precision()
        precision = np.transpose(precision, (3,4,0,2,1)) #(TRKAM) -> #(AMTKR)

        recall = cocoEval._export_recall()
        recall = np.transpose(recall, (2,3,0,1)) #(TKAM) -> (AMTK)

        return precision, recall