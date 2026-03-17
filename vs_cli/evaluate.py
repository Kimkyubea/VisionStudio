# -*- coding:utf-8 -*-

import os, sys
sys.path.append('core/evaluator/cocoapi/PythonAPI')
import json
import numpy as np

from tqdm import tqdm

from utils.common import get_files, imread_unicode
from utils.eval_utils import gt_convert_yolo2coco, write_as_txt, write_as_json, _summary

def create_prdictor(cfg):
    frw = cfg['framework']
    if frw == "ultralytics":
        from core.predictor.predictor import YOLOPredictor
        predictor = YOLOPredictor(cfg)
        print("[INFO]: Create vision predictor ULTRALYTICS")

    elif frw == "rfdetr": 
        from core.predictor.predictor import RFDETRPredictor
        trainer = RFDETRPredictor(cfg)
        print("[INFO]: Create vision predictor RFDETR")

    return predictor

def create_evaluator(cfg):
    task = cfg.get('task', 'detection')
    if task == 'detection':
        from core.evaluator.evaluator import DetectionEvaluator
        evaluator = DetectionEvaluator(cfg)

    return evaluator

def run_evaluate(cfg):
    predictor = create_prdictor(cfg)
    evaluator = create_evaluator(cfg)

    img_dir  = cfg.get('image_dir', "")
    lbl_dir  = cfg.get('label_dir', "")
    cls_file = cfg.get('class_file', "")
    dst_dir  = cfg.get('dst_dir', "./") # Temp
    model_path = cfg.get('model_path', "")

    if img_dir == "" or lbl_dir == "": raise Exception("[ERROR]: Image or label director path is BLANK")
    if cls_file == "": raise Exception("[ERROR]: Category file path is BLANK")
    if model_path == "": raise Exception("[ERROR]: Vision Model path is BLANK")

    with open(cls_file, 'r') as cf:
        lines = cf.read().splitlines()
    cats = [x for x in lines]

    out_gt = os.path.join(dst_dir, './gt_out.json')
    out_dt = os.path.join(dst_dir, './dt_out.json')
    dst_file = os.path.join(dst_dir, 'e_result.txt')
    dst_json = os.path.join(dst_dir, 'e_result.json')

    gt_convert_yolo2coco(img_dir, lbl_dir, cats, out_gt)

    files = get_files(img_dir)

    print('[INFO]: Start inference ... ')
    print('[INFO]: Model path = {}'.format(model_path))
    print('[INFO]: Eval DS    = {}'.format(img_dir))

    result_list = []
    for img_id, img_path in tqdm(enumerate(files)):
        _img = imread_unicode(img_path)
        objs = predictor.predict(_img)

        for obj in objs:
            conf = float(obj[0])
            cls  = int(obj[1])
            x1, y1, x2, y2 = list(map(float, obj[2:]))
            w = x2 - x1
            h = y2 - y1

            result_list.append({
                "image_id"   : img_id,
                "category_id": cls,
                "bbox"       : [x1, y1, w, h],
                "score"      : conf
            })

    with open(out_dt, 'w') as jf: json.dump(result_list, jf)
    print('[INFO]: DONE inference, Save COCO pred {}'.format(out_dt))

    precision, recall = evaluator.evaluate_coco(out_gt, out_dt)

    write_as_txt(dst_file, _summary(precision, recall, note='All'))
    write_as_json(dst_json, precision, recall)
    
    precision = np.transpose(precision, (3,0,1,2,4)) # Metrics transpose (AMTKR) -> (KAMTR)
    recall    = np.transpose(recall, (3,0,1,2))      # Metrics transpose (AMTK) -> (KAMT)
    for i, (p, r) in enumerate(zip(precision, recall)):
        write_as_txt(dst_file, _summary(p, r, note=cats[i]))
    
    print("[INFO]: Evaluation Complete. Results saved in : {}".format(dst_dir))