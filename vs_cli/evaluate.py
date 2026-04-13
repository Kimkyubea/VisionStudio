# -*- coding:utf-8 -*-

import os, sys
sys.path.append('core/evaluator/cocoapi/PythonAPI')
import json
import cv2
import yaml
import numpy as np

from tqdm import tqdm

from utils.common import get_files, imread_unicode
from utils.eval_utils import gt_convert_yolo2coco, write_as_txt, write_as_json, _summary

def create_predictor(cfg):
    frw = cfg['framework']
    task = cfg.get('task', 'detection')

    if task != 'detection':
        raise Exception("[ERROR]: Unsupported evaluation task : {}".format(task))

    if frw == "ultralytics":
        from core.predictor.predictor import UltralyticsDetectionPredictor
        predictor = UltralyticsDetectionPredictor(cfg)
        print("[INFO]: Create vision predictor ULTRALYTICS")

    elif frw == "rfdetr":
        from core.predictor.predictor import RFDETRDetectionPredictor
        predictor = RFDETRDetectionPredictor(cfg)
        print("[INFO]: Create vision predictor RFDETR")

    else:
        raise Exception("[ERROR]: Unsupported framework : {}".format(frw))

    return predictor

def create_evaluator(cfg):
    task = cfg.get('task', 'detection')
    if task == 'detection':
        from core.evaluator.evaluator import DetectionEvaluator
        evaluator = DetectionEvaluator(cfg)

    return evaluator

def create_visualizer(cfg):
    task = cfg.get('task', 'detection')
    if task == 'detection':
        from core.visualizer.visualizer import DetectionVisualizer
        return DetectionVisualizer(cfg)

    return None

def _select_fixed_sample_indices(file_count, sample_count):
    if file_count <= 0 or sample_count <= 0:
        return set()

    sample_count = min(file_count, sample_count)
    if sample_count == 1:
        return {0}

    indices = set()
    last_index = file_count - 1
    for i in range(sample_count):
        idx = round(i * last_index / (sample_count - 1))
        indices.add(int(idx))

    return indices

def _imwrite_unicode(path, image):
    ext = os.path.splitext(path)[1] or ".jpg"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        return False

    encoded.tofile(path)
    return True

def run_evaluate(cfg):
    predictor = create_predictor(cfg)
    evaluator = create_evaluator(cfg)
    visualizer = create_visualizer(cfg)

    img_dir    = cfg.get('image_dir', "")
    lbl_dir    = cfg.get('label_dir', "")
    cls_file   = cfg.get('class_file', "")
    model_path = cfg.get('model_path', "")
    dst_dir    = cfg.get('dst_dir', os.path.dirname(model_path))
    result_name = cfg.get('result_name', "evaluation_result")
    sample_count = int(cfg.get('save_pred_vis_count', 10))
    sample_dir = os.path.join(dst_dir, 'eval_samples')

    if img_dir == "" or lbl_dir == "": raise Exception("[ERROR]: Image or label director path is BLANK")
    if cls_file == "": raise Exception("[ERROR]: Category file path is BLANK")
    if model_path == "": raise Exception("[ERROR]: Vision Model path is BLANK")

    _cfg_bak = os.path.join(dst_dir, 'eval_config.yaml')
    with open(_cfg_bak, 'w') as yf:
        yaml.dump(cfg, yf, indent=4, default_flow_style=False, sort_keys=False)

    with open(cls_file, 'r') as cf:
        lines = cf.read().splitlines()
    cats = [x for x in lines]

    out_gt = os.path.join(dst_dir, './gt_out.json')
    out_dt = os.path.join(dst_dir, './dt_out.json')
    dst_file = os.path.join(dst_dir, f'{result_name}.txt')
    dst_json = os.path.join(dst_dir, f'{result_name}.json')

    gt_convert_yolo2coco(img_dir, lbl_dir, cats, out_gt)

    files = get_files(img_dir)
    sample_indices = _select_fixed_sample_indices(len(files), sample_count)
    if sample_indices:
        os.makedirs(sample_dir, exist_ok=True)
        print('[INFO]: Save fixed evaluation samples to {}'.format(sample_dir))
        print('[INFO]: Fixed sample indices = {}'.format(sorted(sample_indices)))

    print('[INFO]: Start inference ... ')
    print('[INFO]: Model path = {}'.format(model_path))
    print('[INFO]: Eval DS    = {}'.format(img_dir))

    # TODO 검출에 종속적이므로 wrapper에서는 빼야됨 #
    result_list = []
    for img_id, img_path in tqdm(enumerate(files)):
        _img = imread_unicode(img_path)
        if _img is None:
            continue

        objs = predictor.predict(_img)

        if img_id in sample_indices and visualizer is not None:
            vis_img = visualizer.draw(_img, objs)
            sample_name = '{:04d}_{}'.format(img_id, os.path.basename(img_path))
            sample_path = os.path.join(sample_dir, sample_name)
            if _imwrite_unicode(sample_path, vis_img):
                print('[INFO]: Saved eval sample {}'.format(sample_path))

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
