
# -*- coding:utf-8 -*-

import os, sys
import json 
import numpy as np

from tqdm import tqdm

from utils.common import imread_unicode, get_files

def gt_convert_yolo2coco(img_dir, lbl_dir, class_names, out_json):
    coco = {
        "images"     : [],
        "annotations": [],
        "categories" : []
    }

    for idx, cat in enumerate(class_names):
        coco['categories'].append({
            "id": idx,
            "name": cat,
            "supercategory": "object"
        })

    image_files = get_files(img_dir)

    img_id = 0
    ant_id = 0

    print('[INFO]: Start GT format converting YOLO -> COCO')

    for image_file in tqdm(image_files):
        image_name = os.path.basename(image_file)
        label_name = '{}.txt'.format(os.path.splitext(image_name)[0])
        label_path = os.path.join(lbl_dir, label_name)

        # _img = cv2.imread(image_file)
        _img = imread_unicode(image_file)
        if _img is None: 
            print("[WARN]: Cannot open {}, skipped.".format(image_file))
            continue

        h,w,_ = _img.shape

        current_img_id = img_id
        coco["images"].append({
            "id": current_img_id,
            "file_name": image_name,
            "width": w,
            "height": h
        })
        img_id += 1

        if not os.path.exists(label_path): 
            print("[WARN]: Don't exists {}, skipped.".format(label_path))
            continue

        with open(label_path, 'r') as rf:
            lines = rf.read().splitlines()

        for line in lines:
            raw_data = line.split()
            if len(raw_data) != 5: continue

            cls, xc, yc, bw, bh = map(float, raw_data)
            cls = int(cls)

            x = (xc - bw/2) * w
            y = (yc - bh/2) * h
            box_w = bw * w
            box_h = bh * h
            box_a = box_w * box_h

            coco["annotations"].append({
                "id"         : ant_id,
                "image_id"   : current_img_id,
                "category_id": cls,
                "bbox"       : [x, y, box_w, box_h],
                "area"       : box_a,
                "iscrowd"    : 0
            })
            ant_id += 1

    with open(out_json, 'w') as jf:
        json.dump(coco, jf, indent='\t')

    print('[INFO]: Done GT format Converting {} to COCO GT {}'.format(lbl_dir, out_json))

def write_as_txt(dst_file, lines):
    # Append evaluation summary lines to a text file and prints to console.
    with open(dst_file, 'a') as f:
        for line in lines:
            f.write(line)
            print(line.strip())

def write_as_json(dst_json, precision, recall):
    """
        Extract key metrics and saves them to a JSON file.
        Note on indices: 
        precision[Area][MaxDet] -> Area 0 is 'All', MaxDet 2 is '100 objects'
    """
    mAP             = mean(precision[0][2])
    AP_50           = mean(precision[0][2][0])
    AP_60           = mean(precision[0][2][2])
    AP_70           = mean(precision[0][2][4])

    mAR             = mean(recall[0][2])
    AR_50           = mean(recall[0][2][0])
    AR_60           = mean(recall[0][2][2])
    AR_70           = mean(recall[0][2][4])

    mF1_scr         = (2*mAP*mAR) / (mAP+mAR)

    _d = {
        'mAP' : mAP,
        'AP50': AP_50,
        'AP70': AP_70,
        'mAR' : mAR,
        'AR50': AR_50,
        'AR70': AR_70,
        'mF1' : round(mF1_scr,5)
    }

    with open(dst_json, 'w') as jf:
        json.dump(_d, jf)
    
def mean(x):
    # Calculates mean of valid values (ignores -1 which signifies 'no data').
    x = x[x > -1]
    return round(np.mean(x),5) if x.size > 0. else np.nan

def make_line(k, v):
    # Formats key-value pairs for text reporting.
    return '{}: {:.5f}\n'.format(k, v)

def _summary(precision, recall, note=''):
    # Generates a readable summary of COCO metrics.
    mAP             = mean(precision[0][2])
    AP_50           = mean(precision[0][2][0])
    AP_60           = mean(precision[0][2][2])
    AP_70           = mean(precision[0][2][4])
    AP_80           = mean(precision[0][2][6])
    AP_90           = mean(precision[0][2][8])
    AP_S            = mean(precision[1][2])
    AP_M            = mean(precision[2][2])
    AP_L            = mean(precision[3][2])

    mAR             = mean(recall[0][2])
    AR_50           = mean(recall[0][2][0])
    AR_60           = mean(recall[0][2][2])
    AR_70           = mean(recall[0][2][4])
    AR_80           = mean(recall[0][2][6])
    AR_90           = mean(recall[0][2][8])
    AR_S            = mean(recall[1][2])
    AR_M            = mean(recall[2][2])
    AR_L            = mean(recall[3][2])

    if mAP * mAR: mF1_scr = (2*mAP*mAR) / (mAP+mAR)
    else: mF1_scr = 0.0

    return [
        '{} === Evaluation Result === \n'.format(note),
        '=== Precision === \n',
        make_line('mAP',   mAP),
        make_line('AP_50', AP_50),
        make_line('AP_60', AP_60),
        make_line('AP_70', AP_70),
        make_line('AP_80', AP_80),
        make_line('AP_90', AP_90),
        make_line('AP_S',  AP_S ),
        make_line('AP_M',  AP_M ),
        make_line('AP_L',  AP_L ),

        '=== Recall === \n',
        make_line('mAR', mAR),
        make_line('AR_50', AR_50),
        make_line('AR_60', AR_60),
        make_line('AR_70', AR_70),
        make_line('AR_80', AR_80),
        make_line('AR_90', AR_90),
        make_line('AR_S', AR_S),
        make_line('AR_M', AR_M),
        make_line('AR_L', AR_L),

        '=== F1-Score === \n',
        make_line('mF1-score', mF1_scr),
        '========================= \n'
    ]
