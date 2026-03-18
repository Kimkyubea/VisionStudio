# -*- coding:utf-8 -*-

import os, sys
import cv2
import glob
import json 
import yaml

import numpy as np

from tqdm import tqdm
from datetime import datetime
from pathlib import Path

def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(path, dtype=np.uint8)
    if data is None or data.size == 0: return None
    img = cv2.imdecode(data, flags)
    
    return img

def get_files(folder_path, extensions=None):
    print('[INFO]: Retrieving file path from {} ... '.format(folder_path))

    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp']
    
    image_files = []
    for ext in extensions:
        pattern = os.path.join(folder_path, '**', f'*{ext}')
        image_files.extend(glob.glob(pattern, recursive=True))
    
    return sorted(image_files)

def create_timestamped_folder(base_dir="."):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    folder_path = Path(base_dir) / '{}_{}'.format(timestamp, 'evaluationResult')
    folder_path.mkdir(parents=True, exist_ok=True)
    print("[INFO]: Create output directory {}".format(folder_path.absolute()
    ))
    return str(folder_path)

def load_json(json_path):
    _d = None
    if not os.path.exists(json_path): return {}

    with open(json_path, 'r', encoding="utf-8") as jf:
        _d = json.load(jf)
        if _d is None: _d = {}

    return _d

def load_yaml(yaml_path):
    _d = None
    if not os.path.exists(yaml_path): return {}

    with open(yaml_path, 'r', encoding="utf-8") as yf:
        _d = yaml.load(yf, Loader=yaml.FullLoader)
        if _d is None: _d = {}
    return _d