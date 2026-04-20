# -*- coding:utf-8 -*-

import json
import os
import time
import hashlib

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

VALID_IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}
VALID_LABEL_EXTENSIONS = {".txt"}

def _normalize_path(path):
    return os.path.normcase(os.path.normpath(path))

def create_single_symlink(paths):
    src, dst = paths
    try:
        if os.path.islink(dst):
            linked_src = os.readlink(dst)
            if not os.path.isabs(linked_src):
                linked_src = os.path.join(os.path.dirname(dst), linked_src)

            if _normalize_path(linked_src) == _normalize_path(src):
                return "skipped"

            os.remove(dst)

        elif os.path.exists(dst):
            os.remove(dst)

        os.symlink(src, dst)
        return "created"

    except Exception as e:
        print(f"Error linking {src} to {dst}: {e}")
        return "failed"


def fast_bulk_symlink(path_list, max_workers=20):
    start_time = time.time()
    total_count = len(path_list)

    print(f"[INFO]: Link start - total files: {total_count}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(create_single_symlink, path_list))

    created_count = sum(1 for result in results if result == "created")
    skipped_count = sum(1 for result in results if result == "skipped")
    failed_count = sum(1 for result in results if result == "failed")
    end_time = time.time()

    print("-" * 30)
    print(f"[INFO]: Elapsed time: {end_time - start_time:.2f}s")
    print(f"[INFO]: Created: {created_count} / Skipped: {skipped_count} / Failed: {failed_count}")
    print("-" * 30)

def _build_link_name(src_path, dataset_dir):
    base_name = os.path.basename(src_path)
    stem, ext = os.path.splitext(base_name)
    dir_hash = hashlib.sha1(dataset_dir.encode("utf-8")).hexdigest()[:10]
    return "{}__{}{}".format(stem, dir_hash, ext)


def _is_valid_extension(file_name, valid_extensions):
    return os.path.splitext(file_name)[1].lower() in valid_extensions

def build_dataset_tasks(cfg: dict, data_cfg: dict):
    work_dir = os.path.join(cfg["project_dir"], cfg["project_name"], "dataset")
    dataset_tasks = {
        "train_images": [],
        "train_labels": [],
        "valid_images": [],
        "valid_labels": [],
    }

    if isinstance(data_cfg['train'], str): src_train_datasets = [data_cfg['train']]
    elif isinstance(data_cfg['train'], list): src_train_datasets = data_cfg['train']
    else: raise Exception('[Error]: Train dataset path in dataset yaml file is not expected data type')

    for train_dir in src_train_datasets:
        label_dir = train_dir.replace("images", "labels")
        with os.scandir(train_dir) as train_entries:
            dataset_tasks["train_images"].extend([
                (
                    entry.path,
                    os.path.join(work_dir, "train", "images", _build_link_name(entry.path, train_dir))
                )
                for entry in train_entries
                if entry.is_file() and _is_valid_extension(entry.name, VALID_IMAGE_EXTENSIONS)
            ])
        with os.scandir(label_dir) as label_entries:
            dataset_tasks["train_labels"].extend([
                (
                    entry.path,
                    os.path.join(work_dir, "train", "labels", _build_link_name(entry.path, train_dir))
                )
                for entry in label_entries
                if entry.is_file() and _is_valid_extension(entry.name, VALID_LABEL_EXTENSIONS)
            ])

    if isinstance(data_cfg['val'], str): src_valid_datasets = [data_cfg['val']]
    elif isinstance(data_cfg['val'], list): src_valid_datasets = data_cfg['val']
    else: raise Exception('[Error]: Valid dataset path in dataset yaml file is not expected data type')

    for valid_dir in src_valid_datasets:
        label_dir = valid_dir.replace("images", "labels")
        with os.scandir(valid_dir) as valid_entries:
            dataset_tasks["valid_images"].extend([
                (
                    entry.path,
                    os.path.join(work_dir, "valid", "images", _build_link_name(entry.path, valid_dir))
                )
                for entry in valid_entries
                if entry.is_file() and _is_valid_extension(entry.name, VALID_IMAGE_EXTENSIONS)
            ])
        with os.scandir(label_dir) as label_entries:
            dataset_tasks["valid_labels"].extend([
                (
                    entry.path,
                    os.path.join(work_dir, "valid", "labels", _build_link_name(entry.path, valid_dir))
                )
                for entry in label_entries
                if entry.is_file() and _is_valid_extension(entry.name, VALID_LABEL_EXTENSIONS)
            ])

    return dataset_tasks

def link_ds2work(dataset_tasks: list):
    if dataset_tasks is None or len(dataset_tasks) == 0:
        return

    dst_dir = os.path.dirname(dataset_tasks[0][1])
    os.makedirs(dst_dir, exist_ok=True)
    print("[INFO]: Work Directory has been created {}".format(dst_dir))
    fast_bulk_symlink(dataset_tasks, max_workers=32)

def link_dataset_tasks(dataset_tasks: dict):
    for split_name in ["train_images", "train_labels", "valid_images", "valid_labels"]:
        link_ds2work(dataset_tasks.get(split_name, []))

def write_link_manifest(cfg: dict, dataset_tasks: dict):
    prj_dir = cfg["project_dir"]
    prj_name = cfg["project_name"]
    work_dir = os.path.join(prj_dir, prj_name)
    manifest_path = os.path.join(work_dir, "dataset_link_manifest.json")

    os.makedirs(work_dir, exist_ok=True)

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "framework": cfg.get("framework", ""),
        "project_dir": prj_dir,
        "project_name": prj_name,
        "dataset_config": cfg.get("dataset", ""),
        "counts": {},
        "links": {},
    }

    for split_name, tasks in dataset_tasks.items():
        manifest["counts"][split_name] = len(tasks)
        manifest["links"][split_name] = [
            {
                "src": src,
                "dst": dst,
            }
            for src, dst in tasks
        ]

    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2, ensure_ascii=False)

    print("[INFO]: Dataset link manifest saved to {}".format(manifest_path))

def delete_link(link_path):
    try:
        if os.path.islink(link_path):
            os.remove(link_path)
    except Exception as e:
        print(f"Error deleting {link_path}: {e}")

def bulk_unlink(target_dir, max_workers=40):
    link_paths = []
    for root, _, files in os.walk(target_dir):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            if os.path.islink(file_path):
                link_paths.append(file_path)

    print(f"[INFO]: Unlink start - total links: {len(link_paths)}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(delete_link, link_paths))

    print("[INFO]: Unlink complete")
