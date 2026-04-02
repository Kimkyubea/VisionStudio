# -*- coding:utf-8 -*-

def _load_ultralytics_detection_trainer():
    from core.trainer.trainer import UltralyticsDetectionTrainer
    return UltralyticsDetectionTrainer

def _load_ultralytics_classification_trainer():
    from core.trainer.trainer import UltralyticsClassificationTrainer
    return UltralyticsClassificationTrainer

def _load_rfdetr_detection_trainer():
    from core.trainer.trainer import RFDETRTrainer
    return RFDETRTrainer

TRAINER_REGISTRY = {
    ("ultralytics", "detection"): _load_ultralytics_detection_trainer,
    ("ultralytics", "classification"): _load_ultralytics_classification_trainer,
    ("rfdetr", "detection"): _load_rfdetr_detection_trainer,
}

def create_trainer(cfg):
    frw = cfg["framework"] 
    task = cfg["task"]

    key = (frw, task)
    loader = TRAINER_REGISTRY.get(key)
    if loader is None: raise Exception("[ERROR]: Unsupported framework / task : {}".format(key))

    trainer_cls = loader()
    return trainer_cls(cfg)
    

def run_train(cfg):
    print("[INFO]: Start training ...")

    trainer = create_trainer(cfg)
    trainer.train()

    print("[INFO]: End training")
    