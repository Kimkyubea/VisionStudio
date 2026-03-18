# -*- coding:utf-8 -*-

def create_trainer(cfg):
    frw = cfg['framework']
    if frw == "ultralytics":
        from core.trainer.trainer import UltralyticsTrainer
        trainer = UltralyticsTrainer(cfg)
        print("[INFO]: Create vision trainer ULTRALYTICS")

    elif frw == "rfdetr": 
        from core.trainer.trainer import RFDETRTrainer
        trainer = RFDETRTrainer(cfg)
        print("[INFO]: Create vision trainer RFDETR")

    else: raise Exception("[ERROR]: Unsupported framework : {}".format(frw))

    return trainer

def run_train(cfg):
    print("[INFO]: Start training ...")

    trainer = create_trainer(cfg)
    trainer.train()

    print("[INFO]: End training")
    