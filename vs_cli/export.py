# -*- coding:utf-8 -*-

def create_exporter(cfg):
    frw = cfg['framework']
    if frw == "ultralytics":
        from core.exporter.exporter import UltralyticsExportor
        exporter = UltralyticsExportor(cfg)
        print('[INFO]: Create model exporter ULTRALYTICS')

    elif frw == "rfdetr":
        from core.exporter.exporter import RFDETRExporter
        exporter = RFDETRExporter(cfg)
        print('[INFO]: Create model exporter RFDETR')

    return exporter

def run_export(cfg):
    print("[INFO]: Start exporting ...")

    exporter = create_exporter(cfg)
    exporter.export()

    print("[INFO]: End exporting")
