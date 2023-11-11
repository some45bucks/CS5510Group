#!/bin/bash
yolo val model=yolov8s-cls.pt data=imagenet batch=1 imgsz=224 plots=true save_json=true