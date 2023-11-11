@echo off
python .\yolov5\classify\val.py --weights yolov5s-cls.pt --data .\datasets\imagenet\ --img 224 --batch-size 1