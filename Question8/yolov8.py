from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov8n-cls.pt", task='classify')
    model.val(data='imagenet', split='val', imgsz=224)
