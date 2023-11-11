For this question, python 3.11 was used. Prevous versions of python may work, but only 3.11 has been tested on these questions.

The dataset used by this question is a subset of the ImageNet ILSVRC 2012 dataset. See the README in the `datasets/imagenet` directory for information on how to download the dataset. These files must be downloaded and placed in the `datasets/imagenet` directory before running the code in this question and the dependencies in `requirements.txt` must be installed. For best results, it is recommended to manually install pytorch first.

If you don't want to worry about extracting the imagenet archive files, the alexnet.py file will extract the archives as part of the data loading process. After this step has completed, yolo will then be able to run the imagenet dataset.

## Running Each Question

### Part a:

Run `python alexnet.py` from the `Question8` directory after downloading the needed imagenet files and after installing the dependencies listed in the repository's `requirements.txt`.

### Part b and c

The yolo models can be run with the appropratie script:
- For yolov8: `./yolov8.sh` or `.\yolov8.bat` on windows should be run from the `Question8` directory

Note: If YOLO complains about not being able to find the dataset, try running the alexnet script first. This script also handles unpacking the tarballs for the imagenet dataset.

Laptop Specs:
- CPU: Intel Core i7-8650U @ 2.11 GHz base clock
- GPU: NVIDIA GeForce GTX 1050

YOLOv8 Run Results for Laptop:

```
yolo val model=yolov8s-cls.pt data=imagenet batch=1 imgsz=224 plots=true save_json=true 
Ultralytics YOLOv8.0.208  Python-3.11.5 torch-2.1.0 CUDA:0 (GeForce GTX 1050, 2048MiB)
YOLOv8s-cls summary (fused): 73 layers, 6356200 parameters, 0 gradients, 13.5 GFLOPs
train: C:\Users\kytec\source\repos\USU\CS5510GroupOopsAllGrads\Question8\datasets\imagenet\train... found 1 images in 1 classes

val: C:\Users\kytec\source\repos\USU\CS5510GroupOopsAllGrads\Question8\datasets\imagenet\val... found 50000 images in 1000 classes: ERROR  requires 1 classes, not 1000
test: None...
val: Scanning C:\Users\kytec\source\repos\USU\CS5510GroupOopsAllGrads\Question8\datasets\imagenet\val... 50000 images, 0 corru
               classes   top1_acc   top5_acc: 100%|██████████| 50000/50000 [20:47<00:00, 40.09it/s]  
                   all      0.677      0.881
Speed: 0.5ms preprocess, 10.3ms inference, 0.0ms loss, 0.0ms postprocess per image
Results saved to runs\classify\val6
 Learn more at https://docs.ultralytics.com/modes/val
```

YOLOv8 Run Results for Pi
