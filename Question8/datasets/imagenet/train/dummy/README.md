This file and directory are here to just make YOLO choke on the training data folder so that it will
fall back on the validation data folder. This is a hack to get around the fact that
YOLO wants the training dataset to be present to determine classes, but the dataset is huge.