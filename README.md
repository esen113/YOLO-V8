YOLOv8 Object Detection

Overview

This project uses YOLOv8 (You Only Look Once Version 8) for object detection. YOLOv8 is a real-time object detection model that can identify objects in images and videos with high accuracy.

Features

Object detection with bounding boxes and class labels.
Train and fine-tune the model on a custom dataset.
Evaluate the model using metrics like mAP (mean Average Precision).
Visualize detection results in images and videos.
Requirements
Install the required libraries using the following command:

bash

pip install ultralytics opencv-python numpy matplotlib
Quick Start
1. Load the Pretrained Model
python

from ultralytics import YOLO

# Load the pretrained model
model = YOLO('yolov8n.pt')
2. Prepare Your Dataset
Organize the dataset in YOLO format and configure data.yaml:

yaml

path: ./data
train: images/train
val: images/val

names:
  0: person
  1: car
  2: dog
3. Train the Model
python

model.train(data='data.yaml', epochs=50, batch=16, imgsz=640)
4. Run Inference
Run object detection on images or videos:

python

results = model.predict(source='path/to/image_or_video', save=True, conf=0.5)

Results

View detection results saved in the runs/ folder.

Example metrics:

mAP: 85.3%

Precision: 90.5%

Recall: 88.7%

License

This project is licensed under the MIT License.

