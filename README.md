## Mini YOLO Object Detection (From Scratch)

This project implements a complete object detection pipeline from scratch using a custom YOLO-style CNN, without any pre-trained weights. The model is trained on a cleaned subset of the PASCAL VOC 2012 dataset and supports real-time object detection.

## Dataset

Source: PASCAL VOC 2012
Cleaned image–annotation pairs: 5,138
Train / Val split: 4,110 / 1,028
Object classes: 20 VOC classes

## Model

Custom lightweight YOLO-style single-stage detector
CNN backbone + fully connected detection head
Input size: 224 × 224
Outputs bounding box coordinates, confidence, and class

## Training

Optimizer: Adam
Learning rate: 1e-3
Epochs: 10
Loss: Mean Squared Error (MSE)
Trained fully from scratch
Results
mAP@0.5: ~0.28
Inference speed: ~25–30 FPS (CPU)
Model size: ~4.2 MB

## Features

Clean VOC preprocessing pipeline
Fast training and real-time inference
OpenCV-based live detection (video/webcam)

## Trade-offs

Faster and lightweight compared to Faster R-CNN
Slightly lower accuracy due to simplified loss and no anchor boxes

## MODEL TRAINING
<img width="1322" height="450" alt="Screenshot 2026-01-13 231207" src="https://github.com/user-attachments/assets/05591569-289f-4f62-ab17-db8e1d2f5db9" />


## RESULTS
![recognized_img1](https://github.com/user-attachments/assets/9eb39ad2-8cdc-47d9-aa37-56bdd1798f55)
![recognized_img2](https://github.com/user-attachments/assets/16181f64-68cc-49be-b1fe-58f3ee69dbdc)

