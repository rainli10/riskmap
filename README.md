# Risk Map Prediction for Autonomous Driving

Predicts a dense per-pixel **risk map** that guide the egovehicle which objects are needed to pay attention to.

## Input / Output
- **Input:** RGB-D image  
- **Output:** Single-channel risk map (higher = more dangerous)

## Data
- Cityscapes + KITTI  
- Dense depth generated using a foundation depth model

## Risk Label
Risk is computed per pixel using semantic danger weight and depth:
R(x,y) = w_c · g(D)

## Model
- **Main:** U-Net style encoder–decoder CNN  
- **Baseline:** Simple CNN without multi-scale features

## Goal
Learn risk-aware perception beyond binary obstacle detection.

## Course
APS360 – Applied Fundamentals of Deep Learning (UofT)
