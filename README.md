# Parking Sign Detection using YOLOv4

<!-- ![Demo Image](demo_detection.png) -->
*Example of the model detecting EV charging stations, accessible parking signs, and electric vehicles*

## Overview

This project implements a deep learning-based object detection system for parking infrastructure using YOLOv4. The model is trained to detect three key classes in parking environments:
- **EV**: Electric vehicles
- **Charger**: EV charging stations  
- **Accessible**: Accessible parking signs

The project was developed as part of a MathWorks Deep Learning for Object Detection course, demonstrating transfer learning techniques and comprehensive model evaluation.

## Features

- **Multi-class Detection**: Simultaneously detects electric vehicles, charging stations, and accessible parking signs
- **Transfer Learning**: Built upon pre-trained YOLOv4 "tiny-yolov4-coco" backbone for efficient training
- **Comprehensive Evaluation**: Includes precision-recall curves, confusion matrices, and mAP metrics
- **Anchor Box Optimization**: Uses K-means clustering to determine optimal anchor boxes
- **Threshold Optimization**: Automated detection threshold tuning for best performance
- **Data Analysis Tools**: Outlier detection and bounding box statistics analysis

## Model Performance

| Class | Objects | AP@0.5 | AP@0.25 |
|-------|---------|--------|---------|
| EV | 223 | 0.5403 | 0.6064 |
| Accessible | 167 | 0.3464 | 0.4853 |
| Charger | 106 | 0.1359 | 0.1702 |

**Overall mAP@0.5**: 0.3408  
**Overall mAP@0.25**: 0.4206

## Dataset

- **Total Images**: 400
- **Total Objects**: 496
- **Training/Validation Split**: 80/20
- **Input Size**: 320×480×3 pixels

### Class Distribution
- Electric Vehicles: 223 instances across 201 images
- Accessible Signs: 167 instances across 144 images  
- Chargers: 106 instances across 106 images

## Architecture

The model uses YOLOv4 architecture with:
- **Backbone**: Tiny-YOLOv4-COCO (pre-trained)
- **Anchor Boxes**: 12 optimized anchor boxes (6 per detection scale)
- **Input Resolution**: 320×480 pixels
- **Detection Scales**: 2 scales for multi-scale detection

## Files Structure

```
├── training_model.mlx              # Model training script
├── evaluating_model.mlx            # Model evaluation and testing
├── analyzing_data.mlx              # Data analysis and visualization
├── parkingTrainGTFINISHED.mat      # Ground truth annotations
├── trained_yolov4_detector.mat     # Trained model weights
└── README.md                       # This file
```

## Training Details

### Training Parameters
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Batch Size**: 64
- **Epochs**: 60
- **L2 Regularization**: 0.001
- **Validation Frequency**: Every 5 epochs

### Training Progress
The model achieved convergence with final losses:
- **Training Loss**: 35.441
- **Validation Loss**: 35.958

<!-- ![Training Progress](training_progress.png)
*Training and validation loss curves over 20 epochs* -->

<!-- ## Evaluation Metrics

### Precision-Recall Curves
![Precision-Recall](precision_recall_curves.png)
*Precision-recall curves for each class at different IoU thresholds*

### Confusion Matrix
![Confusion Matrix](confusion_matrix.png)
*Confusion matrix showing classification performance*

### Detection Threshold Analysis
![Threshold Analysis](threshold_analysis.png)
*mAP vs Detection Threshold curve for optimal threshold selection*

## Data Analysis

### Bounding Box Statistics
![Area vs Aspect Ratio](area_aspect_ratio.png)
*Scatter plot showing area vs aspect ratio distribution for each class*

### Box Plots
![Aspect Ratio Boxplot](aspect_ratio_boxplot.png)
*Box plots showing aspect ratio distribution by class*

![Area Boxplot](area_boxplot.png)
*Box plots showing area distribution by class* -->

## Requirements

- MATLAB R2020b or later
- Computer Vision Toolbox
- Deep Learning Toolbox
- Image Processing Toolbox

## Usage

### Training a New Model
```matlab
% Load the training script
run('training_model.m')
```

### Evaluating the Model
```matlab
% Load the evaluation script  
run('evaluating_model.m')
```

### Analyzing Data
```matlab
% Load the data analysis script
run('analyzing_data.m')
```

### Running Inference
```matlab
% Load trained detector
load('trained_yolov4_detector.mat', 'detector');

% Detect objects in new image
img = imread('your_image.jpg');
[bboxes, scores, labels] = detect(detector, img, 'Threshold', 0.05);

% Visualize results
detectedImg = insertObjectAnnotation(img, 'rectangle', bboxes, labels);
imshow(detectedImg);
```

## Key Insights

1. **EV Detection**: Achieved the highest performance (AP@0.5: 0.5403), likely due to larger object size and distinct features
2. **Charger Detection**: Most challenging class (AP@0.5: 0.1359), possibly due to smaller size and visual similarity to other objects
3. **Accessible Signs**: Moderate performance (AP@0.5: 0.3464), benefiting from standardized design but suffering from varying scales

## Acknowledgments

This project was completed as part of the MathWorks Deep Learning for Object Detection course. The YOLOv4 implementation is based on the MATLAB Computer Vision Toolbox.
