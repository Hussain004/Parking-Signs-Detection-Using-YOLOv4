# Parking Sign Detection using YOLOv4

![Demo Image](img/demo_detection.png)
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

## Model Performance (Test Data Results)

| Class | Objects | AP@0.5 | AP@0.25 |
|-------|---------|--------|---------|
| EV | 58 | 0.5854 | 0.6523 |
| Charger | 29 | 0.0345 | 0.0690 |
| Accessible | 48 | 0.2348 | 0.3098 |

**Overall mAP@0.5**: 0.2849  
**Overall mAP@0.25**: 0.3437

## Dataset

- **Total Images**: 400
- **Training/Validation Split**: 80/20
- **Test Images**: 100
- **Test Objects**: 135
- **Input Size**: 320×480×3 pixels

### Test Data Distribution
- Electric Vehicles: 58 instances
- Accessible Signs: 48 instances  
- Chargers: 29 instances

## Architecture

The model uses YOLOv4 architecture with:
- **Backbone**: Tiny-YOLOv4-COCO (pre-trained)
- **Anchor Boxes**: 12 optimized anchor boxes (6 per detection scale)
- **Input Resolution**: 320×480 pixels
- **Detection Scales**: 2 scales for multi-scale detection

## Files Structure

```
├── model/
│   └── trained_yolov4_detector.mat     # Trained model weights (60 epochs)
├── labeled_data/
│   ├── parkingTrainGTFINISHED.mat      # Training ground truth annotations
│   └── parkingTestGT.mat               # Test ground truth annotations
├── training_model.mlx                  # Model training script
├── evaluating_model.mlx                # Model evaluation and testing
├── analyzing_data.mlx                  # Data analysis and visualization
└── README.md                           # This file
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
The model achieved convergence over 60 epochs of training on the training dataset.

<!-- ![Training Progress](training_progress.png)
*Training and validation loss curves over 60 epochs*

## Evaluation Metrics

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
run('training_model.mlx')
```

### Evaluating the Model
```matlab
% Load the evaluation script  
run('evaluating_model.mlx')
```

### Analyzing Data
```matlab
% Load the data analysis script
run('analyzing_data.mlx')
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

1. **Test Performance Analysis**: The model shows varying performance across different object classes when evaluated on unseen test data:
   - **EV Detection**: Achieved the best performance (AP@0.5: 0.5854), confirming that electric vehicles are the most reliably detected class
   - **Accessible Signs**: Moderate performance (AP@0.5: 0.2348), indicating room for improvement in detecting smaller signage
   - **Charger Detection**: Lowest performance (AP@0.5: 0.0345), suggesting this class presents the greatest detection challenge

2. **Class-Specific Challenges**: 
   - **EVs**: Larger objects with distinctive features lead to better detection rates
   - **Chargers**: Small size and potential occlusion make detection difficult
   - **Accessible Signs**: Variable appearance and size contribute to moderate performance

3. **IoU Threshold Impact**: Performance improves at lower IoU thresholds (0.25 vs 0.5), indicating the model produces reasonable localizations but may benefit from more precise bounding box regression

4. **Detection Threshold Optimization**: The evaluation includes systematic threshold testing to identify optimal detection confidence levels for deployment

5. **Real-world Applicability**: Test results provide realistic expectations for model performance on new, unseen parking infrastructure images

## Acknowledgments

This project was completed as part of the MathWorks Deep Learning for Object Detection course. The YOLOv4 implementation is based on the MATLAB Computer Vision Toolbox.