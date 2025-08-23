# Multi-Object Tracking - SORT Algorithm Implementation

A comprehensive object tracking system that implements the Simple Online and Realtime Tracking (SORT) algorithm with additional support for YOLO 11's built-in tracking capabilities. This project provides two distinct approaches for multi-object tracking in video sequences.

## 🎯 Project Overview

This project demonstrates two different approaches to object tracking:

1. **Custom SORT Algorithm Implementation**: A from-scratch implementation of the SORT algorithm with advanced data association techniques
2. **YOLO 11 Built-in Tracking**: Utilizing YOLO 11's native tracking capabilities for comparison

The system is designed to track multiple objects across video frames while maintaining consistent object identities, making it suitable for applications like surveillance, traffic monitoring, and sports analytics.

## 🎬 Demo Results

### Custom SORT Algorithm Implementation

![SORT Tracking Demo](output/sort_tracking.gif)

### YOLO 11 Built-in Tracking

![YOLO 11 Tracking Demo](output/yolo11_tracking.gif)

## 🔧 Features

- **Custom SORT Implementation**: Complete implementation of the SORT algorithm with Hungarian algorithm for data association
- **Advanced Cost Metrics**: Multiple similarity metrics including IoU, Sanchez-Matilla, and Yu metrics
- **YOLO 11 Integration**: Support for both custom tracking and YOLO 11's built-in tracking
- **Real-time Processing**: Optimized for real-time object tracking applications

The project provides two tracking approaches - switch between them by commenting/uncommenting lines in `main.py`:

- **YOLO 11 Built-in Tracking**: `YOLO11(images_path)`
- **Custom SORT Algorithm**: `simpleSortAlgorithm(images_path)`

## 🧠 Algorithm Implementation

### SORT Algorithm Components

#### 1. Object Detection

- Uses YOLO 11 (`yolo11s.pt`) for initial object detection
- Configurable confidence threshold (default: 0.5)
- Outputs bounding boxes in `[x1, y1, x2, y2]` format

#### 2. Data Association

The heart of the SORT algorithm uses the **Hungarian Algorithm** for optimal assignment between existing tracks and new detections.

**Cost Metrics**:

- **IoU (Intersection over Union)**: Spatial overlap between bounding boxes
- **Sanchez-Matilla Metric**: Linear cost considering both distance and shape
- **Yu Metric**: Exponential cost with weighted distance and shape components

#### 3. Track Management

- **Track Initialization**: New detections create new tracks with unique IDs
- **Track Update**: Matched detections update existing track positions
- **Track Termination**: Tracks are removed after `MAX_UNMATCHED_AGE` frames without matches

#### 4. State Filtering

- **Minimum Hit Streak**: Tracks must be matched for `MIN_HIT_STREAK` consecutive frames before being displayed
- **Maximum Unmatched Age**: Tracks are deleted after `MAX_UNMATCHED_AGE` consecutive unmatched frames

### Key Parameters

```python
MIN_HIT_STREAK = 1      # Minimum consecutive matches before displaying
MAX_UNMATCHED_AGE = 2   # Maximum consecutive unmatched frames before removal
iou_threshold = 0.3     # Minimum IoU for valid matches
confidence = 0.5        # Detection confidence threshold
```

## 📊 Cost Metrics Explained

### 1. IoU (Intersection over Union)

```text
IoU = Area of Intersection / Area of Union
```

Measures spatial overlap between bounding boxes (0.0 to 1.0).

### 2. Sanchez-Matilla Linear Cost

```text
C(A,B) = C_dist(A,B) × C_shape(A,B)
```

Where:

- `C_dist`: Normalized distance between box centers
- `C_shape`: Normalized difference in box dimensions

### 3. Yu Exponential Cost

```text
C(A,B) = C_dist(A,B) × C_shape(A,B)
C_dist = exp(-w1 × normalized_distance²)
C_shape = exp(-w2 × shape_difference)
```

Exponential decay functions for distance and shape similarity.
