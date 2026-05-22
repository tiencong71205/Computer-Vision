# SafeVision AI

<p align="center">

![Stars](https://img.shields.io/github/stars/yourusername/safevision-ai?style=for-the-badge)
![Forks](https://img.shields.io/github/forks/yourusername/safevision-ai?style=for-the-badge)
![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)

![Python](https://img.shields.io/badge/Python-3.9-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red?style=for-the-badge&logo=pytorch)
![OpenCV](https://img.shields.io/badge/OpenCV-ComputerVision-orange?style=for-the-badge&logo=opencv)
![YOLO](https://img.shields.io/badge/YOLO-v8s/v11s-yellow?style=for-the-badge)
![LSTM](https://img.shields.io/badge/LSTM-TemporalLearning-purple?style=for-the-badge)

</p>

---

# 🔥 SafeVision AI

> Realtime Fall and Fire Detection Using YOLO Pose and LSTM

SafeVision AI is an intelligent surveillance system using Computer Vision and Deep Learning to detect:

- Human falls
- Fire and smoke
- Dangerous situations in realtime

The project is optimized for realtime camera deployment with a target speed of **20–30 FPS**.

---

# ✨ Features

## 🔥 Fire & Smoke Detection

- Realtime fire detection
- Smoke detection
- YOLO11n-based object detection
- Realtime alert system

---

## 🚨 Fall Detection

- Human pose estimation
- Temporal action recognition
- Multi-class fall classification
- False-positive reduction

---

## ⚡ Realtime AI Pipeline

- YOLOv8n-pose
- YOLO11n
- LSTM temporal learning
- OpenCV video processing
- TensorRT deployment support

---

# 🧠 System Architecture

```text
Camera Stream
      │
      ▼
Frame Extraction
      │
      ├── YOLO11n Fire Detection
      │         │
      │         ▼
      │   Fire / Smoke Alert
      │
      └── YOLOv8n Pose
                │
                ▼
           Human Keypoints
                │
                ▼
              LSTM
                │
                ▼
        Fall Classification
```

---

# 📂 Project Structure

```text
Thi_Giac_May_Tinh/
│
├── datasets/
│   ├── fire/
│   ├── fall/
│   └── fallv2/
│
├── processed/
│
├── models/
│
├── scripts/
│
└── deploy/
```

---

# 📊 Datasets

## 1. Fire-Smoke Dataset

Used for:

- Fire detection
- Smoke detection
- YOLO11n training

Dataset structure:

```text
train/
valid/
test/
```

Each folder contains:

```text
images/
labels/
```

---

## 2. Fall Dataset

Contains:

- RGB frame sequences
- Temporal motion data
- CSV motion features
- Human falling actions

Used for:

- Temporal learning
- Sequence modeling
- LSTM training

Pipeline:

```text
Frames
   ↓
YOLO Pose
   ↓
Keypoints
   ↓
LSTM
```

---

## 3. Fallv2 Dataset

Multi-class action recognition dataset.

### Classes

| Label | Class |
|---|---|
| 0 | Blank |
| 1 | Fall |
| 2 | Lie |
| 3 | LikeFall |
| 4 | Stand |

Used for:

- Semantic posture classification
- False-positive reduction
- Action recognition

---

# 🛠️ Technologies Used

| Component | Technology |
|---|---|
| Object Detection | YOLO11n |
| Pose Estimation | YOLOv8n-pose |
| Temporal Learning | LSTM |
| Framework | PyTorch |
| Video Processing | OpenCV |
| Deployment | ONNX / TensorRT |

---

# ⚡ Expected Performance

| Function | Target FPS |
|---|---|
| Fire Detection | 60–100 FPS |
| Pose Detection | 35–50 FPS |
| Pose + LSTM | 20–35 FPS |

---

# 🚀 Future Improvements

- Jetson Nano deployment
- TensorRT optimization
- Telegram/Email alerts
- Multi-camera monitoring
- Edge AI deployment
- Cloud dashboard integration

---

# 📚 Dataset Citation

## Fall Detection Datasets

- Le2i Fall Detection Dataset
- UR Fall Detection Dataset

## Fire Detection Dataset

- Roboflow Fire & Smoke Dataset

---

# 🔗 Google Drive


https://drive.google.com/file/d/12T_g081M2xbZeL3DtLGPk2fwmQE4wuzl/view?usp=drive_link

---

# 👨‍💻 Author

### Công Nguyễn
### Tuấn Anh
### Đăng Huy

AI Research & Computer Vision Project

Technology Stack:

- YOLO
- LSTM
- PyTorch
- OpenCV
- Computer Vision

---

# ⭐ If you like this project

Give this repository a star ⭐
