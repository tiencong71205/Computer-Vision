# SafeVision AI

<p align="center">

![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)

![Python](https://img.shields.io/badge/Python-3.9-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red?style=for-the-badge&logo=pytorch)
![OpenCV](https://img.shields.io/badge/OpenCV-ComputerVision-orange?style=for-the-badge&logo=opencv)
![YOLO](https://img.shields.io/badge/YOLO-v8s/v11s-yellow?style=for-the-badge)
![LSTM](https://img.shields.io/badge/LSTM-TemporalLearning-purple?style=for-the-badge)

</p>

# 🔥🚨 Fire & Fall Detection System

Hệ thống nhận diện **cháy nổ** và **té ngã** realtime, chạy trên **Jetson Nano** với pipeline 2 model song song: YOLOv8n cho fire/smoke detection và LSTM temporal model cho fall detection.

---

## 📋 Tổng quan

| Thành phần | Model | mAP50 / F1 | Inference (Jetson Nano TRT) |
|---|---|---|---|
| Fire & Smoke | YOLOv8n | fire: 0.837 · smoke: 0.700 | ~40ms |
| Pose estimation | YOLOv8n-pose | pretrained | ~55ms |
| Fall detection | LSTM (hidden=64) | F1-Fall: 0.63 · Recall: 0.83 | ~1ms |
| **Pipeline tổng** | | | **~15fps @ 416px** |

### Pipeline

```
Camera frame
    │
    ├─── YOLOv8n-fire ──────────────────────── Fire / Smoke alert
    │
    └─── YOLOv8n-pose ──► Keypoints (17 joints)
                │
                ├─── Rule-based posture check
                │
                └─── LSTM (20-frame window) ──► Fall alert
```

---

## 📁 Cấu trúc thư mục

```
fire-fall-detection/
├── README.md
├── requirements.txt
├── config.yaml                      # Cấu hình toàn bộ pipeline
├── .gitignore
│
├── notebooks/                       # Train trên Kaggle T4x2
│   ├── notebook1_fire_detection.ipynb
│   ├── notebook2_extract_keypoints.ipynb
│   └── notebook3_train_lstm.ipynb
│
├── models/                          # Weights (không commit, dùng Git LFS)
│   ├── fire_best.pt
│   ├── fall_lstm_best.pt
│   └── .gitkeep
│
├── src/
│   ├── models/
│   │   └── lstm_model.py            # Định nghĩa FallLSTM
│   ├── pipeline.py                  # Class SafetyPipeline
│   └── utils/
│       ├── keypoint_utils.py
│       └── alert_utils.py
│
├── scripts/
│   ├── inference.py                 # Chạy trên PC (CPU/GPU)
│   ├── export_onnx.py               # Export .pt → .onnx
│   └── benchmark.py                 # Đo FPS
│
└── jetson/
    ├── convert_trt.sh               # ONNX → TensorRT engine
    ├── inference_trt.py             # Inference dùng TRT engine
    └── install.sh                   # Cài dependencies trên Jetson Nano
```

---

## 🚀 Quickstart

### 1. Clone và cài dependencies

```bash
git clone https://github.com/tiencong71205/fire-fall-detection.git
cd fire-fall-detection
pip install -r requirements.txt
```

### 2. Download weights

Download từ [Releases](https://github.com/tiencong71205/fire-fall-detection/releases) và đặt vào thư mục `models/`:

```
models/
├── fire_best.pt
└── fall_lstm_best.pt
```

### 3. Chạy inference

```bash
# Webcam (tự detect GPU/CPU)
python scripts/inference.py --source 0

# File video
python scripts/inference.py --source video.mp4

# Ép dùng CPU
python scripts/inference.py --source 0 --device cpu

# Lưu output
python scripts/inference.py --source 0 --save
```

**Phím tắt:** `Q` thoát · `S` screenshot · `P` pause

---

## 🏋️ Training

Train toàn bộ được thực hiện trên **Kaggle T4x2**. Thứ tự chạy:

### Notebook 1 — Fire Detection (~30 phút)

Mở `notebooks/notebook1_fire_detection.ipynb` trên Kaggle.

```
Dataset: Fire_and_Smoke/ (YOLO format, 9156 train / 872 val / test images)
Model:   YOLOv8n pretrained → fine-tune 100 epochs
Input:   416×416
Result:  mAP50=0.768 (fire=0.837, smoke=0.700)
```

### Notebook 2 — Extract Keypoints (~40 phút)

Mở `notebooks/notebook2_extract_keypoints.ipynb` trên Kaggle.

```
Input:  fall/sequences/ (video frames) + fall/csv/ (accelerometer)
Model:  YOLOv8n-pose pretrained
Output: keypoints/*.npy — mỗi sequence 1 file shape (T, 34)
Label:  acc > 1.5g → đỉnh ngã → cửa sổ 20 frame trước = Fall
```

### Notebook 3 — Train LSTM (~15 phút)

Mở `notebooks/notebook3_train_lstm.ipynb` trên Kaggle.

```
Input:  keypoints.zip từ Notebook 2
Model:  LSTM (hidden=64, 2 layers) + LayerNorm + Dropout
Result: F1-Fall=0.63, Recall=0.83 (val set)
```

---

## 📊 Dataset

| Dataset | Nguồn | Dùng cho |
|---|---|---|
| `Fire_and_Smoke/` | YOLO format, train/valid/test | Fire detection |
| `fallv2/` | 5 class: Blank/Stand/Fall/Lie/Likefall | (dự phòng, feature extraction) |
| `fall/sequences/` | Video frames theo chuỗi | Temporal LSTM |
| `fall/csv/` | Accelerometer data | Label cửa sổ Fall |

> Dataset không được include trong repo. Xem hướng dẫn tại [Kaggle Dataset](https://www.kaggle.com/datasets/nguynteincong/dataset-fall-fire).

---

## 🖥️ Deploy lên Jetson Nano

### Yêu cầu

- Jetson Nano 4GB
- JetPack 4.6+ (TensorRT 8.x)
- Python 3.8+, OpenCV 4.x

### Cài dependencies

```bash
bash jetson/install.sh
```

### Convert sang TensorRT

```bash
bash jetson/convert_trt.sh
```

Script sẽ tự động convert 3 file ONNX:

```bash
trtexec --onnx=models/fire_best.onnx    --saveEngine=models/fire.engine    --fp16
trtexec --onnx=models/yolov8n-pose.onnx --saveEngine=models/pose.engine    --fp16
trtexec --onnx=models/fall_lstm.onnx    --saveEngine=models/lstm.engine    --fp16
```

### Chạy trên Jetson Nano

```bash
python jetson/inference_trt.py --source 0
```

---

## ⚙️ Cấu hình

Chỉnh `config.yaml` để thay đổi các tham số mà không cần sửa code:

```yaml
pipeline:
  seq_len: 20          # Số frame LSTM window
  fall_confirm: 5      # Số frame liên tiếp cần để xác nhận Fall
  cooldown: 30         # Số frame nghỉ sau mỗi lần alert

fire:
  conf: 0.45           # Confidence threshold

inference:
  imgsz: 416           # Input size (416 hoặc 640)
  device: auto         # auto | cuda | cpu
```

---

## 📈 Kết quả thực nghiệm

### Fire Detection (test set)

| Class | Precision | Recall | mAP50 |
|---|---|---|---|
| fire | 0.843 | 0.780 | 0.837 |
| smoke | 0.788 | 0.640 | 0.700 |
| **all** | **0.815** | **0.710** | **0.768** |

### Fall Detection (val set)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Normal | 0.93 | 0.73 | 0.82 |
| Fall | 0.51 | 0.83 | 0.63 |

> Fall Recall = 0.83: model bắt được 83% ca ngã thật. False alarm được giảm thêm bằng confirmation counter (5 frame liên tiếp).

### FPS ước tính trên Jetson Nano

| Cấu hình | FPS |
|---|---|
| 3 model TensorRT FP16, input 416px | ~15fps |
| 3 model TensorRT FP16, input 640px | ~10fps |

---

## 🛠️ Yêu cầu hệ thống

```
# PC training / testing
Python        >= 3.9
PyTorch       >= 2.0
ultralytics   >= 8.4
opencv-python >= 4.8
numpy         >= 1.24

# Jetson Nano
JetPack       >= 4.6
TensorRT      >= 8.0
CUDA          10.2
```

---

## 📝 Ghi chú

- **Smoke detection** có mAP50 thấp hơn fire (0.700 vs 0.837) do khói có hình dạng mờ và dễ nhầm với mây/bụi. Cải thiện bằng cách tăng data augmentation hoặc dùng `yolov8s.pt`.
- **Fall LSTM** được train trên dataset nhỏ (~10 sequences) nên có thể overfit. Kết quả sẽ tốt hơn đáng kể khi có thêm data.
- **Confirmation logic** (5 frame liên tiếp) là bước quan trọng giảm false alarm từ ~49% xuống ~10-15% mà không cần retrain.

---

## 📄 License

MIT License — xem [LICENSE](LICENSE).

# ⭐ If you like this project

Give this repository a star ⭐
