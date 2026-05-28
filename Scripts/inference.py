"""
Inference hoàn chỉnh — Fire & Fall Detection
=============================================
Chạy pipeline đầy đủ trên webcam hoặc video file.

Cài thư viện:
    pip install ultralytics opencv-python torch numpy

Cách chạy:
    # Webcam, tự detect GPU/CPU
    python inference.py --source 0

    # File video
    python inference.py --source video.mp4

    # Ép dùng CPU
    python inference.py --source 0 --device cpu

    # Ép dùng GPU
    python inference.py --source 0 --device cuda

    # Lưu output ra file
    python inference.py --source video.mp4 --save

Phím tắt khi đang chạy:
    Q  →  thoát
    S  →  chụp screenshot frame hiện tại
    P  →  pause / resume
"""

import argparse
import time
import sys
from collections import deque
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════════════════════
# 1. ĐỊNH NGHĨA MODEL LSTM (phải khớp với lúc train)
# ══════════════════════════════════════════════════════════════════════════════

class FallLSTM(nn.Module):
    def __init__(self, input_size=34, hidden=64, n_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


# ══════════════════════════════════════════════════════════════════════════════
# 2. CLASS PIPELINE CHÍNH
# ══════════════════════════════════════════════════════════════════════════════

class SafetyPipeline:
    def __init__(self, fire_model_path, lstm_model_path, device, seq_len=20, conf_fire=0.45):
        from ultralytics import YOLO

        self.device   = device
        self.seq_len  = seq_len
        self.conf_fire = conf_fire

        print(f"\n{'='*55}")
        print(f"  Device   : {device}")
        print(f"  seq_len  : {seq_len} frames")
        print(f"  conf_fire: {conf_fire}")
        print(f"{'='*55}")

        # ── Load models ──────────────────────────────────────────────────────
        print("\nLoading models...")

        print("  [1/3] YOLOv8n-fire ...", end=" ", flush=True)
        self.fire_model = YOLO(fire_model_path)
        print("✓")

        print("  [2/3] YOLOv8n-pose ...", end=" ", flush=True)
        self.pose_model = YOLO("yolov8n-pose.pt")  # auto-download nếu chưa có
        print("✓")

        print("  [3/3] Fall LSTM     ...", end=" ", flush=True)
        self.lstm = FallLSTM().to(device)
        state = torch.load(lstm_model_path, map_location=device)
        self.lstm.load_state_dict(state)
        self.lstm.eval()
        print("✓")

        # ── Buffer keypoints ─────────────────────────────────────────────────
        self.kp_buffer = deque(maxlen=seq_len)   # rolling window

        # ── Fall confirmation counter ─────────────────────────────────────────
        self.fall_counter   = 0
        self.FALL_CONFIRM   = 5    # cần 5 frame liên tiếp mới báo
        self.cooldown_timer = 0
        self.COOLDOWN       = 30   # 30 frame sau alert không báo tiếp

        # ── Stats ─────────────────────────────────────────────────────────────
        self.frame_count  = 0
        self.fire_alerts  = 0
        self.fall_alerts  = 0
        self.t_start      = time.time()

        print("\nReady ✓\n")

    # ── Lấy keypoints từ 1 frame ─────────────────────────────────────────────
    def _get_keypoints(self, frame):
        results = self.pose_model(
            frame, imgsz=416, conf=0.3,
            device=self.device, verbose=False
        )
        r = results[0]
        if r.keypoints is not None and len(r.keypoints.xyn) > 0:
            boxes = r.boxes.xyxy.cpu().numpy()
            areas = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])
            best  = int(np.argmax(areas))
            kp    = r.keypoints.xyn[best].cpu().numpy()   # (17, 2)
            # Vẽ skeleton
            annotated = r.plot()
        else:
            kp        = np.zeros((17, 2), dtype=np.float32)
            annotated = frame.copy()
        return kp.flatten().astype(np.float32), annotated

    # ── Fire detection ────────────────────────────────────────────────────────
    def _detect_fire(self, frame):
        results = self.fire_model(
            frame, imgsz=416, conf=self.conf_fire,
            device=self.device, verbose=False
        )
        r = results[0]
        detections = []
        if r.boxes is not None:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                label = r.names[cls_id]
                detections.append((label, conf, x1, y1, x2, y2))
        return detections

    # ── Fall detection từ buffer ──────────────────────────────────────────────
    def _detect_fall(self):
        if len(self.kp_buffer) < self.seq_len:
            return False, 0.0

        seq    = np.stack(list(self.kp_buffer))          # (seq_len, 34)
        tensor = torch.tensor(seq).unsqueeze(0).to(self.device)  # (1, seq_len, 34)

        with torch.no_grad():
            logits = self.lstm(tensor)
            prob   = torch.softmax(logits, dim=1)[0, 1].item()   # prob Fall

        is_fall = prob > 0.5
        return is_fall, prob

    # ── Xử lý confirmation + cooldown ────────────────────────────────────────
    def _update_fall_state(self, is_fall_raw):
        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1
            return False

        if is_fall_raw:
            self.fall_counter += 1
        else:
            self.fall_counter = max(0, self.fall_counter - 1)

        if self.fall_counter >= self.FALL_CONFIRM:
            self.fall_counter   = 0
            self.cooldown_timer = self.COOLDOWN
            self.fall_alerts   += 1
            return True
        return False

    # ── Vẽ overlay thông tin ──────────────────────────────────────────────────
    def _draw_overlay(self, frame, fire_dets, fall_confirmed, fall_prob, fps):
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # ── FPS + Stats bar ───────────────────────────────────────────────────
        cv2.rectangle(overlay, (0,0), (w, 36), (20,20,20), -1)
        elapsed = time.time() - self.t_start
        cv2.putText(overlay, f"FPS: {fps:4.1f}  |  Frame: {self.frame_count}  |  "
                              f"Fire alerts: {self.fire_alerts}  |  Fall alerts: {self.fall_alerts}  |  "
                              f"Elapsed: {elapsed:.0f}s",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)

        # ── Fall probability bar (bottom) ─────────────────────────────────────
        bar_w  = int(w * fall_prob)
        bar_h  = 18
        bar_y  = h - bar_h
        cv2.rectangle(overlay, (0, bar_y), (w, h), (30,30,30), -1)
        bar_color = (0, int(255*(1-fall_prob)), int(255*fall_prob))
        cv2.rectangle(overlay, (0, bar_y), (bar_w, h), bar_color[::-1], -1)
        cv2.putText(overlay, f"Fall prob: {fall_prob:.2f}",
                    (6, h-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # ── Fire detections ───────────────────────────────────────────────────
        for label, conf, x1, y1, x2, y2 in fire_dets:
            color = (0, 60, 255) if label == "fire" else (0, 140, 255)
            cv2.rectangle(overlay, (x1,y1), (x2,y2), color, 2)
            cv2.rectangle(overlay, (x1, y1-22), (x1+160, y1), color, -1)
            cv2.putText(overlay, f"{label} {conf:.2f}",
                        (x1+4, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
            self.fire_alerts += 1 if False else 0   # counted elsewhere

        # ── ALERT overlays ────────────────────────────────────────────────────
        if fire_dets:
            cv2.rectangle(overlay, (0, 40), (w, 100), (0, 0, 180), -1)
            cv2.putText(overlay, "⚠  FIRE / SMOKE DETECTED",
                        (w//2 - 200, 85), cv2.FONT_HERSHEY_DUPLEX, 1.1, (255,255,255), 2)

        if fall_confirmed:
            cv2.rectangle(overlay, (0, 105 if fire_dets else 40),
                          (w, 165 if fire_dets else 100), (0, 120, 0), -1)
            cv2.putText(overlay, "⚠  FALL DETECTED",
                        (w//2 - 160, 155 if fire_dets else 85),
                        cv2.FONT_HERSHEY_DUPLEX, 1.1, (255,255,255), 2)

        # ── Fall counter progress ─────────────────────────────────────────────
        if 0 < self.fall_counter < self.FALL_CONFIRM:
            cv2.putText(overlay, f"Confirming fall: {self.fall_counter}/{self.FALL_CONFIRM}",
                        (w-280, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100,220,100), 2)

        # Blend overlay
        frame[:] = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)
        return frame

    # ── Hàm chính: xử lý 1 frame ─────────────────────────────────────────────
    def process(self, frame):
        self.frame_count += 1

        # 1. Fire detection
        fire_dets = self._detect_fire(frame)
        if fire_dets:
            self.fire_alerts += 1

        # 2. Pose → keypoints
        kp, annotated = self._get_keypoints(frame)
        self.kp_buffer.append(kp)

        # 3. LSTM fall detection
        is_fall_raw, fall_prob = self._detect_fall()
        fall_confirmed = self._update_fall_state(is_fall_raw)

        return annotated, fire_dets, fall_confirmed, fall_prob


# ══════════════════════════════════════════════════════════════════════════════
# 3. MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

def main(args):
    # ── Chọn device ──────────────────────────────────────────────────────────
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"\nDevice được chọn: {device.upper()}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**2} MB")

    # ── Kiểm tra file model ───────────────────────────────────────────────────
    for path, name in [(args.fire_model, "fire model"), (args.lstm_model, "LSTM model")]:
        if not Path(path).exists():
            print(f"\n[LỖI] Không tìm thấy {name}: {path}")
            print("  Hãy download từ Kaggle và để đúng chỗ.")
            sys.exit(1)

    # ── Mở video / webcam ─────────────────────────────────────────────────────
    source = int(args.source) if args.source.isdigit() else args.source
    cap    = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"\n[LỖI] Không mở được source: {args.source}")
        sys.exit(1)

    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"\nSource: {args.source}  ({W}x{H} @ {FPS:.0f}fps)")

    # ── Video writer (nếu --save) ─────────────────────────────────────────────
    writer = None
    if args.save:
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"output_{ts}.mp4"
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        writer   = cv2.VideoWriter(out_path, fourcc, FPS, (W, H))
        print(f"Saving to: {out_path}")

    # ── Khởi tạo pipeline ────────────────────────────────────────────────────
    pipeline = SafetyPipeline(
        fire_model_path = args.fire_model,
        lstm_model_path = args.lstm_model,
        device          = device,
        seq_len         = args.seq_len,
        conf_fire       = args.conf_fire,
    )

    # ── Vòng lặp chính ────────────────────────────────────────────────────────
    t_prev = time.time()
    fps    = 0.0
    paused = False

    print("Đang chạy... (Q: thoát | S: screenshot | P: pause)\n")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                if not isinstance(source, int):
                    print("Video kết thúc.")
                break

            # Đo FPS
            t_now = time.time()
            fps   = 0.9 * fps + 0.1 * (1.0 / max(t_now - t_prev, 1e-6))
            t_prev = t_now

            # Xử lý frame
            annotated, fire_dets, fall_confirmed, fall_prob = pipeline.process(frame)

            # Vẽ overlay
            annotated = pipeline._draw_overlay(
                annotated, fire_dets, fall_confirmed, fall_prob, fps
            )

            # In terminal khi có alert
            if fire_dets:
                labels = [f"{l}({c:.2f})" for l, c, *_ in fire_dets]
                print(f"[Frame {pipeline.frame_count:5d}] 🔥 FIRE: {', '.join(labels)}")
            if fall_confirmed:
                print(f"[Frame {pipeline.frame_count:5d}] 🚨 FALL CONFIRMED!")

            if writer:
                writer.write(annotated)

        # Hiển thị
        cv2.imshow("Safety Monitor — Q:quit | S:screenshot | P:pause", annotated)

        # Phím tắt
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
            fn  = f"screenshot_{ts}.jpg"
            cv2.imwrite(fn, annotated)
            print(f"Screenshot: {fn}")
        elif key == ord('p'):
            paused = not paused
            print("PAUSED" if paused else "RESUMED")

    # ── Cleanup ────────────────────────────────────────────────────────────────
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - pipeline.t_start
    print(f"\n{'='*50}")
    print(f"  Tổng frames  : {pipeline.frame_count}")
    print(f"  Thời gian    : {elapsed:.1f}s")
    print(f"  FPS trung bình: {pipeline.frame_count/elapsed:.1f}")
    print(f"  Fire alerts  : {pipeline.fire_alerts}")
    print(f"  Fall alerts  : {pipeline.fall_alerts}")
    print(f"{'='*50}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 4. ARGS
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fire & Fall Detection")
    parser.add_argument("--source",     default="0",
                        help="0=webcam, hoặc path video (default: 0)")
    parser.add_argument("--fire_model", default="fire_best.pt",
                        help="Path tới fire_best.pt (download từ Kaggle)")
    parser.add_argument("--lstm_model", default="fall_lstm_best.pt",
                        help="Path tới fall_lstm_best.pt (download từ Kaggle)")
    parser.add_argument("--device",     default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Device: auto | cuda | cpu (default: auto)")
    parser.add_argument("--seq_len",    type=int, default=20,
                        help="Số frame LSTM window (default: 20)")
    parser.add_argument("--conf_fire",  type=float, default=0.45,
                        help="Confidence threshold cho fire (default: 0.45)")
    parser.add_argument("--save",       action="store_true",
                        help="Lưu video output ra file .mp4")
    args = parser.parse_args()
    main(args)
