# Autonomous Vehicle Control Modules

This project contains two Python-based computer vision modules developed as part of an autonomous vehicle prototype running on Raspberry Pi 5. The modules support:

1. **Eye-Controlled Emergency Braking**  
   Uses blink detection with MediaPipe and OpenCV. If the driver keeps eyes closed for more than 5 seconds, the vehicle stops.

2. **Traffic Light & Stop Sign Detection**  
   Uses YOLOv3 object detection with color classification in HSV to detect traffic lights and their states (Red, Yellow, Green), as well as stop signs.

---

## 🛠️ Features

- 👁️ Blink detection using eye aspect ratio and face mesh landmarks  
- 🚦 Traffic light color recognition with YOLO and HSV filtering  
- 🛑 Stop sign detection using YOLOv3  
- ⚙️ Integrated with Raspberry Pi 5 and Arduino for real-world deployment

---

## 📂 Folder Structure

```bash
src/
├── eye_control.py            # Detects blinks and controls car state
└── traffic_light_yolo.py     # Real-time object detection for traffic lights and stop signs

yolo-coco/
├── yolov3.cfg
├── yolov3.weights  # (download separately)
└── coco.names
