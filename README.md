# Autonomous Vehicle Control Modules

This project contains two Python-based computer vision modules developed for an autonomous vehicle prototype running on Raspberry Pi 5:

1. **Eye Blink Detection** – Stops the vehicle if the driver keeps their eyes closed for 5 seconds.  
2. **Traffic Light & Stop Sign Detection** – Uses YOLOv3 and HSV color filtering to detect traffic light states and stop signs.

---

## 🔍 Project Description

This project includes two real-time computer vision modules developed for an autonomous vehicle prototype on Raspberry Pi 5. The first module stops the car when the driver’s eyes remain closed for 5 seconds. The second uses YOLOv3 and HSV filtering to detect traffic lights and stop signs.

---

## 📁 Files Included

- `eye_control.py` — Real-time blink detection using MediaPipe
- `traffic_light_yolo.py` — YOLOv3-based traffic light and stop sign detection

---

## ⚠️ YOLOv3 Files Required

To run `traffic_light_yolo.py`, download the following YOLOv3 files and place them in a folder named `yolo-coco/` in the same directory:

1. `yolov3.cfg` – Model configuration  
2. `yolov3.weights` – Pre-trained model weights  
3. `coco.names` – Class labels file  

📥 You can get them from: [https://pjreddie.com/darknet/yolo/](https://pjreddie.com/darknet/yolo/)

Your folder structure should look like:

```bash
project-root/
├── eye_control.py
├── traffic_light_yolo.py
└── yolo-coco/
    ├── yolov3.cfg
    ├── yolov3.weights
    └── coco.names
````

---

## 🧩 Dependencies

Install required packages:

```bash
pip install opencv-python mediapipe numpy
```

Or use the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Eye Blink Detection

```bash
python eye_control.py
```

### Traffic Light & Stop Sign Detection

Ensure you have the YOLO files in `yolo-coco/`, then run:

```bash
python traffic_light_yolo.py
```

---

## 👤 Author

**\[Your Name]** – Computer vision modules developer

```

---

Would you like me to generate the full `requirements.txt` and `.gitignore` too, or are you already handling those?
```
