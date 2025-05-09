import cv2
import numpy as np

# 1. YOLO MODEL FILES (UPDATE THESE PATHS AS NEEDED)
# Paths to the YOLO model files
weights_path = " yolo-coco/yolov3.weights"
cfg_path = " yolo-coco/yolov3.cfg"
names_path = " yolo-coco/coco.names"
# 2. LOAD THE CLASS LABELS
with open(names_path, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# 3. SET UP RANDOM COLORS FOR EACH CLASS (OPTIONAL)
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(class_names), 3), dtype="uint8")

# 4. LOAD THE YOLO MODEL
print("[INFO] Loading YOLO model...")
net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)

# 5. GET THE OUTPUT LAYER NAMES (YOLO)
ln = net.getUnconnectedOutLayersNames()

# 6. DEFINE HSV RANGES FOR TRAFFIC LIGHT COLORS
# You may need to fine-tune these depending on your lighting
RED_LOWER1 = np.array([0, 70, 70])
RED_UPPER1 = np.array([10, 255, 255])
RED_LOWER2 = np.array([170, 70, 70])
RED_UPPER2 = np.array([180, 255, 255])

GREEN_LOWER = np.array([40, 70, 70])
GREEN_UPPER = np.array([90, 255, 255])

YELLOW_LOWER = np.array([15, 70, 70])
YELLOW_UPPER = np.array([35, 255, 255])


def detect_light_color(roi_bgr):
    """
    Given a BGR ROI of a traffic light, determine if it is red, yellow, or green
    using naive color thresholding in HSV.
    """
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    # Red mask (two ranges combined)
    mask_red1 = cv2.inRange(hsv, RED_LOWER1, RED_UPPER1)
    mask_red2 = cv2.inRange(hsv, RED_LOWER2, RED_UPPER2)
    red_mask = mask_red1 + mask_red2
    red_pixels = cv2.countNonZero(red_mask)

    # Green mask
    green_mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
    green_pixels = cv2.countNonZero(green_mask)

    # Yellow mask
    yellow_mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
    yellow_pixels = cv2.countNonZero(yellow_mask)

    # Determine which color has the most pixels
    color_counts = {
        'red': red_pixels,
        'yellow': yellow_pixels,
        'green': green_pixels
    }

    # Return the color (string) with the highest count
    detected_color = max(color_counts, key=color_counts.get)
    return detected_color


# 7. INITIALIZE YOUR CAMERA (0 FOR DEFAULT WEBCAM)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Could not open webcam.")
    exit()

print("[INFO] Starting real-time detection. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Unable to read from camera. Exiting...")
        break

    (H, W) = frame.shape[:2]

    # 8. CREATE A BLOB & FORWARD PASS THROUGH YOLO
    blob = cv2.dnn.blobFromImage(frame,
                                 scalefactor=1 / 255.0,
                                 size=(416, 416),
                                 swapRB=True,
                                 crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(ln)

    boxes = []
    confidences = []
    class_ids = []

    # 9. PARSE YOLO OUTPUT
    for output in layer_outputs:
        # Each detection is [center_x, center_y, width, height, confidence, class1, class2,...]
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                box = detection[0:4] * np.array([W, H, W, H])
                centerX, centerY, width, height = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # 10. NON-MAXIMUM SUPPRESSION (NMS) TO AVOID MULTIPLE OVERLAPPING BOXES
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    # 11. LOOP THROUGH REMAINING DETECTIONS
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, w, h = boxes[i]
            class_id = class_ids[i]
            label = class_names[class_id]
            confidence = confidences[i]

            # Grab a color for the bounding box (optional)
            color_box = [int(c) for c in colors[class_id]]

            # ========== TRAFFIC LIGHT DETECTION & COLOR CLASSIFICATION ========== #
            if label == "traffic light":
                # Crop the region of interest
                roi = frame[y:y + h, x:x + w]

                # Basic check to avoid issues if ROI is out of frame bounds
                if roi.size > 0:
                    # Detect the color of the traffic light
                    light_color = detect_light_color(roi)

                    # Build label with color info
                    text = f"{label} ({light_color}) {confidence:.2f}"

                    # Draw bounding box and label
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color_box, 2)
                    cv2.putText(frame, text, (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_box, 2)
                else:
                    # If ROI is invalid, just draw a rectangle with the basic label
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color_box, 2)
                    cv2.putText(frame, label, (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_box, 2)

            # ========== EXAMPLE: STOP SIGN ========== #
            elif label == "stop sign":
                text = f"{label} {confidence:.2f}"
                cv2.rectangle(frame, (x, y), (x + w, y + h), color_box, 2)
                cv2.putText(frame, text, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_box, 2)

            # ========== OTHER OBJECTS (OPTIONAL) ========== #
            else:
                text = f"{label}: {confidence:.2f}"
                cv2.rectangle(frame, (x, y), (x + w, y + h), color_box, 2)
                cv2.putText(frame, text, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_box, 2)

    # 12. SHOW THE FRAME
    cv2.imshow("YOLO Real-Time Object Detection (Traffic Light Colors)", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 13. CLEAN UP
cap.release()
cv2.destroyAllWindows()
