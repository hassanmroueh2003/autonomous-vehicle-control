import cv2 as cv
import mediapipe as mp
import time
import math
import numpy as np

# Variables
frame_counter = 0
CEF_COUNTER = 0
TOTAL_BLINKS = 0
CLOSED_EYES_FRAME = 3  # Frames threshold for blink detection
FONTS = cv.FONT_HERSHEY_COMPLEX

# Blink timing variables
blink_start_time = None
open_start_time = None
blink_threshold = 5  # seconds before car stops
open_threshold = 1  # seconds before car reactivates
car_stopped = False  # State of the car (stopped or moving)

# Face mesh setup
map_face_mesh = mp.solutions.face_mesh
camera = cv.VideoCapture(0)

# Eye landmarks
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]


# Landmark detection function
def landmarksDetection(img, results):
    img_height, img_width = img.shape[:2]
    return [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]


# Euclidean distance function
def euclideanDistance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# Blink ratio function
def blinkRatio(landmarks, right_indices, left_indices):
    # Right eye
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]

    # Left eye
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    # Compute distances
    rhDistance = euclideanDistance(rh_right, rh_left)
    rvDistance = euclideanDistance(rv_top, rv_bottom)
    lhDistance = euclideanDistance(lh_right, lh_left)
    lvDistance = euclideanDistance(lv_top, lv_bottom)

    reRatio = rhDistance / rvDistance
    leRatio = lhDistance / lvDistance

    return (reRatio + leRatio) / 2


# Face mesh detection
with map_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    start_time = time.time()

    while True:
        frame_counter += 1
        ret, frame = camera.read()
        if not ret:
            break

        frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            mesh_coords = landmarksDetection(frame, results)
            ratio = blinkRatio(mesh_coords, RIGHT_EYE, LEFT_EYE)

            # Draw circles around the eyes
            cv.polylines(frame, [np.array([mesh_coords[p] for p in LEFT_EYE], dtype=np.int32)], True, (0, 255, 0), 1,
                         cv.LINE_AA)
            cv.polylines(frame, [np.array([mesh_coords[p] for p in RIGHT_EYE], dtype=np.int32)], True, (0, 255, 0), 1,
                         cv.LINE_AA)

            if ratio > 5.5:  # Eyes closed
                if blink_start_time is None:
                    blink_start_time = time.time()
                elapsed_time = time.time() - blink_start_time

                if elapsed_time >= blink_threshold and not car_stopped:
                    car_stopped = True  # Car stops moving
                    cv.putText(frame, "Car Stops Moving", (50, 100), FONTS, 1.5, (0, 0, 255), 3)

                open_start_time = None  # Reset open eye timer

            else:  # Eyes open
                if open_start_time is None:
                    open_start_time = time.time()
                elapsed_time_open = time.time() - open_start_time

                if elapsed_time_open >= open_threshold and car_stopped:
                    car_stopped = False  # Car starts moving again
                    cv.putText(frame, "Car Moving", (50, 100), FONTS, 1.5, (0, 255, 0), 3)

                blink_start_time = None  # Reset blink timer

            # Display car status
            if car_stopped:
                cv.putText(frame, "Car Stops Moving", (50, 100), FONTS, 1.5, (0, 0, 255), 3)
            else:
                cv.putText(frame, "Car Moving", (50, 100), FONTS, 1.5, (0, 255, 0), 3)

        # Display frame
        cv.imshow('Eye Tracking', frame)
        key = cv.waitKey(2)
        if key == ord('q') or key == ord('Q'):
            break

    cv.destroyAllWindows()
    camera.release()
