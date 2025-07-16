# ICAO detection base
import cv2
import numpy as np
from openvino import Core
import math

core = Core()

face_model = core.read_model("models/face-detection-0200/face-detection-0200.xml")
face_compiled = core.compile_model(face_model, "CPU")
face_input = face_compiled.input(0)
face_output = face_compiled.output(0)

landmark_model = core.read_model("models/landmarks-regression-retail-0009/landmarks-regression-retail-0009.xml")
landmark_compiled = core.compile_model(landmark_model, "CPU")
landmark_input = landmark_compiled.input(0)
landmark_output = landmark_compiled.output(0)

def check_sharpness(img):
    return cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()

def angle_between_eyes(landmarks):
    left, right = landmarks[0], landmarks[1]
    return abs(math.degrees(math.atan2(right[1] - left[1], right[0] - left[0])))

def check_icao(face_box, landmarks, frame_shape):
    xmin, ymin, xmax, ymax = face_box
    h, w = frame_shape[:2]
    head_h = ymax - ymin
    head_ratio = head_h / h

    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    centered_x_val = abs(cx - w / 2)
    centered_y_val = abs(cy - h / 2)

    eyes_y = (landmarks[0][1] + landmarks[1][1]) / 2
    eyes_rel = (eyes_y - ymin) / (ymax - ymin)

    angle = angle_between_eyes(landmarks)
    sharp = check_sharpness(frame[ymin:ymax, xmin:xmax])

    conditions = [
        # 0.7 <= head_ratio <= 0.8,
        0.6 <= head_ratio <= 0.7,
        centered_x_val < 0.1 * w,
        centered_y_val < 0.1 * h,
        # 0.6 <= eyes_rel <= 0.7,
        0.35 <= eyes_rel <= 0.5,
        angle < 5,
        sharp > 50
    ]

    return all(conditions), {
        "head_ratio": head_ratio,
        "centered_x": centered_x_val,
        "centered_y": centered_y_val,
        "eyes_rel": eyes_rel,
        "angle": angle,
        "sharpness": sharp
    }

# cap = cv2.VideoCapture('http://192.168.1.11:8080/video')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_image = cv2.resize(frame, (256, 256)).transpose((2, 0, 1))[None].astype(np.float32)
    output = face_compiled([input_image])[face_output]
    h, w = frame.shape[:2]

    for det in output[0][0]:
        if det[2] > 0.6:
            xmin = int(det[3] * w)
            ymin = int(det[4] * h)
            xmax = int(det[5] * w)
            ymax = int(det[6] * h)

            face = frame[ymin:ymax, xmin:xmax]
            if face.size == 0:
                continue

            face_input = cv2.resize(face, (48, 48)).transpose((2, 0, 1))[None].astype(np.float32)
            landmarks = landmark_compiled([face_input])[landmark_output][0].reshape(-1, 2)
            landmarks[:, 0] = landmarks[:, 0] * (xmax - xmin) + xmin
            landmarks[:, 1] = landmarks[:, 1] * (ymax - ymin) + ymin

            # icao_pass = check_icao((xmin, ymin, xmax, ymax), landmarks, frame.shape)
            icao_pass, metrics = check_icao((xmin, ymin, xmax, ymax), landmarks, frame.shape)
            
            status = "ICAO OK" if icao_pass else "ICAO FAIL"
            color = (0, 255, 0) if icao_pass else (0, 0, 255)

            y_text = ymin - 30
            for key, val in metrics.items():
                text = f"{key}: {val:.2f}"
                cv2.putText(frame, text, (xmin, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_text -= 15
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(frame, status, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            for (x, y) in landmarks.astype(int):
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

    cv2.imshow("ICAO Check", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
