# ICAO detection base
import cv2
import numpy as np
from openvino import Core
import math

core = Core()

emotion_model = core.read_model("models/emotions-recognition-retail-0003/emotions-recognition-retail-0003.xml")
emotion_compiled = core.compile_model(emotion_model, "CPU")
emotion_input_layer = emotion_compiled.input(0)
emotion_output_layer = emotion_compiled.output(0)

face_model = core.read_model("models/face-detection-0200/face-detection-0200.xml")
face_compiled = core.compile_model(face_model, "CPU")
face_input = face_compiled.input(0)
face_output = face_compiled.output(0)

landmark_model = core.read_model("models/landmarks-regression-retail-0009/landmarks-regression-retail-0009.xml")
landmark_compiled = core.compile_model(landmark_model, "CPU")
landmark_input = landmark_compiled.input(0)
landmark_output = landmark_compiled.output(0)

eye_model = core.read_model("models/open-closed-eye-0001/open_closed_eye.xml")
eye_compiled = core.compile_model(eye_model, "CPU")
eye_input = eye_compiled.input(0)
eye_output = eye_compiled.output(0)

headpose_model = core.read_model("models/head-pose-estimation-adas-0001/head-pose-estimation-adas-0001.xml")
headpose_compiled = core.compile_model(headpose_model, "CPU")
headpose_input = headpose_compiled.input(0)
angle_yaw_output = headpose_compiled.output("angle_y_fc")
angle_pitch_output = headpose_compiled.output("angle_p_fc")
angle_roll_output = headpose_compiled.output("angle_r_fc")


def preprocess_emotion(face_img):
    resized = cv2.resize(face_img, (64, 64))
    blob = resized.transpose(2, 0, 1)[np.newaxis, :].astype(np.float32)
    return blob

def check_eye_open(eye_img):
    eye_input_img = cv2.resize(eye_img, (32, 32)).astype(np.float32)
    eye_input_img = (eye_input_img - 127.0) / 255.0
    eye_input_img = eye_input_img.transpose(2, 0, 1)[None]

    result = eye_compiled([eye_input_img])[eye_output]
    probs = result[0].reshape(-1)
    state = np.argmax(probs)  # 0=open, 1=closed
    return state == 0

def check_sharpness(img):
    return cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()

def angle_between_eyes(landmarks):
    left, right = landmarks[0], landmarks[1]
    return abs(math.degrees(math.atan2(right[1] - left[1], right[0] - left[0])))

def check_icao(face_box, landmarks, frame_shape, left_eye_closed, right_eye_closed):
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
        0.62 <= head_ratio <= 0.73,     # tightened from 0.6–0.7 for stricter size
        centered_x_val < 0.15 * w,      # tightened centering tolerance (7%)
        centered_y_val < 0.15 * h,      # tightened vertical centering tolerance
        0.37 <= eyes_rel <= 0.48,       # stricter vertical eye position range
        angle < 3,                      # max eye angle reduced from 5 degrees
        sharp > 80,                     # increased sharpness threshold
        left_eye_closed != True,                     # increased sharpness threshold
        right_eye_closed != True                     # increased sharpness threshold
    ]

    return all(conditions), {
        "left_eye_open": float(left_eye_closed),
        "right_eye_open": float(right_eye_closed),
        "head_ratio": head_ratio,
        "centered_x": centered_x_val,
        "centered_y": centered_y_val,
        "eyes_rel": eyes_rel,
        "angle": angle,
        "sharpness": sharp,
    }

eye_crop_size = 20

def crop_eye(img, center, size):
    x, y = center
    x1 = max(x - size, 0)
    y1 = max(y - size, 0)
    x2 = min(x + size, img.shape[1]-1)
    y2 = min(y + size, img.shape[0]-1)
    return img[y1:y2, x1:x2]


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

            emotions = ['neutral', 'happy', 'sad', 'surprise', 'anger']
            emotion_blob = preprocess_emotion(face)
            emotion_result = emotion_compiled([emotion_blob])[emotion_output_layer]
            emotion_probs = emotion_result[0]
            emotion_id = np.argmax(emotion_probs)
            emotion_label = emotions[emotion_id]

            face_input = cv2.resize(face, (48, 48)).transpose((2, 0, 1))[None].astype(np.float32)
            landmarks = landmark_compiled([face_input])[landmark_output][0].reshape(-1, 2)
            landmarks[:, 0] = landmarks[:, 0] * (xmax - xmin) + xmin
            landmarks[:, 1] = landmarks[:, 1] * (ymax - ymin) + ymin

            head_input = cv2.resize(face, (60, 60)).transpose(2, 0, 1)[None].astype(np.float32)
            results = headpose_compiled([head_input])
            yaw = results[angle_yaw_output][0][0]
            pitch = results[angle_pitch_output][0][0]
            roll = results[angle_roll_output][0][0]

            head_pose_pass = abs(yaw) < 5 and abs(pitch) < 5 and abs(roll) < 5
            
            left_eye_center = landmarks[0].astype(int)
            right_eye_center = landmarks[1].astype(int)

            left_eye_img = crop_eye(frame, left_eye_center, eye_crop_size)
            right_eye_img = crop_eye(frame, right_eye_center, eye_crop_size)

            left_eye_closed = check_eye_open(left_eye_img)
            right_eye_closed = check_eye_open(right_eye_img)

            icao_base_pass, metrics = check_icao((xmin, ymin, xmax, ymax), landmarks, frame.shape, left_eye_closed , right_eye_closed)

            icao_pass = (
                icao_base_pass and
                left_eye_closed is not False and
                right_eye_closed is not False and
                emotion_label == 'neutral' or emotion_label == 'sad'  
            )

            status = "ICAO OK" if icao_pass else "ICAO FAIL"
            color = (0, 255, 0) if icao_pass else (0, 0, 255)


            y_text = ymin - 30
            for key, val in metrics.items():
                text = f"{key}: {val:.2f}"
                cv2.putText(frame, text, (xmin, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_text -= 15
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(frame, status, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"Emotion: {emotion_label}", (ymin,xmin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(frame, f"Yaw: {yaw:.1f}", (ymin,xmin - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(frame, f"Pitch: {pitch:.1f}", (ymin,xmin - 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(frame, f"Roll: {roll:.1f}", (ymin,xmin - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            for (x, y) in landmarks.astype(int):
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

    cv2.imshow("ICAO Check", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
