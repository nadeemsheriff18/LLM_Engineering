from flask import Flask, Response, jsonify
from ultralytics import YOLO
import cv2
import serial
import time

app = Flask(__name__)

# ---------------- SETUP ----------------
model = YOLO("yolov8n.pt")

# Bluetooth
arduino = serial.Serial('COM6', 9600, timeout=1)
time.sleep(2)

last_stop_time = 0
STOP_COOLDOWN = 4  # secondspython ar   

# Camera
url = "http://10.230.215.103:4747/video"
cap = cv2.VideoCapture(url)

# Labels
TARGET_LABELS = [
    "bottle", "cup", "cell phone", "remote",
    "backpack", "book", "keyboard", "mouse"
]

# Stats
fps = 0
debris_count = 0
total_detections = 0
last_labels = []
last_sent = None


# ---------------- VIDEO STREAM ----------------
def generate_frames():
    global fps, debris_count, total_detections, last_labels, last_sent, last_stop_time

    prev_time = time.time()
    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        if frame_count % 3 != 0:
            continue

        frame = cv2.resize(frame, (480, 360))
        frame_width = frame.shape[1]

        results = model(frame, imgsz=256, conf=0.5)

        debris_count = 0
        last_labels = []
        detected = False

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]

                if label in TARGET_LABELS:
                    detected = True
                    debris_count += 1
                    last_labels.append(label)

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # ---- POSITION LOGIC ----
                    center = (x1 + x2) // 2

                    if center < frame_width * 0.4:
                        command = 'a'
                    elif center > frame_width * 0.6:
                        command = 'b'
                    else:
                        command = 'c'

                    # ---- SEND BLUETOOTH ----
                    if last_sent != command:
                        arduino.write(command.encode())
                        last_sent = command
                        time.sleep(0.2)

                    # ---- DRAW ----
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, label, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                    break

            if detected:
                break

        current_time = time.time()

        if not detected:
            if (current_time - last_stop_time) > STOP_COOLDOWN:
                arduino.write(b'r')
                last_sent = 'r'
                last_stop_time = current_time
                print("Command: STOP")

        total_detections += debris_count

        # FPS
        curr_time = time.time()
        fps = int(1 / (curr_time - prev_time))
        prev_time = curr_time

        # Encode frame
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# ---------------- ROUTES ----------------
@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stats')
def stats():
    return jsonify({
        "fps": fps,
        "debris_count": debris_count,
        "total_detections": total_detections,
        "last_labels": last_labels
    })


# ---------------- MAIN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)