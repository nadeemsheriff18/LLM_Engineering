from ultralytics import YOLO
import cv2

# ---------------- SETUP ----------------
model = YOLO("yolov8n.pt")

url = "http://192.168.0.102:4747/video"
cap = cv2.VideoCapture(url)

TARGET_LABELS = [
    "bottle", "cup", "cell phone", "remote",
    "backpack", "book", "keyboard", "mouse"
]

frame_count = 0

# ---------------- LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Reduce lag
    frame_count += 1
    if frame_count % 3 != 0:
        continue

    frame = cv2.resize(frame, (480, 360))
    frame_width = frame.shape[1]

    results = model(frame, imgsz=256, conf=0.5)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label in TARGET_LABELS:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # ---- POSITION LOGIC ----
                center = (x1 + x2) // 2

                if center < frame_width * 0.4:
                    direction = "LEFT"
                elif center > frame_width * 0.6:
                    direction = "RIGHT"
                else:
                    direction = "FORWARD"

                # ---- DRAW ----
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"{label} ({direction})",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0,255,0), 2)

                break  # only first object

    # ---- OPTIONAL: draw center zones ----
    cv2.line(frame, (int(frame_width*0.4), 0), (int(frame_width*0.4), 360), (255,0,0), 1)
    cv2.line(frame, (int(frame_width*0.6), 0), (int(frame_width*0.6), 360), (255,0,0), 1)

    # ---- DISPLAY ----
    cv2.imshow("YOLO Detection Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------- CLEANUP ----------------
cap.release()
cv2.destroyAllWindows()