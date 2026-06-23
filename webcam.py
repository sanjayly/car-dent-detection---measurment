import cv2
from ultralytics import YOLO

model = YOLO(r"D:\sanjay\kpit\models\yolo26m.pt")

# === MULTI-REFERENCE OBJECTS ===
REFERENCE_OBJECTS = {
    "person": 45,
    "bottle": 7,
    "cup": 8,
    "laptop": 35,
    "mouse": 6,
    "keyboard": 45,
    "book": 20,
    "cell phone": 7,
    "tv": 100,
    "chair": 40
}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    PIXEL_TO_CM = None

    # === STEP 1: AUTO SCALE ===
    for box in results.boxes:
        label = model.names[int(box.cls[0])]

        if label in REFERENCE_OBJECTS:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            pixel_width = x2 - x1
            real_width = REFERENCE_OBJECTS[label]

            PIXEL_TO_CM = real_width / pixel_width
            break

    # === STEP 2: DETECTION + MEASUREMENT ===
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = model.names[int(box.cls[0])]
        conf = float(box.conf[0])

        width_px = x2 - x1
        height_px = y2 - y1

        if PIXEL_TO_CM:
            width_cm = width_px * PIXEL_TO_CM
            height_cm = height_px * PIXEL_TO_CM
            size_text = f"{width_cm:.1f} x {height_cm:.1f} cm"
            
        else:
            size_text = "No scale"

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Label
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Measurement
        cv2.putText(frame, size_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow("Detection + Measurement", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()