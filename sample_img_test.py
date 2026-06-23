

import os
import cv2
from ultralytics import YOLO

model = YOLO(r"D:\sanjay\kpit\to_upload\models\yolov8n.pt")

image_path = r"D:\sanjay\kpit\archive\bottle.jpeg"
output_folder = r"D:\sanjay\kpit\output_folder"

# Create folder if not exists
os.makedirs(output_folder, exist_ok=True)

img = cv2.imread(image_path)

pixels_per_cm = 32

results = model(img)[0]

for box in results.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    conf = float(box.conf[0])
    cls = int(box.cls[0])

    label = model.names[cls]

    # -----------------------------
    # 📏 MEASUREMENTS
    # -----------------------------
    pixel_width = x2 - x1
    pixel_height = y2 - y1

    width_cm = pixel_width / pixels_per_cm
    height_cm = pixel_height / pixels_per_cm

    # -----------------------------
    # PRINT VALUES
    # -----------------------------
    print("\n📌 Detection Found")
    print(f"Class      : {label}")
    print(f"Confidence : {conf:.2f}")
    print(f"Width      : {width_cm:.2f} cm")
    print(f"Height     : {height_cm:.2f} cm")

    # -----------------------------
    # Draw bounding box
    # -----------------------------
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display label + confidence + size
    text = f"{label}  W:{width_cm:.2f}cm H:{height_cm:.2f}cm"
    cv2.putText(img, text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

# -----------------------------
# 💾 SAVE OUTPUT IMAGE
# -----------------------------
output_path = os.path.join(output_folder, "output.jpg")
cv2.imwrite(output_path, img)

print(f"\n✅ Output saved at: {output_path}")

cv2.imshow("Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()