
import os
import cv2
from ultralytics import YOLO

# -----------------------------
# CONFIG
# -----------------------------
model_path = r"D:\sanjay\kpit\best.pt"
input_folder = r"D:\sanjay\kpit\archive\test"
output_folder = r"D:\sanjay\kpit\output_folder"


pixels_per_cm = 10  # Example value (CHANGE THIS)

CONF_THRESHOLD = 0.75

os.makedirs(output_folder, exist_ok=True)

# Load model
model = YOLO(model_path)

image_extensions = (".jpg", ".jpeg", ".png", ".bmp")

# -----------------------------
# PROCESS ALL IMAGES
# -----------------------------
for img_name in os.listdir(input_folder):

    if not img_name.lower().endswith(image_extensions):
        continue

    img_path = os.path.join(input_folder, img_name)
    img = cv2.imread(img_path)

    if img is None:
        print(f"❌ Failed to read {img_name}")
        continue

    results = model(img)[0]

    for box in results.boxes:
        conf = float(box.conf[0])

        # ✅ FILTER BY CONFIDENCE
        if conf < CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        label = model.names[cls]

        # -----------------------------
        # 📏 MEASUREMENT (REAL WORLD)
        # -----------------------------
        pixel_width = x2 - x1
        pixel_height = y2 - y1

        width_cm = pixel_width / pixels_per_cm
        height_cm = pixel_height / pixels_per_cm
        area_cm2 = width_cm * height_cm

        # -----------------------------
        # DRAW BOUNDING BOX
        # -----------------------------
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # -----------------------------
        # TEXT DISPLAY
        # -----------------------------
        text1 = f"{label} ({conf:.2f})"
        text2 = f"W:{width_cm:.2f}cm H:{height_cm:.2f}cm"
        text3 = f"Area:{area_cm2:.2f} cm2"

        cv2.putText(img, text1, (x1, y1 - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(img, text2, (x1, y1 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        cv2.putText(img, text3, (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Print values
        print(f"\n📌 {img_name}")
        print(f"Class: {label}")
        print(f"Confidence: {conf:.2f}")
        print(f"Width: {width_cm:.2f} cm")
        print(f"Height: {height_cm:.2f} cm")
        print(f"Area: {area_cm2:.2f} cm²")

    # Save output
    output_path = os.path.join(output_folder, img_name)
    cv2.imwrite(output_path, img)

    print(f"✅ Processed: {img_name}")

    # Optional display
    cv2.imshow("Detection", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()

print("\n🎯 All images processed successfully!")