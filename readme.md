# 🚗 Car Dent Detection & Measurement using YOLO

## 📌 Overview

This project focuses on detecting **car dents** using a YOLO model and estimating their **width, height, and area** from images. It includes the complete pipeline from data labeling → training → testing.

---

# 🧰 1. Requirements & Installation

## 🔹 Install Dependencies

Create a virtual environment (recommended):

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

Install required packages:

```bash
pip install -r requirements.txt
```

## 🔹 Sample `requirements.txt`

```txt
ultralytics
opencv-python
numpy
matplotlib
```

---

# 🏷️ 2. Data Annotation using LabelImg

## 🔹 Install LabelImg

```bash
pip install labelImg
labelImg
```

## 🔹 Steps to Label Images

1. Open **LabelImg**
2. Select your image folder
3. Choose **YOLO format**
4. Draw bounding boxes around car dents
5. Assign class name (e.g., `dent`)
6. Save annotations

👉 This generates `.txt` files for each image.

---

## 📂 Dataset Structure

Make sure your dataset is structured like this:

```bash
dataset/
│
├── images/
│   ├── train/
│   └── val/
│
├── labels/
│   ├── train/
│   └── val/
```

---

## 📄 Create `data.yaml`

```yaml
path: dataset
train: images/train
val: images/val

names:
  0: dent
```

---

# 🧠 3. Train YOLO Model

## 🔹 Training Command (CPU)

```bash
yolo detect train \
data=data.yaml \
model=yolov8n.pt \
epochs=50 \
imgsz=640 \
batch=4 \
device=cpu
```

## 🔹 Output

After training, best model will be saved at:

```bash
runs/detect/train/weights/best.pt
```

---

# 🧪 4. Test the Model on Images

## 🔹 Run Detection Script

Update paths in your script:

```python
model_path = "best.pt"
input_folder = "test_images"
output_folder = "output_images"
```

Run:

```bash
python detect.py
```

---

## 📊 Output Features

For each detected dent:

* ✅ Bounding box
* ✅ Confidence score (>75%)
* ✅ Width (cm)
* ✅ Height (cm)
* ✅ Area (cm²)

---

## 📁 Output Example

Processed images will be saved in:

```bash
output_folder/
```

Each image will display:

```
dent (0.89)
W: 5.2 cm  H: 3.1 cm
Area: 16.1 cm²
```

---

# ⚠️ Important Notes

* Real-world measurement depends on:

  * Camera distance
  * Image resolution
* A fixed `pixels_per_cm` is used for approximation
* For accurate results, use a reference object (like ruler/card)


---

# 👨‍💻 Author

**Sanjay L Y**

