# 🚗 Car Dent Detection & Measurement using YOLO

## 📌 Overview

This project focuses on detecting **car dents** using a YOLO model and estimating their **width, height, and area** from images. It includes the complete pipeline from data labeling → training → testing.

---

sample_img_test.py file will test the image by detect and measures the real world  using coc dataset pretrained model
webcan.py file will test using webcam by detect and measures the real world  using coc dataset pretrained model
custom_dent_test file uses is best.pt which is trained on some images and it is tested for dent detection and measures the real world values

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
Outputs
<img width="1024" height="1024" alt="33" src="https://github.com/user-attachments/assets/04b37408-167c-40b6-b8cb-ebb80368b449" />

<img width="1024" height="1024" alt="60" src="https://github.com/user-attachments/assets/abbc1344-303f-4a1b-91aa-7661b0346d6a" />

<img width="1024" height="1024" alt="6" src="https://github.com/user-attachments/assets/a3f41978-532f-47c9-80aa-36791950955a" />
<img width="900" height="1600" alt="output" src="https://github.com/user-attachments/assets/f9c246a9-9c26-4fb0-824a-8b48a23b0504" />
<img width="637" height="512" alt="Screenshot 2026-06-23 234131" src="https://github.com/user-attachments/assets/74aecfc6-04a6-40b0-a17b-c06b8e53a5e8" />
<img width="639" height="511" alt="Screenshot 2026-06-23 234241" src="https://github.com/user-attachments/assets/efd0df10-4782-474d-b1ce-96ce21bd219f" />
