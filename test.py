import os
import json
from pathlib import Path
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from keras.models import load_model

# config
MODEL_PATH = Path("resnet-models-single-label") / "resnet_single_label.keras"
DATASET_DIR = Path("../dashcam 2.v1i.coco")
VALID_ANN = DATASET_DIR / "valid" / "_annotations.coco.json"
TRAIN_ANN = DATASET_DIR / "train" / "_annotations.coco.json"
IMG_SIZE = (32, 32)

# load model
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
model = load_model(str(MODEL_PATH))
print("Loaded model:", MODEL_PATH)

# build label encoder from train annotations (replicate training encoder)
with open(TRAIN_ANN, "r", encoding="utf-8") as f:
    train_coco = json.load(f)
train_cat_map = {c["id"]: c["name"] for c in train_coco.get("categories", [])}
train_names = [train_cat_map[ann["category_id"]] for ann in train_coco.get("annotations", [])]
label_encoder = LabelEncoder().fit(train_names)
print("Classes (encoder):", label_encoder.classes_)

# load validation annotations
with open(VALID_ANN, "r", encoding="utf-8") as f:
    valid_coco = json.load(f)

images_info = {img["id"]: img for img in valid_coco.get("images", [])}
# map image_id -> list of annotations
anns_by_image = {}
for ann in valid_coco.get("annotations", []):
    anns_by_image.setdefault(ann["image_id"], []).append(ann)

x_list = []
y_list = []
for img_id, img_info in images_info.items():
    img_path = DATASET_DIR / "valid" / img_info["file_name"]
    if not img_path.exists():
        # try without 'valid' prefix if file paths are absolute/relative differently
        alt = DATASET_DIR / img_info["file_name"]
        if alt.exists():
            img_path = alt
        else:
            print("Missing image:", img_path)
            continue

    im = cv2.imread(str(img_path))
    if im is None:
        print("Failed to read:", img_path)
        continue
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, IMG_SIZE)
    x_list.append(im.astype("float32") / 255.0)

    # choose first annotation's category as label (adapt if multi-label)
    anns = anns_by_image.get(img_id, [])
    if not anns:
        y_list.append(None)
    else:
        cat_name = train_cat_map.get(anns[0]["category_id"], None)
        if cat_name is None:
            # fallback to categories defined in valid file
            valid_cat_map = {c["id"]: c["name"] for c in valid_coco.get("categories", [])}
            cat_name = valid_cat_map.get(anns[0]["category_id"])
        y_list.append(cat_name)

# filter out images without labels
x = []
y = []
for xi, yi in zip(x_list, y_list):
    if yi is None:
        continue
    x.append(xi)
    y.append(yi)
x = np.array(x)
y_names = np.array(y)
y_true = label_encoder.transform(y_names)

print("Test samples:", x.shape[0], "Classes in test:", np.unique(y_names))

# predict
y_pred_probs = model.predict(x, batch_size=32)
y_pred = np.argmax(y_pred_probs, axis=1)

# metrics
acc = accuracy_score(y_true, y_pred)
print(f"Accuracy: {acc:.4f}")
print("Classification report:")
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))
# optional confusion matrix
print("Confusion matrix:")
print(confusion_matrix(y_true, y_pred))

# save results
out_dir = Path("saved_models") / "test_results"
out_dir.mkdir(parents=True, exist_ok=True)
np.save(out_dir / "y_true.npy", y_true)
np.save(out_dir / "y_pred.npy", y_pred)
print("Predictions saved to", out_dir)

import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# Add this to the end of your script to plot the matrix
# ---------------------------------------------------------

# 1. Compute the confusion matrix
cm = confusion_matrix(y_true, y_pred)

# 2. Setup the plot
plt.figure(figsize=(12, 10))
sns.heatmap(cm, 
            annot=True,       # Show numbers in cells
            fmt='d',          # distinct integer format
            cmap='Blues',     # Blue color scheme
            xticklabels=label_encoder.classes_, 
            yticklabels=label_encoder.classes_)

plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45)       # Rotate x labels if they overlap
plt.tight_layout()

# 3. Save and Show
plot_path = out_dir / "confusion_matrix.png"
plt.savefig(plot_path)
print(f"Confusion matrix plot saved to: {plot_path}")

plt.show()