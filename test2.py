import sys
from pathlib import Path
import json
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
import matplotlib.pyplot as plt

MODEL_PATH = Path("saved_models") / "cifar10_ResNet  20v  1_model.200.keras"
TRAIN_ANN = Path("dashcam 2.v1i.coco") / "train" / "_annotations.coco.json"
IMG_SIZE = (32, 32)

def load_label_encoder(train_ann_path):
    with open(train_ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    cat_map = {c["id"]: c["name"] for c in coco.get("categories", [])}
    names = [cat_map.get(ann["category_id"]) for ann in coco.get("annotations", []) if ann.get("category_id") in cat_map]
    le = LabelEncoder().fit(names)
    return le

def preprocess_image(path):
    im = cv2.imread(str(path))
    if im is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, IMG_SIZE)
    im = im.astype("float32") / 255.0
    return np.expand_dims(im, 0)

def main():
    img_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("car-motor.png")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not TRAIN_ANN.exists():
        raise FileNotFoundError(f"Train annotations not found: {TRAIN_ANN}")
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    model = load_model(str(MODEL_PATH))
    le = load_label_encoder(TRAIN_ANN)
    x = preprocess_image(img_path)
    probs = model.predict(x)
    idx = int(np.argmax(probs, axis=1)[0])
    prob = float(probs[0, idx])
    class_name = le.classes_[idx] if idx < len(le.classes_) else str(idx)

    print(f"Image: {img_path}")
    print(f"Predicted: {class_name} (index={idx})  Probability: {prob:.4f}")

    # Show the image with matplotlib and prediction in the title
    img_display = x[0]  # floats in [0,1], matplotlib supports this
    plt.figure(figsize=(4,4))
    plt.imshow(img_display)
    plt.title(f"Pred: {class_name} ({prob:.2f})")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()