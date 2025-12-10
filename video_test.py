import json
from pathlib import Path
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
import sys
import time

MODEL_PATH = Path("saved_models") / "cifar10_ResNet  20v  1_model.200.keras"
TRAIN_ANN = Path("dashcam 2.v1i.coco") / "train" / "_annotations.coco.json"
IMG_SIZE = (32, 32)  # model input size

def load_label_encoder(train_ann_path):
    with open(train_ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    categories = [c["name"] for c in coco.get("categories", [])]
    return LabelEncoder().fit(categories)

def preprocess_frame(frame):
    # frame: BGR from cv2
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb, IMG_SIZE)
    arr = small.astype("float32") / 255.0
    return np.expand_dims(arr, 0)

def main(cam_index=0):
    if not MODEL_PATH.exists():
        print("Model not found:", MODEL_PATH); sys.exit(1)
    if not TRAIN_ANN.exists():
        print("Train annotations not found:", TRAIN_ANN); sys.exit(1)

    model = load_model(str(MODEL_PATH))
    le = load_label_encoder(TRAIN_ANN)

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("Cannot open camera"); sys.exit(1)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame"); break

            x = preprocess_frame(frame)
            probs = model.predict(x, verbose=0)
            idx = int(np.argmax(probs, axis=1)[0])
            prob = float(probs[0, idx])
            class_name = le.classes_[idx] if idx < len(le.classes_) else str(idx)

            # terminal output
            print(f"[{time.time():.2f}] Pred: {class_name} (idx={idx})  P={prob:.4f}", flush=True)

            # overlay on frame
            label = f"{class_name}: {prob:.2f}"
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("Webcam - press q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    main(idx)