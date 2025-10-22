"""This auto-labeler code attempts to automatically fit duck + duck on side labels
to a set of unlabled images by fine tuning YOLOv8, and classifying the images."""

import os
import glob
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from PIL import Image

# ==== CONFIG ====
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
BASE_DIR = "data_bg"      # directory containing all images + CSVs
IMG_SIZE = 256
EPOCHS = 10
CONF_THRESH = 0.7
MODEL_NAME = "yolov8n.pt"    # or yolov8s.pt for slightly more accuracy

def prepare_dataset(base_dir):
    """Separate labeled and unlabeled images."""
    base = Path(base_dir)
    labeled, unlabeled = [], []

    for img_path in base.glob("**/*"):
        if img_path.suffix.lower() in IMAGE_EXTS:
            csv_path = img_path.with_suffix(".csv")
            if csv_path.exists():
                labeled.append(img_path)
            else:
                unlabeled.append(img_path)

    print(f"Found {len(labeled)} labeled and {len(unlabeled)} unlabeled images.")
    return labeled, unlabeled


def make_yolo_dataset(labeled_imgs, out_dir):
    """Convert CSV annotations to YOLO txt format."""
    out_dir = Path(out_dir)
    labels_dir = out_dir / "labels"
    images_dir = out_dir / "images"
    labels_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(labeled_imgs, desc="Converting CSVs"):
        csv_path = img_path.with_suffix(".csv")
        df = pd.read_csv(csv_path)

        # Load image to get actual dimensions
        with Image.open(img_path) as im:
            w, h = im.size

        label_lines = []
        for _, row in df.iterrows():
            x, y, bw, bh, cls_name = row
            cls = 0 if cls_name == "duck" else 1
            xc, yc = (x + bw / 2) / w, (y + bh / 2) / h
            label_lines.append(f"{cls} {xc} {yc} {bw/w} {bh/h}\n")

        with open(labels_dir / (img_path.stem + ".txt"), "w") as f:
            f.writelines(label_lines)

        shutil.copy(img_path, images_dir / img_path.name)

    print("YOLO dataset created at:", out_dir)


def create_duck_yaml(out_path, train_dir):
    yaml_text = f"""
path: {train_dir}
train: images
val: images
names:
  0: duck
  1: duck_on_side
"""
    Path(out_path).write_text(yaml_text.strip())


def train_yolo(train_dir, epochs):
    model = YOLO(MODEL_NAME)
    model.train(data=str(Path(train_dir) / "duck.yaml"), epochs=epochs, imgsz=IMG_SIZE)
    return model


def autolabel_unlabeled(model, unlabeled_imgs, conf_thresh):
    for img_path in tqdm(unlabeled_imgs, desc="Autolabeling"):
        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception:
            print(f"Skipping unreadable image: {img_path}")
            continue

        results = model.predict(source=str(img_path), conf=conf_thresh, save=False, imgsz=IMG_SIZE, verbose=False)
        out_csv = img_path.with_suffix(".csv")

        if not results or len(results[0].boxes) == 0:
            # Write empty CSV for traceability
            pd.DataFrame(columns=["x", "y", "width", "height", "class"]).to_csv(out_csv, index=False)
            continue

        boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()  # [x1,y1,x2,y2]
        classes = results[0].boxes.cls.cpu().numpy()

        # Convert back to pixel-space CSV
        rows = []
        for (x1, y1, x2, y2), c in zip(boxes_xyxy, classes):
            bw, bh = x2 - x1, y2 - y1
            rows.append([x1, y1, bw, bh, "duck" if int(c) == 0 else "duck_on_side"])

        pd.DataFrame(rows, columns=["x", "y", "width", "height", "class"]).to_csv(out_csv, index=False)

    print("Autolabeling complete — CSVs written beside source images.")


if __name__ == "__main__":
    labeled_imgs, unlabeled_imgs = prepare_dataset(BASE_DIR)
    if len(labeled_imgs) == 0:
        raise RuntimeError("No labeled images found. Cannot train.")

    dataset_dir = Path("duck_yolo_dataset")
    make_yolo_dataset(labeled_imgs, dataset_dir)
    create_duck_yaml(dataset_dir / "duck.yaml", dataset_dir)

    model = train_yolo(dataset_dir, EPOCHS)

    if unlabeled_imgs:
        autolabel_unlabeled(model, unlabeled_imgs, CONF_THRESH)
    else:
        print("No unlabeled images found.")
