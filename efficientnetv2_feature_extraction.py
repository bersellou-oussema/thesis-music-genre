# efficientnetv2_feature_extraction.py
import os, glob, math, joblib
import numpy as np
from tqdm import tqdm
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0, preprocess_input
from tensorflow.keras.models import Model

# ---- Paths ----
DATA_ROOT = "spectrogram_dataset_windowed"
GTZAN_DIR = os.path.join(DATA_ROOT, "gtzan")
FMA_DIR   = os.path.join(DATA_ROOT, "fma")
OUT_DIR   = "models"
os.makedirs(OUT_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 64

def load_image_paths(root_dir):
    paths, labels = [], []
    if not os.path.isdir(root_dir):
        return paths, labels
    for genre in sorted(os.listdir(root_dir)):
        gdir = os.path.join(root_dir, genre)
        if not os.path.isdir(gdir): 
            continue
        files = glob.glob(os.path.join(gdir, "*.png"))
        paths.extend(files)
        labels.extend([genre] * len(files))
    return paths, labels

def make_model():
    base = EfficientNetV2B0(include_top=False, weights="imagenet", pooling="avg")
    return Model(inputs=base.input, outputs=base.output)

def iter_batches(img_paths, labels, batch_size=BATCH_SIZE):
    n = len(img_paths)
    for i in range(0, n, batch_size):
        batch_paths = img_paths[i:i+batch_size]
        x = np.zeros((len(batch_paths), IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
        for j, p in enumerate(batch_paths):
            # Load RGB (PNG may be saved grayscale)
            im = Image.open(p).convert("RGB").resize(IMG_SIZE)
            x[j] = np.array(im, dtype=np.float32)
        x = preprocess_input(x)
        y = labels[i:i+batch_size]
        yield x, y

def extract_and_save(root_dir, out_pkl):
    paths, labels = load_image_paths(root_dir)
    if not paths:
        print(f"⚠️ No images in {root_dir}")
        return
    print(f"📂 {root_dir} -> {len(paths)} images across {len(set(labels))} genres")
    model = make_model()

    feats = np.zeros((len(paths), 1280), dtype=np.float32)
    lbls = []
    idx = 0

    steps = math.ceil(len(paths)/BATCH_SIZE)
    for xb, yb in tqdm(iter_batches(paths, labels), total=steps, desc=f"Extracting {os.path.basename(root_dir)}"):
        f = model.predict(xb, verbose=0)
        feats[idx:idx+len(yb)] = f
        lbls.extend(yb)
        idx += len(yb)

    joblib.dump((feats, np.array(lbls), np.array(paths)), out_pkl)
    print(f"✅ Saved: {out_pkl}  |  X:{feats.shape}  labels:{len(lbls)}")

if __name__ == "__main__":
    #extract_and_save(GTZAN_DIR, os.path.join(OUT_DIR, "features_effnet_gtzan.pkl"))
    extract_and_save(FMA_DIR,   os.path.join(OUT_DIR, "features_effnet_fma.pkl"))
