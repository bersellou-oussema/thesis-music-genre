import os, glob, json
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import argparse

IMG_H = 224
IMG_W = 224

def resize_to(img_np, h=IMG_H, w=IMG_W):
    t = tf.convert_to_tensor(img_np[None, ...])
    t = tf.image.resize(t, (h, w), method="bilinear", antialias=True)
    return t.numpy()[0]

def center_crop_or_pad_width(x, target_w):
    h, w, c = x.shape
    if w == target_w: return x
    if w > target_w:
        s = (w - target_w)//2
        return x[:, s:s+target_w, :]
    pad = target_w - w
    l = pad // 2; r = pad - l
    return np.pad(x, ((0,0),(l,r),(0,0)), mode="constant")

def cmvn_per_frequency(mel_01):
    # mel_01: [128, T] in [0,1]
    m = mel_01.mean(axis=1, keepdims=True).astype(np.float32)
    s = mel_01.std(axis=1, keepdims=True).astype(np.float32) + 1e-6
    return (mel_01 - m) / s

def per_image_standardize(img_np):
    m = img_np.mean(dtype=np.float32)
    s = img_np.std(dtype=np.float32) + 1e-6
    return (img_np - m) / s

def simple_track_key(path):
    base = os.path.basename(path)          # e.g., blues.00000__abcd.npy
    tid = base.split("__")[0]              # blues.00000
    genre = Path(path).parent.name         # blues
    return f"{genre}/{tid}"

def load_label_map(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_root", required=True)
    ap.add_argument("--model", default="models/Deep Learning/EfficientNet/efficientnetv2b0_harmonized_best.keras")
    ap.add_argument("--label_map", default="models/Deep Learning/EfficientNet/label_map.json")
    args = ap.parse_args()

    label_map = load_label_map(args.label_map)
    inv_map = {v:k for k,v in label_map.items()}

    print("Loading model…")
    model = tf.keras.models.load_model(args.model, compile=False)

    files = glob.glob(os.path.join(args.test_root, "*", "*.npy"))
    print("Test windows:", len(files))

    # Collect pooled predictions per track
    by_track_probs = {}
    by_track_true  = {}

    for f in files:
        mel = np.load(f).astype(np.float32)     # [128, T] in dB range [-80, 0]
        mel = (mel + 80.0) / 80.0               # -> [0,1]
        mel = cmvn_per_frequency(mel)           # CMVN per frequency
        img = np.stack([mel, mel, mel], axis=-1)          # [128, T, 3]
        img = center_crop_or_pad_width(img, min(img.shape[1], 1249))
        img = resize_to(img, IMG_H, IMG_W)                # -> 224x224
        img = per_image_standardize(img)                  # global z-score
        p = model.predict(img[None, ...], verbose=0)[0]   # probs [C]

        tid = simple_track_key(f)
        if tid not in by_track_probs:
            by_track_probs[tid] = []
            by_track_true[tid]  = label_map[Path(f).parent.name]
        by_track_probs[tid].append(p)

    # Mean-pool per track
    y_true, y_pred = [], []
    for tid, plist in by_track_probs.items():
        P = np.stack(plist, axis=0).mean(axis=0)
        y_true.append(by_track_true[tid])
        y_pred.append(int(np.argmax(P)))

    y_true = np.array(y_true); y_pred = np.array(y_pred)
    acc = (y_true == y_pred).mean()
    print(f"\n🎯 Track-level test accuracy: {acc:.3f}")

    print("\nClassification report:")
    names = [inv_map[i] for i in range(len(inv_map))]
    print(classification_report(y_true, y_pred, target_names=names, digits=3))

    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    main()
