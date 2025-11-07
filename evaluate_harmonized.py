import os, glob, json, csv
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import argparse
import matplotlib.pyplot as plt

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
    m = mel_01.mean(axis=1, keepdims=True).astype(np.float32)
    s = mel_01.std(axis=1, keepdims=True).astype(np.float32) + 1e-6
    return (mel_01 - m) / s

def per_image_standardize(img_np):
    m = img_np.mean(dtype=np.float32)
    s = img_np.std(dtype=np.float32) + 1e-6
    return (img_np - m) / s

def simple_track_key(path):
    base = os.path.basename(path)
    tid = base.split("__")[0]
    genre = Path(path).parent.name
    return f"{genre}/{tid}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_root", required=True)
    ap.add_argument("--model", required=True, help="path to .keras or .h5")
    ap.add_argument("--label_map", required=True)
    ap.add_argument("--out_dir", default="models/Deep Learning/Evaluation")
    args = ap.parse_args()

    with open(args.label_map, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    inv_map = {v:k for k,v in label_map.items()}
    names = [inv_map[i] for i in range(len(inv_map))]

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    print("Loading model…")
    model = tf.keras.models.load_model(args.model, compile=False)

    files = glob.glob(os.path.join(args.test_root, "*", "*.npy"))
    print("Test windows:", len(files))

    by_track_probs, by_track_true = {}, {}
    for f in files:
        mel = np.load(f).astype(np.float32)     # [-80, 0] dB
        mel = (mel + 80.0) / 80.0               # -> [0,1]
        mel = cmvn_per_frequency(mel)
        img = np.stack([mel, mel, mel], axis=-1)
        img = center_crop_or_pad_width(img, min(img.shape[1], 1249))
        img = resize_to(img, IMG_H, IMG_W)
        img = per_image_standardize(img)
        p = model.predict(img[None, ...], verbose=0)[0]

        tid = simple_track_key(f)
        if tid not in by_track_probs:
            by_track_probs[tid] = []
            by_track_true[tid]  = label_map[Path(f).parent.name]
        by_track_probs[tid].append(p)

    y_true, y_pred = [], []
    for tid, plist in by_track_probs.items():
        P = np.stack(plist, axis=0).mean(axis=0)
        y_true.append(by_track_true[tid])
        y_pred.append(int(np.argmax(P)))

    y_true = np.array(y_true); y_pred = np.array(y_pred)
    acc = (y_true == y_pred).mean()
    print(f"\n🎯 Track-level test accuracy: {acc:.3f}")

    rep = classification_report(y_true, y_pred, target_names=names, digits=3)
    print("\nClassification report:")
    print(rep)

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    # ---- Save outputs ----
    # report
    rep_path = os.path.join(args.out_dir, "classification_report.txt")
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.3f}\n\n")
        f.write(rep)
    print("Saved:", rep_path)

    # CSV
    csv_path = os.path.join(args.out_dir, "confusion_matrix.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + names)
        for i, row in enumerate(cm):
            w.writerow([names[i]] + list(row))
    print("Saved:", csv_path)

    # PNG heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (track-level)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(names)), names, rotation=45, ha="right")
    plt.yticks(range(len(names)), names)
    plt.tight_layout()
    png_path = os.path.join(args.out_dir, "confusion_matrix.png")
    plt.savefig(png_path, dpi=200)
    plt.close()
    print("Saved:", png_path)

if __name__ == "__main__":
    main()
