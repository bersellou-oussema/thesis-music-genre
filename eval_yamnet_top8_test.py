import os, argparse, hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Set
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import librosa
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

SR = 16000
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
TOP8 = ["Electronic","Experimental","Folk","Hip-Hop","Instrumental","International","Pop","Rock"]

os.environ.setdefault("TFHUB_CACHE_DIR", "yamnet_cache/tfhub")
tf.get_logger().setLevel("ERROR")

def log(m): print(m, flush=True)
def sha1(s): return hashlib.sha1(s.encode("utf-8")).hexdigest()

def stems_from_split(root):
    r = Path(root); stems = set()
    for p in r.rglob("*.npy"): stems.add(p.stem)
    for p in r.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            stems.add(p.stem[:6] if p.stem[:6].isdigit() else p.stem)
    return stems

def read_fma_genre_map(tracks_csv):
    df = pd.read_csv(tracks_csv, index_col=0, header=[0,1])
    ser = df[("track","genre_top")].dropna()
    m = {}
    for tid,g in ser.items():
        tid6 = f"{int(tid):06d}"
        g = str(g)
        if g in TOP8: m[tid6]=g
    log(f"[INFO] Loaded {len(m)} TOP-8 track→genre mappings from tracks.csv.")
    return m

def find_audio(global_root: Path, stem: str):
    for p in global_root.rglob(f"{stem}*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            return p
    return None

def collect_pairs(split_root, audio_root, genre_map):
    pairs, used = [], set()
    for stem in sorted(stems_from_split(split_root)):
        g = genre_map.get(stem)
        if g is None or stem in used: continue
        ap = find_audio(Path(audio_root), stem)
        if ap is None: continue
        pairs.append((str(ap), g, stem)); used.add(stem)
    log(f"[INFO] Collected {len(pairs)} unique items from {split_root}.")
    return pairs

def load_audio_16k(path):
    y,_ = librosa.load(path, sr=SR, mono=True)
    return tf.convert_to_tensor(np.asarray(y, np.float32), dtype=tf.float32)

def build_yamnet():
    log("[INFO] Loading YAMNet from TF Hub…")
    lyr = hub.KerasLayer("https://tfhub.dev/google/yamnet/1", trainable=False)
    log("[INFO] YAMNet loaded."); return lyr

def yamnet_embed(wav, layer):
    _, emb, _ = layer(wav)
    return tf.reduce_mean(emb, axis=0).numpy().astype(np.float32)

def embeddings_from_pairs(pairs, cache_dir, label_map, layer):
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    X, y = [], []
    for i,(ap,g,stem) in enumerate(pairs,1):
        out_dir = Path(cache_dir)/g; out_dir.mkdir(parents=True, exist_ok=True)
        cache_file = out_dir/(sha1(ap)+".npy")
        if cache_file.exists(): e = np.load(cache_file)
        else:
            try:
                e = yamnet_embed(load_audio_16k(ap), layer); np.save(cache_file, e)
            except Exception as ex:
                log(f"[WARN] Skipping {ap}: {ex}"); continue
        X.append(e); y.append(label_map[g])
        if i%200==0: log(f"[INFO] Cached {i}/{len(pairs)}")
    return np.stack(X).astype(np.float32), np.asarray(y, np.int32)

def top5_acc(probs, y_true):
    top5 = np.argpartition(-probs, 5, axis=1)[:, :5]
    return float(np.mean([yi in row for yi,row in zip(y_true, top5)]))

def save_confusion_images(cm: np.ndarray, labels: List[str], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    def draw(data, title, fname, fmt_counts=True):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,7))
        plt.imshow(data, interpolation='nearest')
        plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
        plt.yticks(range(len(labels)), labels)
        plt.colorbar(label="count" if fmt_counts else "%")
        plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(title)
        vmax = data.max() if data.size else 1.0
        thresh = vmax/2.0
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i,j]
                txt = f"{int(val)}" if fmt_counts else f"{val:.1f}%"
                plt.text(j, i, txt, ha="center", va="center",
                         color=("white" if val > thresh else "black"), fontsize=9)
        plt.tight_layout(); plt.savefig(out_dir/fname, dpi=180, bbox_inches="tight"); plt.close()

    # counts
    draw(cm, "Confusion Matrix (counts)", "confusion_matrix_test_counts.png", True)
    # row-normalized percentages
    cm_pct = cm.astype(np.float64)
    cm_pct = (cm_pct.T/np.maximum(cm_pct.sum(axis=1),1)).T*100.0
    draw(cm_pct, "Confusion Matrix (%)", "confusion_matrix_test_percent.png", False)
    # CSVs too
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(out_dir/"confusion_matrix_test_counts.csv")
    pd.DataFrame(np.round(cm_pct,1), index=labels, columns=labels).to_csv(out_dir/"confusion_matrix_test_percent.csv")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", required=True)
    ap.add_argument("--val_root",   required=True)
    ap.add_argument("--test_root",  required=True)
    ap.add_argument("--audio_root", required=True)
    ap.add_argument("--tracks_csv", required=True)
    ap.add_argument("--cache_root", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--labels_path", required=True)
    args = ap.parse_args()

    # Leak check
    tr, va, te = stems_from_split(args.train_root), stems_from_split(args.val_root), stems_from_split(args.test_root)
    log(f"[LEAK] train∩val={len(tr&va)}  train∩test={len(tr&te)}  val∩test={len(va&te)}")

    # labels from training
    labels, label_map = [], {}
    with open(args.labels_path, "r", encoding="utf-8") as f:
        for line in f:
            idx, name = line.strip().split("\t")
            labels.append(name); label_map[name]=int(idx)
    log(f"[INFO] Label order: {labels}")

    # collect test pairs
    genre_map = read_fma_genre_map(args.tracks_csv)
    pairs = collect_pairs(args.test_root, args.audio_root, genre_map)
    if not pairs: raise RuntimeError("No test items found.")

    layer = build_yamnet()
    Xte, yte = embeddings_from_pairs(pairs, str(Path(args.cache_root)/"test"), label_map, layer)

    clf = tf.keras.models.load_model(args.model_path)
    probs = clf.predict(Xte, verbose=0)
    yhat = np.argmax(probs, axis=1)

    top1, top5 = float(np.mean(yhat==yte)), top5_acc(probs, yte)
    log(f"[RESULT][TEST] top1: {top1:.4f} | top5: {top5:.4f}")

    # print classification report
    print("\n===== Classification Report (TEST) =====\n")
    print(classification_report(yte, yhat, target_names=labels, digits=4))

    # print matrices
    cm = confusion_matrix(yte, yhat, labels=list(range(len(labels))))
    print("\n===== Confusion Matrix (counts) =====\n")
    print(pd.DataFrame(cm, index=labels, columns=labels).to_string())

    cm_pct = (cm.astype(np.float64).T/np.maximum(cm.sum(axis=1),1)).T*100.0
    print("\n===== Confusion Matrix (%) — rows sum to 100 =====\n")
    print(pd.DataFrame(np.round(cm_pct,1), index=labels, columns=labels).to_string())

    # save images with numbers (like your previous experiments)
    save_confusion_images(cm, labels, Path(args.cache_root))
    log(f"[INFO] Saved images to: {args.cache_root}\\confusion_matrix_test_counts.png "
        f"and ..._percent.png")

if __name__ == "__main__":
    main()
