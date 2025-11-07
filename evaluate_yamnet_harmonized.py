# evaluate_yamnet_harmonized.py
import os, json
from pathlib import Path
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

SR = 16000           # YAMNet sample rate
EMB_DIM = 1024       # YAMNet frame embedding size
POOL_DIM = EMB_DIM * 2  # mean + std pooling -> 2048

# ---------------- Utils ----------------
def collect_unique_tracks(split_root: str):
    """Return dict track_id -> genre. One entry per ORIGINAL track (not per window)."""
    track_to_genre = {}
    root = Path(split_root)
    if not root.exists():
        print(f"[ERROR] test_root does not exist: {root.resolve()}")
        return track_to_genre
    for genre_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        genre = genre_dir.name
        for f in genre_dir.glob("*.npy"):
            # example: blues.00000__9a1b4bc6fe.npy  ->  core = 'blues.00000'
            core = f.stem.split("__")[0]
            tid = f"{genre}/{core}"
            track_to_genre[tid] = genre
    return track_to_genre

def find_audio_for_track(audio_root: str, track_id: str) -> str:
    """
    Try several patterns to find the raw audio file for a GTZAN track.
    track_id is 'genre/blues.00097'
    """
    from glob import glob
    genre, name = track_id.split("/", 1)   # genre='blues', name='blues.00097'
    base = Path(audio_root)

    # 1) Common pattern: audio_root/genre/genre.00097.ext
    candidates = []
    for ext in (".wav",".mp3",".flac",".au",".ogg",".m4a",".WAV",".MP3",".FLAC",".AU",".OGG",".M4A"):
        p = base / genre / (name + ext)
        candidates.append(p)

    # 2) Underscore style: audio_root/genre/blues_00097.ext
    if "." in name:
        left, right = name.split(".", 1)   # 'blues', '00097'
        for ext in (".wav",".mp3",".flac",".au",".ogg",".m4a",".WAV",".MP3",".FLAC",".AU",".OGG",".M4A"):
            candidates.append(base / genre / f"{left}_{right}{ext}")
            candidates.append(base / genre / f"{right}{ext}")  # just 00097.ext

    # 3) One extra nesting level (e.g., audio_root/gtzan/genre/...)
    for ext in (".wav",".mp3",".flac",".au",".ogg",".m4a",".WAV",".MP3",".FLAC",".AU",".OGG",".M4A"):
        candidates.append(base / "gtzan" / genre / (name + ext))

    # 4) As a last resort, glob within genre folder for anything containing the number
    num_part = name.split(".")[-1]
    for pattern in [f"*{name}.*", f"*{num_part}.*", f"*{num_part}*.*"]:
        candidates += [Path(p) for p in glob(str((base/genre/pattern)))]

    for p in candidates:
        if p.exists():
            return str(p)
    return ""


def load_wave(path: str) -> np.ndarray:
    y, sr = librosa.load(path, sr=SR, mono=True)
    if y.size == 0:
        return np.zeros(SR, dtype=np.float32)
    return y.astype(np.float32)

def yamnet_embed(wave: np.ndarray, yamnet_layer) -> np.ndarray:
    """Return [frames, 1024] embeddings from 1-D float32 waveform."""
    wave_tf = tf.convert_to_tensor(wave, dtype=tf.float32)  # shape (samples,)
    scores, embeddings, _ = yamnet_layer(wave_tf)
    emb = embeddings.numpy()
    if emb.ndim == 3:
        emb = emb.squeeze(0)
    return emb

def pool_mean_std(emb: np.ndarray) -> np.ndarray:
    """Temporal mean+std pooling -> [2048]."""
    if emb.ndim != 2 or emb.shape[0] == 0:
        return np.zeros(POOL_DIM, dtype=np.float32)
    mu = emb.mean(axis=0)
    sd = emb.std(axis=0)
    return np.concatenate([mu, sd], axis=0).astype(np.float32)

def build_or_load_embeddings(split_root: str, audio_root: str, cache_root: str,
                             label_map: dict, yamnet_layer):
    """
    Build per-track pooled YAMNet embeddings for the split, or load from cache.
    Returns X:[N,2048], y:[N]
    """
    track_to_genre = collect_unique_tracks(split_root)
    if not track_to_genre:
        print("[ERROR] No .npy window files found under test_root (per-genre subfolders).")
        return np.zeros((0, POOL_DIM), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    X, y = [], []
    missing_audio = 0

    items = sorted(track_to_genre.items())
    total = len(items)
    print(f"[INFO] Found {total} unique test tracks to evaluate.")

    for k, (tid, genre) in enumerate(items, 1):
        cache_path = cache_root / (tid.replace("/", "_") + ".npy")
        if cache_path.exists():
            pooled = np.load(cache_path)
        else:
            audio_path = find_audio_for_track(audio_root, tid)
            if not audio_path:
                missing_audio += 1
                if k % 10 == 0 or k == total:
                    print(f"[WARN] [{k}/{total}] Missing audio for {tid}")
                continue
            wave = load_wave(audio_path)
            emb = yamnet_embed(wave, yamnet_layer)       # [frames, 1024]
            pooled = pool_mean_std(emb)                  # [2048]
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, pooled)

        X.append(pooled)
        y.append(label_map[genre])

        if k % 10 == 0 or k == total:
            print(f"[INFO] Embedded {k}/{total} tracks")

    if missing_audio:
        print(f"[WARN] Missing audio for {missing_audio} tracks (skipped).")

    if not X:
        print("[ERROR] No embeddings were produced. Check audio_root path.")
        return np.zeros((0, POOL_DIM), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"[INFO] Built/loaded embeddings: X={X.shape}, y={y.shape}")
    return X, y

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_root", required=True, help="spectrogram_dataset_harmonized/gtzan_split/test")
    ap.add_argument("--audio_root", required=True, help="raw audio root containing genre subfolders (e.g., data_wav)")
    ap.add_argument("--cache_root", required=True, help="cache dir for YAMNet test embeddings (e.g., yamnet_cache/gtzan/test)")
    ap.add_argument("--model", required=True, help="path to classifier .keras model")
    ap.add_argument("--label_map", required=True, help="path to label_map.json saved during training")
    ap.add_argument("--out_cm", default="confusion_matrix_yamnet.png")
    args = ap.parse_args()

    print("[DEBUG] evaluate_yamnet_harmonized.py starting…")

    # Load label map
    with open(args.label_map, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    inv = {v: k for k, v in label_map.items()}
    class_names = [inv[i] for i in range(len(inv))]

    # Load YAMNet hub layer
    print("[INFO] Loading YAMNet hub…")
    yamnet_layer = hub.KerasLayer("https://tfhub.dev/google/yamnet/1", trainable=False)

    # Build/load test embeddings
    print("[INFO] Building/Loading test embeddings…")
    X_test, y_test = build_or_load_embeddings(args.test_root, args.audio_root, args.cache_root,
                                              label_map, yamnet_layer)
    if X_test.shape[0] == 0:
        print("[FATAL] No test data to evaluate.")
        return

    # Load classifier
    print("[INFO] Loading classifier…")
    model = tf.keras.models.load_model(args.model)

    # Predict
    print("[INFO] Predicting…")
    y_prob = model.predict(X_test, batch_size=32, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    # Reports
    print("\n=== Classification report ===")
    print(classification_report(y_test, y_pred, target_names=class_names, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix — YAMNet Harmonized (Track-level)")
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(args.out_cm)
    print(f"[INFO] Confusion matrix saved to {args.out_cm}")

    acc = (y_pred == y_test).mean()
    print(f"\nFINAL TEST ACCURACY: {acc*100:.2f}% on {len(y_test)} tracks")

if __name__ == "__main__":
    main()
