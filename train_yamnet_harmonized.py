import os, glob, json, math, random
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
from tqdm import tqdm

from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.optimizers import AdamW

# -----------------------------
# Config
# -----------------------------
SR = 16000                 # YAMNet expects 16 kHz mono
EMB_DIM = 1024             # YAMNet embedding size
PATIENCE = 10
LABEL_SMOOTH = 0.05
WEIGHT_DECAY = 1e-4
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# -----------------------------
# Utils
# -----------------------------
def list_label_names(root_dir: str) -> List[str]:
    return sorted([d.name for d in Path(root_dir).iterdir() if d.is_dir()])

def collect_unique_tracks(split_root: str) -> Dict[str, str]:
    """
    Parse track IDs from harmonized .npy filenames.
    Returns: { track_id: genre } where track_id like 'blues/blues.00000'
    """
    track_to_genre = {}
    for genre_dir in sorted([p for p in Path(split_root).iterdir() if p.is_dir()]):
        genre = genre_dir.name
        for f in genre_dir.glob("*.npy"):
            base = f.stem                     # e.g., blues.00000__abcd1234
            core = base.split("__")[0]        # e.g., blues.00000
            tid = f"{genre}/{core}"
            track_to_genre[tid] = genre
    return track_to_genre

def find_audio_for_track(audio_root: str, track_id: str) -> str:
    """
    Given audio_root and 'genre/trackname', try common extensions and return path.
    """
    genre, name = track_id.split("/", 1)
    stem = Path(audio_root) / genre / name
    for ext in (".wav", ".mp3", ".flac", ".au", ".ogg", ".m4a"):
        cand = str(stem) + ext
        if Path(cand).exists():
            return cand
    # also try lowercase/uppercase quirks
    for ext in (".WAV", ".MP3", ".FLAC", ".AU", ".OGG", ".M4A"):
        cand = str(stem) + ext
        if Path(cand).exists():
            return cand
    return ""  # not found

def load_wave_mono16k(path: str) -> np.ndarray:
    y, sr = librosa.load(path, sr=SR, mono=True)
    # ensure float32
    if y.size == 0:
        return np.zeros(SR, dtype=np.float32)
    return y.astype(np.float32)

def yamnet_embed(wave: np.ndarray, yamnet_model) -> np.ndarray:
    """
    Run YAMNet and return [frames, 1024] embeddings.
    TF-Hub YAMNet expects a 1-D float32 waveform at 16 kHz.
    """
    wave_tf = tf.convert_to_tensor(wave, dtype=tf.float32)  # 1-D, shape (samples,)
    scores, embeddings, _ = yamnet_model(wave_tf)
    emb = embeddings.numpy()
    # Handle either [frames, 1024] or [1, frames, 1024] depending on hub version
    if emb.ndim == 3:
        emb = emb.squeeze(0)
    return emb


def pool_track_embeddings(emb: np.ndarray) -> np.ndarray:
    """
    emb: [frames, 1024]; return mean+std (2048-D)
    """
    if emb.ndim != 2 or emb.shape[0] == 0:
        # fallback if something is odd
        return np.zeros(EMB_DIM*2, dtype=np.float32)
    mu = emb.mean(axis=0)
    sd = emb.std(axis=0)
    return np.concatenate([mu, sd], axis=0).astype(np.float32)

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

# -----------------------------
# Dataset build
# -----------------------------
def build_dataset(split_root: str, audio_root: str, cache_root: str,
                  label_map: Dict[str, int], yamnet_model) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each unique track in split_root, compute (or load) pooled YAMNet embedding and label index.
    Returns X: [N, 2048], y: [N] (int labels)
    """
    ensure_dir(cache_root)
    track_to_genre = collect_unique_tracks(split_root)

    X, y = [], []
    missing = 0

    for tid, genre in tqdm(track_to_genre.items(), desc=f"Embedding {Path(split_root).name} tracks"):
        cache_path = Path(cache_root) / (tid.replace("/", "_") + ".npy")
        if cache_path.exists():
            pooled = np.load(cache_path)
        else:
            audio_path = find_audio_for_track(audio_root, tid)
            if not audio_path:
                missing += 1
                continue
            wave = load_wave_mono16k(audio_path)
            emb = yamnet_embed(wave, yamnet_model)     # [frames, 1024]
            pooled = pool_track_embeddings(emb)        # [2048]
            ensure_dir(cache_path.parent.as_posix())
            np.save(cache_path, pooled)

        X.append(pooled)
        y.append(label_map[genre])

    if missing:
        print(f"[WARN] Missing audio for {missing} tracks under {audio_root} — skipped.")

    X = np.stack(X).astype(np.float32) if len(X) else np.zeros((0, EMB_DIM*2), dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y

# -----------------------------
# Model (MLP head on YAMNet embeddings)
# -----------------------------
def build_head(n_classes: int, input_dim: int = EMB_DIM*2, dropout: float = 0.4):
    inp = layers.Input(shape=(input_dim,), dtype=tf.float32)
    x = layers.BatchNormalization()(inp)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return models.Model(inp, out)

# -----------------------------
# Entry
# -----------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", required=True, help="e.g., spectrogram_dataset_harmonized/gtzan_split/train")
    ap.add_argument("--val_root",   required=True, help="e.g., spectrogram_dataset_harmonized/gtzan_split/val")
    ap.add_argument("--audio_root", required=True, help="Folder with raw GTZAN audio, e.g., data_wav")
    ap.add_argument("--cache_root", default="yamnet_cache/gtzan", help="Where to cache per-track embeddings")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr",     type=float, default=3e-4)
    ap.add_argument("--batch",  type=int, default=32)
    args = ap.parse_args()

    # Labels
    label_names = list_label_names(args.train_root)
    label_map = {n:i for i, n in enumerate(label_names)}
    print("Classes:", label_names)

    # Load YAMNet (TF Hub) — requires internet the first time
    print("Loading YAMNet from TF Hub…")
    yamnet = hub.KerasLayer("https://tfhub.dev/google/yamnet/1", trainable=False)

    # Build (or load cached) embeddings
    train_cache = Path(args.cache_root) / "train"
    val_cache   = Path(args.cache_root) / "val"
    Xtr, ytr = build_dataset(args.train_root, args.audio_root, str(train_cache), label_map, yamnet)
    Xva, yva = build_dataset(args.val_root,   args.audio_root, str(val_cache),   label_map, yamnet)

    print("Train shape:", Xtr.shape, "Val shape:", Xva.shape)

    # One-hot
    ytr_oh = tf.keras.utils.to_categorical(ytr, num_classes=len(label_map)).astype("float32")
    yva_oh = tf.keras.utils.to_categorical(yva, num_classes=len(label_map)).astype("float32")
    # Head model
    model = build_head(n_classes=len(label_map))
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTH)
    opt  = AdamW(learning_rate=args.lr, weight_decay=WEIGHT_DECAY)

    # Output dirs & callbacks
    out_dir = Path("models/Deep Learning/YAMNet")
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path  = out_dir / "yamnet_harmonized_best.keras"
    final_path = out_dir / "yamnet_harmonized_final.h5"

    cbs = [
        callbacks.ModelCheckpoint(best_path.as_posix(), monitor="val_accuracy", save_best_only=True, verbose=1),
        callbacks.EarlyStopping(monitor="val_accuracy", patience=PATIENCE, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=1),
    ]

    # Train
    model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
    model.fit(
        Xtr, ytr_oh,
        validation_data=(Xva, yva_oh),
        epochs=args.epochs,
        batch_size=args.batch,
        callbacks=cbs,
        verbose=1
    )

    # Save final + label map
    model.save(final_path.as_posix())
    with open(out_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)

    print("Saved:", final_path)
    print("Saved best:", best_path)
    print("Saved label map:", (out_dir / "label_map.json").as_posix())

if __name__ == "__main__":
    main()
