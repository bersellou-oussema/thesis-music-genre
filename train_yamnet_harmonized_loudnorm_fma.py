# filename: train_yamnet_harmonized_loudnorm_fma.py
import os, sys, time, argparse, hashlib, csv
from pathlib import Path
from typing import Dict, List, Tuple, Set

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import pandas as pd  # requires: pip install pandas

SR = 16000
EMB_DIM = 1024

# Verbose TF + TF-Hub cache
os.environ.setdefault("TFHUB_CACHE_DIR", "yamnet_cache/tfhub")
tf.get_logger().setLevel("INFO")

TOP8 = ["Electronic", "Experimental", "Folk", "Hip-Hop",
        "Instrumental", "International", "Pop", "Rock"]

def log(m: str): print(m, flush=True)
def sha1(s: str) -> str: return hashlib.sha1(s.encode("utf-8")).hexdigest()

def load_audio_16k(path: str, sr: int = SR) -> tf.Tensor:
    y, _ = librosa.load(path, sr=sr, mono=True)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    return tf.convert_to_tensor(y, dtype=tf.float32)

def ensure_dir(p: str): Path(p).mkdir(parents=True, exist_ok=True)

def _find_audio_in_dir(base: Path, stem: str, exts: List[str]) -> Path | None:
    # exact stem.ext
    for e in exts:
        p = base / f"{stem}{e}"
        if p.exists(): return p
    # stem_*.ext
    for e in exts:
        for p in base.glob(f"{stem}_*{e}"):
            if p.is_file(): return p
    # any file starting with stem
    for p in base.glob(f"{stem}*"):
        if p.is_file() and p.suffix.lower() in exts: return p
    return None

def _find_audio_global(root: Path, stem: str, exts: List[str]) -> Path | None:
    for p in root.rglob(f"{stem}*"):
        if p.is_file() and p.suffix.lower() in exts:
            return p
    return None

def read_fma_genre_map(tracks_csv: str) -> Dict[str, str]:
    """Return {six_digit_track_id: genre_top} for TOP8 only."""
    if not Path(tracks_csv).exists():
        raise FileNotFoundError(f"tracks.csv not found at: {tracks_csv}")
    # tracks.csv has a 2-level header; use pandas to read it
    df = pd.read_csv(tracks_csv, index_col=0, header=[0, 1])
    series = df[("track", "genre_top")].dropna()
    mapping: Dict[str, str] = {}
    for tid, g in series.items():
        gid = f"{int(tid):06d}"
        g = str(g)
        if g in TOP8:
            mapping[gid] = g
    log(f"[INFO] Loaded {len(mapping)} track→genre mappings (TOP-8 only).")
    return mapping

def collect_pairs(split_root: str,
                  audio_root: str,
                  genre_map: Dict[str, str]) -> List[Tuple[str, str, str]]:
    """
    Collect (audio_path, genre_name, stem) for this split.
    Deduplicate by stem so each track is used once.
    Works when split folders contain .npy tiles OR audio.
    """
    audio_exts = [".wav", ".flac", ".mp3", ".ogg", ".m4a"]
    split_root = str(split_root)
    pairs: List[Tuple[str, str, str]] = []
    used: Set[str] = set()
    global_root = Path(audio_root)

    # iterate class folders (these are NOT real genres in your current split)
    for lbl_dir in sorted([d for d in Path(split_root).iterdir() if d.is_dir()]):
        # case A: split has audio already
        audio_files = [p for p in lbl_dir.rglob("*")
                       if p.is_file() and p.suffix.lower() in audio_exts]
        if audio_files:
            for p in audio_files:
                stem = p.stem[:6] if p.stem[:6].isdigit() else p.stem
                if stem in used: continue
                genre = genre_map.get(stem)
                if genre is None: continue
                pairs.append((str(p), genre, stem))
                used.add(stem)
            continue

        # case B: split has .npy tiles -> map tile stems to audio via metadata
        stems = sorted({p.stem for p in lbl_dir.rglob("*.npy")})
        for stem in stems:
            # from tile stem like "000197" → genre via tracks.csv
            if stem not in genre_map:  # not in TOP8 or missing metadata
                continue
            # find audio: try label-mirroring dir first, then global search
            label_audio_dir = Path(audio_root) / lbl_dir.name
            hit = None
            if label_audio_dir.exists():
                hit = _find_audio_in_dir(label_audio_dir, stem, audio_exts)
            if hit is None:
                hit = _find_audio_global(global_root, stem, audio_exts)
            if hit is None:  # audio truly missing
                continue
            if stem in used:  # dedupe
                continue
            pairs.append((str(hit), genre_map[stem], stem))
            used.add(stem)

    log(f"[INFO] Collected {len(pairs)} unique tracks from {split_root}.")
    return pairs

def make_label_map(pairs_tr: List[Tuple[str, str, str]],
                   pairs_va: List[Tuple[str, str, str]]) -> Dict[str, int]:
    present = {g for _, g, _ in pairs_tr} | {g for _, g, _ in pairs_va}
    ordered = [g for g in TOP8 if g in present]
    if not ordered:
        raise RuntimeError("No TOP-8 genres found in the provided splits.")
    label_map = {g: i for i, g in enumerate(ordered)}
    log(f"[INFO] Using labels ({len(label_map)}): {ordered}")
    return label_map

def build_yamnet_layer() -> hub.KerasLayer:
    log("[INFO] Loading YAMNet from TF Hub…")
    layer = hub.KerasLayer("https://tfhub.dev/google/yamnet/1", trainable=False)
    log("[INFO] YAMNet loaded.")
    return layer

def yamnet_embed(waveform: tf.Tensor, yamnet_layer: hub.KerasLayer) -> np.ndarray:
    _, embeddings, _ = yamnet_layer(waveform)     # [frames, 1024]
    emb = tf.reduce_mean(embeddings, axis=0)      # [1024]
    return emb.numpy().astype(np.float32)

def extract_or_load_from_pairs(pairs: List[Tuple[str, str, str]],
                               cache_root: str,
                               label_map: Dict[str, int],
                               yamnet_layer: hub.KerasLayer) -> Tuple[np.ndarray, np.ndarray]:
    ensure_dir(cache_root)
    X_list, y_idx = [], []
    for i, (apath, genre, stem) in enumerate(pairs, 1):
        cls = label_map[genre]
        out_dir = Path(cache_root) / genre
        ensure_dir(out_dir)
        cache_file = out_dir / (sha1(apath) + ".npy")

        if cache_file.exists():
            emb = np.load(cache_file)
        else:
            try:
                wav = load_audio_16k(apath)
                emb = yamnet_embed(wav, yamnet_layer)
                np.save(cache_file, emb)
            except Exception as e:
                log(f"[WARN] Skip {apath} due to: {e}")
                continue

        X_list.append(emb)
        y_idx.append(cls)
        if i % 200 == 0:
            log(f"[INFO] Processed {i}/{len(pairs)} files…")

    if not X_list:
        raise RuntimeError("No embeddings created (all failed?).")

    X = np.stack(X_list).astype(np.float32)
    y_idx = np.asarray(y_idx, dtype=np.int32)
    y = tf.one_hot(y_idx, depth=len(label_map), dtype=tf.float32).numpy()
    return X, y

def class_weights_from_y_idx(y_idx: np.ndarray, n_classes: int) -> Dict[int, float]:
    counts = np.bincount(y_idx, minlength=n_classes).astype(np.float64)
    total = counts.sum()
    counts[counts == 0] = 1.0
    weights = total / (n_classes * counts)
    return {i: float(w) for i, w in enumerate(weights)}

def build_classifier(n_classes: int, lr: float) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(EMB_DIM,), name="yamnet_embedding")
    x = tf.keras.layers.BatchNormalization()(inp)
    x = tf.keras.layers.Dense(512, activation="relu",
                              kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256, activation="relu",
                              kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    out = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs=inp, outputs=out, name="YAMNet_Classifier_TOP8")
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt,
                  loss="categorical_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5")])
    return model

def main():
    ap = argparse.ArgumentParser("YAMNet + TOP-8 genre classification (FMA)")
    ap.add_argument("--train_root", required=True, help="Split root with .npy tiles or audio (TRAIN).")
    ap.add_argument("--val_root",   required=True, help="Split root with .npy tiles or audio (VAL).")
    ap.add_argument("--audio_root", required=True, help="Folder that actually holds audio (any layout).")
    ap.add_argument("--tracks_csv", required=True, help="Path to FMA metadata tracks.csv")
    ap.add_argument("--cache_root", required=True, help="Where to cache embeddings & save artifacts.")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr",     type=float, default=3e-4)
    ap.add_argument("--batch",  type=int, default=32)
    ap.add_argument("--patience", type=int, default=8)
    args = ap.parse_args()

    # TF-Hub cache under cache_root
    hub_cache = Path(args.cache_root) / "tfhub"
    os.environ["TFHUB_CACHE_DIR"] = str(hub_cache)
    ensure_dir(hub_cache)

    # 1) Metadata → track_id(6d) → TOP-8 genre
    genre_map = read_fma_genre_map(args.tracks_csv)

    # 2) Collect pairs (mapped to TOP-8) for train/val
    tr_pairs = collect_pairs(args.train_root, args.audio_root, genre_map)
    va_pairs = collect_pairs(args.val_root,   args.audio_root, genre_map)
    if len(tr_pairs) == 0 or len(va_pairs) == 0:
        raise RuntimeError("No usable tracks found in one of the splits (after TOP-8 mapping).")

    # 3) Label map (only genres present)
    label_map = make_label_map(tr_pairs, va_pairs)

    # 4) Stats
    def counts_by_genre(pairs):
        d: Dict[str, int] = {}
        for _, g, _ in pairs: d[g] = d.get(g, 0) + 1
        return d
    log(f"[INFO] Train counts: {counts_by_genre(tr_pairs)}")
    log(f"[INFO]  Val  counts: {counts_by_genre(va_pairs)}")

    # 5) Model + embeddings
    yamnet_layer = build_yamnet_layer()

    Xtr, Ytr = extract_or_load_from_pairs(tr_pairs, str(Path(args.cache_root)/"train"), label_map, yamnet_layer)
    Xva, Yva = extract_or_load_from_pairs(va_pairs, str(Path(args.cache_root)/"val"),   label_map, yamnet_layer)

    # compute class weights from y_idx
    ytr_idx = np.argmax(Ytr, axis=1)
    class_weights = class_weights_from_y_idx(ytr_idx, n_classes=len(label_map))
    log(f"[INFO] Class weights: {class_weights}")

    model = build_classifier(n_classes=len(label_map), lr=args.lr)
    model.summary(print_fn=lambda s: log("[MODEL] " + s))

    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=args.patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(2, args.patience // 2), min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(filepath=str(Path(args.cache_root) / "yamnet_top8_classifier.keras"),
                                           monitor="val_accuracy", save_best_only=True)
    ]

    log("[INFO] Starting training…")
    t0 = time.time()
    hist = model.fit(Xtr, Ytr,
                     validation_data=(Xva, Yva),
                     epochs=args.epochs,
                     batch_size=args.batch,
                     shuffle=True,
                     class_weight=class_weights,
                     callbacks=cbs,
                     verbose=1)
    log(f"[INFO] Training finished in {(time.time()-t0)/60:.1f} minutes.")

    # Final eval
    results = model.evaluate(Xva, Yva, verbose=0, return_dict=True)
    log(f"[RESULT] Val top1: {results['accuracy']:.4f} | Val top5: {results['top5']:.4f} | Val loss: {results['loss']:.4f}")

    # Save artifacts
    np.savez(str(Path(args.cache_root) / "history_top8.npz"),
             **{k: np.array(v) for k, v in hist.history.items()},
             label_order=np.array([k for k,_ in sorted(label_map.items(), key=lambda x:x[1])]))
    with open(Path(args.cache_root) / "labels_top8.txt", "w", encoding="utf-8") as f:
        for g, i in sorted(label_map.items(), key=lambda x: x[1]):
            f.write(f"{i}\t{g}\n")
    log(f"[INFO] Artifacts saved under: {args.cache_root}")
    log("[INFO] Done.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"[FATAL] {e}")
        sys.exit(1)
