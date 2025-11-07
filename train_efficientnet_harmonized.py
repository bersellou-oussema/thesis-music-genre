import os, json, glob, math, random
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.applications import EfficientNetV2B0

# -----------------------------
# Config (can be overridden by CLI)
# -----------------------------
IMG_SIZE = 256
BATCH = 32
NUM_WORKERS = 4
PATIENCE = 8
DROPOUT = 0.40
LABEL_SMOOTH = 0.05
WEIGHT_DECAY = 1e-4
MIXUP_ALPHA = 0.20
MAX_TIME = 1249                            # max time bins in your .npy (seen in audits)

random.seed(42); np.random.seed(42); tf.random.set_seed(42)

# -----------------------------
# Augment: stronger SpecAugment
# -----------------------------
class SpecAugment:
    def __init__(self, n_time_masks=2, n_freq_masks=2, time_mask_frac=0.08, freq_mask_frac=0.15):
        self.n_time_masks = n_time_masks
        self.n_freq_masks = n_freq_masks
        self.time_mask_frac = time_mask_frac
        self.freq_mask_frac = freq_mask_frac

    def __call__(self, mel):
        M = mel.copy()
        H, W = M.shape
        # frequency masks
        F = max(1, int(self.freq_mask_frac * H))
        for _ in range(self.n_freq_masks):
            f = np.random.randint(0, F + 1)
            f0 = np.random.randint(0, max(1, H - f))
            M[f0:f0 + f, :] = 0.0
        # time masks
        T = max(1, int(self.time_mask_frac * W))
        for _ in range(self.n_time_masks):
            t = np.random.randint(0, T + 1)
            t0 = np.random.randint(0, max(1, W - t))
            M[:, t0:t0 + t] = 0.0
        return M

# -----------------------------
# Dataset utils
# -----------------------------
def list_label_names(root_dir):
    return sorted([d.name for d in Path(root_dir).iterdir() if d.is_dir()])

def collect_pairs(root_dir, label_map):
    pairs = []
    for name, idx in label_map.items():
        for f in glob.glob(os.path.join(root_dir, name, "*.npy")):
            pairs.append((f, idx))
    random.shuffle(pairs)
    return pairs

def cmvn_per_frequency(mel_01):
    # row-wise z-score across time axis
    m = mel_01.mean(axis=1, keepdims=True).astype(np.float32)
    s = mel_01.std(axis=1, keepdims=True).astype(np.float32) + 1e-6
    return (mel_01 - m) / s

def per_image_standardize(img_np):
    m = img_np.mean(dtype=np.float32)
    s = img_np.std(dtype=np.float32) + 1e-6
    return (img_np - m) / s

def random_time_crop(mel, min_w=700, max_w=None):
    """Crop a random time window [min_w, max_w] then center pad/crop to that width before resize."""
    H, W = mel.shape
    if max_w is None: max_w = min(W, MAX_TIME)
    target = np.random.randint(min(min_w, W), min(max_w, W) + 1)
    if W > target:
        start = np.random.randint(0, W - target + 1)
        return mel[:, start:start + target]
    return mel

def center_crop_or_pad_width(x, target_w):
    h, w, c = x.shape
    if w == target_w: return x
    if w > target_w:
        s = (w - target_w) // 2
        return x[:, s:s + target_w, :]
    pad = target_w - w
    l = pad // 2; r = pad - l
    return np.pad(x, ((0, 0), (l, r), (0, 0)), mode="constant")

def resize_to(img_np, size=IMG_SIZE):
    t = tf.convert_to_tensor(img_np[None, ...])
    t = tf.image.resize(t, (size, size), method="bilinear", antialias=True)
    return t.numpy()[0]

# -----------------------------
# MixUp
# -----------------------------
def mixup(X, y, alpha=MIXUP_ALPHA):
    if alpha <= 0: return X, y
    lam = np.random.beta(alpha, alpha, size=(len(X), 1, 1, 1)).astype(np.float32)
    lam_y = lam[:, 0, 0, 0][:, None]
    idx = np.random.permutation(len(X))
    X2 = lam * X + (1 - lam) * X[idx]
    y2 = lam_y * y + (1 - lam_y) * y[idx]
    return X2.astype(np.float32), y2.astype(np.float32)

# -----------------------------
# Sequence
# -----------------------------
class NpyMelSequence(tf.keras.utils.Sequence):
    def __init__(self, pairs, batch_size, n_classes, training=False, img_size=IMG_SIZE):
        self.pairs = pairs
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.training = training
        self.img_size = img_size
        self.indexes = np.arange(len(self.pairs))
        self.aug = SpecAugment() if training else None
        self.on_epoch_end()

    def __len__(self): return math.ceil(len(self.pairs) / self.batch_size)

    def __getitem__(self, idx):
        ids = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        X, y = [], []
        for i in ids:
            fpath, label = self.pairs[i]
            mel = np.load(fpath).astype(np.float32)         # [128, W] in dB (-80..0)
            mel = (mel + 80.0) / 80.0                       # -> [0,1]
            mel = cmvn_per_frequency(mel)
            if self.training:
                mel = random_time_crop(mel, min_w=700, max_w=MAX_TIME)
                if self.aug: mel = self.aug(mel)
            # stack to 3 channels
            img = np.stack([mel, mel, mel], axis=-1)        # [128, w, 3]
            img = center_crop_or_pad_width(img, min(img.shape[1], MAX_TIME))
            img = resize_to(img, self.img_size)             # square resize
            img = per_image_standardize(img)
            X.append(img); y.append(label)

        X = np.stack(X).astype(np.float32)
        y = tf.keras.utils.to_categorical(y, num_classes=self.n_classes, dtype="float32")
        if self.training and MIXUP_ALPHA > 0:
            X, y = mixup(X, y, MIXUP_ALPHA)
        return X, y

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

# -----------------------------
# Model
# -----------------------------
def build_model(num_classes, input_shape, base_trainable=False, dropout=DROPOUT):
    base = EfficientNetV2B0(include_top=False, input_shape=input_shape, pooling="avg")
    base.trainable = base_trainable
    x = base.output
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs=base.input, outputs=out)

# -----------------------------
# Train
# -----------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", required=True)
    ap.add_argument("--val_root",   required=True)
    ap.add_argument("--warmup_epochs", type=int, default=8)
    ap.add_argument("--finetune_epochs", type=int, default=40)
    ap.add_argument("--base_lr", type=float, default=3e-4)
    ap.add_argument("--img_size", type=int, default=IMG_SIZE)
    ap.add_argument("--batch", type=int, default=BATCH)
    args = ap.parse_args()

    label_names = list_label_names(args.train_root)
    label_map = {n: i for i, n in enumerate(label_names)}
    num_classes = len(label_names)
    print("Classes:", label_names)

    train_pairs = collect_pairs(args.train_root, label_map)
    val_pairs   = collect_pairs(args.val_root,   label_map)

    train_seq = NpyMelSequence(train_pairs, batch_size=args.batch, n_classes=num_classes,
                               training=True, img_size=args.img_size)
    val_seq   = NpyMelSequence(val_pairs,   batch_size=args.batch, n_classes=num_classes,
                               training=False, img_size=args.img_size)

    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTH)

    # Cosine schedule (smooth) for warmup / finetune
    steps_per_epoch = max(1, len(train_seq))
    cosine_warm = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=args.base_lr, first_decay_steps=steps_per_epoch*8
    )
    cosine_fine = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=args.base_lr*0.3, first_decay_steps=steps_per_epoch*8
    )

    out_dir = Path("models/Deep Learning/EfficientNetV2")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = str(out_dir / "efficientnetv2b0_harmonized_best.keras")

    cbs = [
        callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        callbacks.EarlyStopping(monitor="val_accuracy", patience=PATIENCE, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=1),
    ]

    # Warmup (freeze base)
    model = build_model(num_classes, input_shape=(args.img_size, args.img_size, 3), base_trainable=False)
    model.compile(optimizer=AdamW(learning_rate=cosine_warm, weight_decay=WEIGHT_DECAY),
                  loss=loss, metrics=["accuracy"])
    print("\n=== Warmup (base frozen) ===")
    model.fit(train_seq, validation_data=val_seq, epochs=args.warmup_epochs,
              callbacks=cbs, workers=NUM_WORKERS, use_multiprocessing=False)

    # Finetune (unfreeze base)
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) or "efficientnetv2" in layer.name.lower():
            layer.trainable = True
    model.compile(optimizer=AdamW(learning_rate=cosine_fine, weight_decay=WEIGHT_DECAY),
                  loss=loss, metrics=["accuracy"])
    print("\n=== Finetune (unfreeze base) ===")
    model.fit(train_seq, validation_data=val_seq, epochs=args.finetune_epochs,
              callbacks=cbs, workers=NUM_WORKERS, use_multiprocessing=False)

    # Save final + labels
    final_h5 = str(out_dir / "efficientnetv2b0_harmonized.h5")
    model.save(final_h5)
    with open(out_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)
    print("Saved:", final_h5)
    print("Saved best:", ckpt_path)

if __name__ == "__main__":
    main()
