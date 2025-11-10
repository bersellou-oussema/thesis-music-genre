import os
import json
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Paths
SPLIT_JSON = "splits_gtzan/gtzan_splits.json"  # created by your previous script
OUT_ROOT   = "spectrogram_dataset_windowed_splits/gtzan"

os.makedirs(OUT_ROOT, exist_ok=True)

# Audio / window params (adjust if needed to match your old setup)
SR                = 22050
SEGMENT_DURATION  = 3.0   # seconds per window
N_MELS            = 128
HOP_LENGTH        = 512

with open(SPLIT_JSON, "r") as f:
    splits = json.load(f)

def save_mel_spectrogram(y_seg, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    S = librosa.feature.melspectrogram(
        y=y_seg,
        sr=SR,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH
    )
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(2.24, 2.24))
    plt.axis("off")
    librosa.display.specshow(S_dB, sr=SR, hop_length=HOP_LENGTH)
    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()

segment_samples = int(SEGMENT_DURATION * SR)

for split_name, items in splits.items():
    print(f"\n=== Processing split: {split_name} ({len(items)} tracks) ===")
    for entry in items:
        audio_path = entry["path"]
        if not os.path.exists(audio_path):
            print(f"[WARN] Missing file: {audio_path}")
            continue

        genre = os.path.basename(os.path.dirname(audio_path))
        base  = os.path.splitext(os.path.basename(audio_path))[0]

        try:
            y, sr = librosa.load(audio_path, sr=SR, mono=True)
        except Exception as e:
            print(f"[ERR] Failed to load {audio_path}: {e}")
            continue

        # Normalize
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))

        total_samples = len(y)

        # Pad short tracks to fit at least one full segment
        if total_samples < segment_samples:
            pad_width = segment_samples - total_samples
            y = np.pad(y, (0, pad_width))
            total_samples = len(y)

        # Non-overlapping segments (you can change the step for overlap)
        start = 0
        seg_idx = 0
        while start + segment_samples <= total_samples:
            y_seg = y[start:start + segment_samples]
            out_path = os.path.join(
                OUT_ROOT,
                split_name,
                genre,
                f"{base}_seg{seg_idx}.png"
            )
            save_mel_spectrogram(y_seg, out_path)
            seg_idx += 1
            start += segment_samples

        if seg_idx == 0:
            print(f"[WARN] No segments for {audio_path}")
        else:
            print(f"{split_name} | {genre} | {base}: {seg_idx} segments")

print("\n✅ Done. Windowed mel-spectrograms saved under:", OUT_ROOT)
