import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------- Config (adjust if your paths change) ----------
SR = 22050
SEGMENT_DURATION = 1.5   # seconds
HOP_DURATION = 0.75      # seconds (50% overlap)
N_MELS = 128
IMG_SIZE = (2.24, 2.24)  # ~224x224 pixels

OUTPUT_BASE = os.path.abspath("spectrogram_dataset_windowed")  # root save dir
FMA_AUDIO_DIR = "fma_small/fma_small"           # your nested folder (per screenshot)
FMA_METADATA_DIR = "fma_metadata/fma_metadata"  # your nested folder (per screenshot)

# ---------- Helpers ----------
def trim_and_normalize(y):
    y_trimmed, _ = librosa.effects.trim(y)
    return librosa.util.normalize(y_trimmed)

def save_mel_spectrogram(y, sr, output_path):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    plt.figure(figsize=IMG_SIZE)
    librosa.display.specshow(mel_db, sr=sr)
    plt.axis('off')
    plt.tight_layout(pad=0)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def infer_track_id_from_path(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    # handle e.g. "000123" -> 123
    try:
        return int(stem)
    except:
        try:
            return int(stem.lstrip("0") or "0")
        except:
            return None

def find_tracks_csv(base_dir):
    """Find tracks.csv / tracks.csv.gz / tracks.csv.zip recursively."""
    for root, _, files in os.walk(base_dir):
        for fn in files:
            low = fn.lower()
            if low in ("tracks.csv", "tracks.csv.gz", "tracks.csv.zip"):
                return os.path.join(root, fn)
    raise FileNotFoundError(f"tracks.csv not found anywhere under {base_dir}/")

def load_fma_genre_map(fma_metadata_dir):
    csv_path = find_tracks_csv(fma_metadata_dir)
    print(f"📄 Using tracks file: {csv_path}")

    df = pd.read_csv(csv_path, index_col=0, header=[0, 1])  # handles compressed by ext
    # Multi-index or flattened columns
    if isinstance(df.columns, pd.MultiIndex):
        col = ('track', 'genre_top')
        if col not in df.columns:
            # fallback: any tuple ending with 'genre_top'
            for c in df.columns:
                if isinstance(c, tuple) and c[-1] == 'genre_top':
                    col = c
                    break
        series = df[col]
    else:
        if 'track.genre_top' in df.columns:
            series = df['track.genre_top']
        else:
            candidates = [c for c in df.columns if 'genre_top' in c]
            if not candidates:
                raise ValueError("Could not locate 'genre_top' column in tracks.csv")
            series = df[candidates[0]]

    # Map: track_id -> normalized genre string
    return series.fillna("unknown").astype(str).str.lower().to_dict()

# print first save path per-genre once
_debug_printed = set()

def process_audio_file(audio_path, out_root, genre, file_id):
    try:
        y, sr = librosa.load(audio_path, sr=SR, mono=True)
        y = trim_and_normalize(y)
        seg = int(SEGMENT_DURATION * sr)
        hop = int(HOP_DURATION * sr)

        for i in range(0, len(y) - seg, hop):
            segment = y[i:i + seg]
            if len(segment) < seg:
                continue

            segment_id = f"{file_id}_{i}"
            out_path = os.path.join(out_root, "fma", genre, f"{segment_id}.png")

            if (genre not in _debug_printed):
                print("Saving to:", out_path)
                _debug_printed.add(genre)

            if not os.path.exists(out_path):  # skip if already exists
                save_mel_spectrogram(segment, sr, out_path)
    except Exception as e:
        print(f"⚠️ Failed: {audio_path} -> {e}")

def process_fma(fma_dir, fma_metadata_dir, out_root):
    fma_out = os.path.join(out_root, "fma")
    if os.path.exists(fma_out):
        print("⏭️ Skipping FMA (already processed).")
        return

    print("\n📚 Loading FMA metadata...")
    genre_map = load_fma_genre_map(fma_metadata_dir)

    print("🎵 Processing FMA audio files...")
    for root, _, files in os.walk(fma_dir):
        audio_files = [f for f in files if f.lower().endswith((".mp3", ".wav"))]
        if not audio_files:
            continue
        for fname in tqdm(audio_files, desc=os.path.relpath(root, fma_dir)):
            apath = os.path.join(root, fname)
            tid = infer_track_id_from_path(apath)
            if tid is None or tid not in genre_map:
                continue
            genre = genre_map[tid].replace("/", "-").replace(" ", "_")
            if genre in ("unknown", "nan", ""):
                continue
            fid = os.path.splitext(fname)[0]
            process_audio_file(apath, out_root, genre, fid)

# ---------- Main ----------
if __name__ == "__main__":
    print("🚀 Generating FMA Mel spectrograms with trim + normalize + 1.5s windowing")
    if not os.path.isdir(FMA_AUDIO_DIR):
        print(f"⚠️ FMA audio dir not found: {FMA_AUDIO_DIR}")
    if not os.path.isdir(FMA_METADATA_DIR):
        print(f"⚠️ FMA metadata dir not found: {FMA_METADATA_DIR}")

    if os.path.isdir(FMA_AUDIO_DIR) and os.path.isdir(FMA_METADATA_DIR):
        process_fma(FMA_AUDIO_DIR, FMA_METADATA_DIR, OUTPUT_BASE)

    print("\n✅ Done. Output root:", OUTPUT_BASE)
