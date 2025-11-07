import os, glob, math, json, hashlib
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from tqdm import tqdm

CONFIG = {
    "sr": 22050,               # same for both datasets
    "n_fft": 2048,
    "hop_length": 512,
    "n_mels": 128,
    "fmin": 20,
    "fmax": 8000,
    "top_db_trim": 40,         # advanced trimming threshold
    "target_rms": 0.05,        # loudness norm target (≈ -26 dBFS-ish)
    "clip_sec": 29.0,          # uniform duration after trim/pad
    "ref_power_eps": 1e-10
}

def rms(x):
    return np.sqrt(np.mean(np.square(x), dtype=np.float64) + 1e-12)

def loudness_normalize(y, target_rms=CONFIG["target_rms"]):
    cur = rms(y)
    if cur < 1e-12: 
        return y
    gain = target_rms / cur
    y = y * gain
    # avoid clipping
    peak = np.max(np.abs(y)) + 1e-12
    if peak > 0.999:
        y = y / peak * 0.999
    return y

def advanced_trim(y, sr, top_db=CONFIG["top_db_trim"]):
    # remove leading/trailing low-energy parts
    yt, idx = librosa.effects.trim(y, top_db=top_db)
    return (yt if len(yt) > 0 else y)

def center_crop_or_pad(y, sr, target_sec):
    target_len = int(sr * target_sec)
    if len(y) == target_len:
        return y
    if len(y) > target_len:
        start = (len(y) - target_len)//2
        return y[start:start+target_len]
    # pad
    pad = target_len - len(y)
    left = pad // 2
    right = pad - left
    return np.pad(y, (left, right))

def mel_spectrogram(y, sr, cfg=CONFIG):
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=cfg["n_fft"], hop_length=cfg["hop_length"],
        n_mels=cfg["n_mels"], fmin=cfg["fmin"], fmax=cfg["fmax"], power=2.0
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db.astype(np.float32)

def file_hash(path):
    h = hashlib.sha1()
    h.update(Path(path).read_bytes())
    return h.hexdigest()[:10]

def process_audio_file(in_path, out_path, cfg=CONFIG):
    y, sr = librosa.load(in_path, sr=cfg["sr"], mono=True)
    y = advanced_trim(y, sr, cfg["top_db_trim"])
    y = loudness_normalize(y, cfg["target_rms"])
    y = center_crop_or_pad(y, sr, cfg["clip_sec"])
    M = mel_spectrogram(y, sr, cfg)   # [n_mels, time]
    # save as npy
    np.save(out_path, M, allow_pickle=False)

def collect_audio_files(root):
    exts = ("*.wav","*.mp3","*.flac","*.ogg","*.m4a")
    files = []
    for ext in exts:
        files += glob.glob(os.path.join(root, "**", ext), recursive=True)
    return files

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", required=True, help="root of raw audio (e.g., datasets/gtzan)")
    ap.add_argument("--out_root", required=True, help="output spectrogram root (e.g., spectrogram_dataset_harmonized/gtzan)")
    args = ap.parse_args()

    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    audio_files = collect_audio_files(args.in_root)
    print(f"Found {len(audio_files)} audio files")

    cfg_path = Path(args.out_root)/"harmonized_config.json"
    with open(cfg_path, "w") as f:
        json.dump(CONFIG, f, indent=2)

    for fpath in tqdm(audio_files):
        rel = os.path.relpath(fpath, args.in_root)
        base = os.path.splitext(rel)[0] + f"__{file_hash(fpath)}.npy"
        out_path = Path(args.out_root)/base
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            continue
        try:
            process_audio_file(fpath, out_path)
        except Exception as e:
            print("ERR:", fpath, e)

if __name__ == "__main__":
    main()
