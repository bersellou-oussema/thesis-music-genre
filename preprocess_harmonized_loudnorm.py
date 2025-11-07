#!/usr/bin/env python3
# preprocess_harmonized_loudnorm.py
# Build harmonized Mel-spectrograms (dB) with EBU R128 loudness norm (safe).
# Works for GTZAN (data_wav) and FMA (fma_wav, etc).

import os
import sys
import math
import argparse
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
import scipy.signal as sps

# -----------------------------
# Config
# -----------------------------
TARGET_SR   = 22050
N_MELS      = 128
HOP_LENGTH  = 512
N_FFT       = 2048
TARGET_LUFS = -14.0  # loudness target

# -----------------------------
# Small utils
# -----------------------------
def nan_to_zero(y: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf with zeros; keep float32."""
    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

def safe_load_audio(path: str, target_sr: int = TARGET_SR) -> np.ndarray:
    """
    Robust loader:
      1) try librosa.load
      2) fallback to soundfile
      3) ensure mono, resample if needed
      4) sanitize NaN/Inf
    """
    # 1) librosa
    try:
        y, sr = librosa.load(path, sr=target_sr, mono=True, dtype=np.float32)
        y = nan_to_zero(y)
        if y.size and np.isfinite(y).all():
            return y
    except Exception:
        pass

    # 2) soundfile fallback
    try:
        y, sr = sf.read(path, always_2d=False, dtype="float32")
    except Exception:
        # unreadable
        return np.array([], dtype=np.float32)

    # ensure mono
    if y.ndim == 2:
        y = y.mean(axis=1)

    # resample if needed (stable polyphase)
    if y.size and sr != target_sr:
        g = math.gcd(int(sr), int(target_sr))
        up, down = target_sr // g, sr // g
        y = sps.resample_poly(y, up, down).astype(np.float32, copy=False)

    return nan_to_zero(y)

def loudnorm_or_skip(y: np.ndarray, target_lufs: float = TARGET_LUFS) -> np.ndarray:
    """
    Apply EBU R128 loudness normalization when possible.
    If loudness can't be computed (silence/NaNs), return sanitized y.
    """
    y = nan_to_zero(y)
    if not y.size or np.allclose(y, 0.0):
        return y
    try:
        import pyloudnorm as pyln
        meter = pyln.Meter(TARGET_SR)
        loud = meter.integrated_loudness(y)
        if not np.isfinite(loud):
            return y
        gain = target_lufs - loud
        y = y * (10.0 ** (gain / 20.0))
        return nan_to_zero(y)
    except Exception:
        return nan_to_zero(y)

def audio_to_mel_db(y: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """Waveform -> Mel spectrogram in dB (float32)."""
    if not y.size or not np.any(np.isfinite(y)):
        raise ValueError("Waveform empty or non-finite after sanitation")

    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, power=2.0
    )
    # avoid log(0)
    S = np.maximum(S, 1e-10)
    mel_db = librosa.power_to_db(S, ref=np.max).astype(np.float32)
    return mel_db

def process_file(in_path: Path, out_path: Path) -> None:
    """Process a single audio file and save Mel dB .npy."""
    y = safe_load_audio(str(in_path), TARGET_SR)
    y = loudnorm_or_skip(y, TARGET_LUFS)

    if not y.size or np.allclose(y, 0.0):
        print(f"[SKIP] Silent/invalid: {in_path}")
        return

    mel_db = audio_to_mel_db(y, TARGET_SR)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, mel_db)

# -----------------------------
# Dataset walker
# -----------------------------
def process_dataset(raw_root: Path, out_root: Path, exts=(".wav", ".mp3", ".flac", ".ogg")):
    """
    Expect structure:
      raw_root/
        genre_a/*.wav
        genre_b/*.wav
        ...
    Output mirrors directory tree into out_root with .npy files.
    """
    if not raw_root.exists():
        print(f"[FATAL] Raw root not found: {raw_root}")
        sys.exit(1)

    files = [p for p in raw_root.rglob("*") if p.suffix.lower() in exts]
    if not files:
        print(f"[WARN] No audio files in {raw_root}")
        return

    print(f"=== Processing {raw_root.name} -> {out_root} ===")
    n = len(files)
    for i, f in enumerate(files, 1):
        rel = f.relative_to(raw_root)
        out_path = (out_root / rel).with_suffix(".npy")
        try:
            process_file(f, out_path)
        except Exception as e:
            print(f"[SKIP-ERROR] {f} → {e}")
        if i % 50 == 0 or i == n:
            print(f"Processed {i}/{n}")

# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Harmonized Mel-spectrogram preprocessor with safe loudness norm."
    )
    ap.add_argument("--raw_root", required=True,
                    help="Input audio root (e.g., data_wav or fma_wav)")
    ap.add_argument("--out_root", required=True,
                    help="Output root for .npy mel-spectrograms")
    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)

    process_dataset(raw_root, out_root)

if __name__ == "__main__":
    main()
