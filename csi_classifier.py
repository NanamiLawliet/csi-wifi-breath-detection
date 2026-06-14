#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNN + BiLSTM CSI Nefes Siniflandirici  (Dual-Input: STFT + Zaman-Istatistik)
==============================================================================
Pipeline:
  Ham CSI -> Temporal %70/%15/%15 (karistirilmadan, sizintisiz)
  -> BNR Top-20 SC secimi (Slow+Fast breath TRAIN verisinden)
       BNR bandi: [0.16-0.60 Hz]  (HVAC 8-10 BPM disi)
  -> 20s sliding windows (1s adim)
  -> Per-window DSP: Hampel(hw=5) + SG(11,3) + Bandpass(0.10-0.60 Hz)
       KRITIK: SG_WINDOW=11 (NOT 51) -- 25 BPM sinyalini korur!
  -> Global PCA(3 PC, sadece train fitlenir)

  [Path 1 - Spectrogram]
    -> STFT Spectrogram [0.10-0.60 Hz]  -> (6 freq, 11 time, 3 PC)
    -> 2D-CNN + Freq-Avg Lambda + BiLSTM -> (128,)

  [Path 2 - Zaman Istatistikleri]
    -> Per-window stats: (var, rms, p2p, dom_bpm, dom_amp, spectral_ent) x 3 PC -> (18,)
    -> Dense(32) + Dense(32) -> (32,)
    Amac: her iki sinif ~28 BPM oldugunda AMPLITUD farki ile ayirt et
          Slow_Breath = derin nefes -> yuksek amplitude
          Fast_Breath = yuzeysel nefes -> dusuk amplitude
          Empty_Room  = nefes yok -> cok dusuk amplitude

  [Merge] Concat([128, 32]) -> Dense(64) -> Dropout -> Softmax(3)

  -> Data Augmentation x4 (gurultu + olcekleme + zaman maskeleme)
  -> Keras/TF backend (PyTorch fallback: sadece spectrogram yolu)

Siniflar: Empty_Room | Slow_Breath | Fast_Breath
"""

import ast, io, os, sys, warnings
warnings.filterwarnings('ignore')

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.signal import butter, filtfilt, savgol_filter, medfilt, stft as scipy_stft, welch
from scipy.integrate import trapezoid
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                              confusion_matrix, classification_report)

# ─── Backend ──────────────────────────────────────────────────────────────────
USE_KERAS = False
try:
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    print(f"Backend: TensorFlow {tf.__version__}")
    USE_KERAS = True
except ImportError:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torch.optim.lr_scheduler import ReduceLROnPlateau as TorchReduceLR
    print(f"Backend: PyTorch {torch.__version__}  (TF bulunamadi)")

# ═══════════════════════════════════════════════════════════════════════════════
# Sabitler
# ═══════════════════════════════════════════════════════════════════════════════
FS          = 13           # ESP32 nominal CSI ornekleme hizi (Hz)
WINDOW_S    = 20           # pencere uzunlugu (saniye)
STEP_S      = 1            # adim (saniye)
WINDOW_N    = WINDOW_S * FS    # = 260 paket
STEP_N      = STEP_S   * FS   # = 13  paket

# ── BNR Subcarrier Secimi ──
TOP_N_SC    = 20           # BNR ile secilecek en iyi SC sayisi
BNR_LO      = 0.16         # BNR hesaplama alt siniri Hz -- HVAC (8-10 BPM) disi
BNR_HI      = 0.60         # BNR hesaplama ust siniri Hz

# ── Global BNR on-off Hampel+SG (tam sinyal uzeri) ──
HAMPEL_HALF_G = 50         # Hampel yari-pencere (global BNR icin)
HAMPEL_NSIG_G = 2.5
SG_WIN_G      = 11         # KRITIK: 11 NOT 51 -- 25 BPM icin T*Fs=31.2 -> win<31
SG_ORD        = 3

# ── Per-window DSP ──
BP_LO       = 0.10         # Bandpass alt sinir Hz  (6 BPM)
BP_HI       = 0.60         # Bandpass ust sinir Hz  (36 BPM, 25 BPM=0.417 Hz dahil)
HAMPEL_HALF_W = 5          # Hampel yari-pencere (pencere bazli)
HAMPEL_NSIG_W = 3.0

# ── STFT ──
STFT_LO     = 0.10         # Hz
STFT_HI     = 0.60         # Hz
STFT_NPERSEG = 10 * FS     # = 130  (10s alt-pencere -> Df=0.10 Hz cozunurluk)
STFT_HOP    = FS           # = 13   (1s hop)
STFT_NOVERLAP = STFT_NPERSEG - STFT_HOP  # = 117

# ── Zaman-Istatistik Ozellikleri ──
# Her PC icin: var, rms, p2p, dom_bpm, dom_amp, spectral_entropy = 6 ozellik
N_STATS_PER_PC = 6
N_STATS_FEATURES = None   # STFT kurulumundan sonra atanir (= N_PCA * N_STATS_PER_PC)

# ── Model ──
N_PCA       = 3
N_CLASSES   = 3
CLASS_NAMES = ['Empty_Room', 'Slow_Breath', 'Fast_Breath']
RANDOM_SEED = 42

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
FILES = {
    'Empty_Room' : os.path.join(DATA_DIR, 'csi_data_empty_my_room.csv'),
    'Slow_Breath': os.path.join(DATA_DIR, 'csi_data_long_breath.csv'),
    'Fast_Breath': os.path.join(DATA_DIR, 'csi_data_25bpm_breath.csv'),
}

# ── Bandpass filtre katsayilari (once hesapla) ──
_nyq = FS / 2.0
_B, _A = butter(4, [BP_LO / _nyq, BP_HI / _nyq], btype='band')

# ── STFT frekans ekseni ve bant maskesi ──
_FREQS_STFT = np.fft.rfftfreq(STFT_NPERSEG, d=1.0 / FS)
_BAND_MASK  = (_FREQS_STFT >= STFT_LO - 1e-9) & (_FREQS_STFT <= STFT_HI + 1e-9)
N_FREQ_BINS = int(_BAND_MASK.sum())
_N_FRAMES   = 1 + (WINDOW_N - STFT_NPERSEG) // STFT_HOP
SPEC_SHAPE  = (N_FREQ_BINS, _N_FRAMES, N_PCA)

N_STATS_FEATURES = N_PCA * N_STATS_PER_PC   # = 18

print(f"\nSTFT: nperseg={STFT_NPERSEG}({STFT_NPERSEG//FS}s)  "
      f"hop={STFT_HOP//FS}s  Df={FS/STFT_NPERSEG:.3f} Hz")
print(f"Bant: {_FREQS_STFT[_BAND_MASK][0]:.2f}-{_FREQS_STFT[_BAND_MASK][-1]:.2f} Hz  "
      f"= {_FREQS_STFT[_BAND_MASK][0]*60:.0f}-{_FREQS_STFT[_BAND_MASK][-1]*60:.0f} BPM  "
      f"({N_FREQ_BINS} bin)")
print(f"Spectrogram sekli: {SPEC_SHAPE}   |  Stats ozellikleri: {N_STATS_FEATURES}")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CSI Yukleme
# ═══════════════════════════════════════════════════════════════════════════════
def _parse_amp(csi_str: str):
    try:
        raw  = np.array(ast.literal_eval(str(csi_str).strip()), dtype=np.float32)
        vals = raw[4:]           # ilk 4: pilot subcarrier (gecersiz)
        n    = (len(vals) // 2) * 2
        return np.sqrt(vals[0:n:2] ** 2 + vals[1:n:2] ** 2)
    except Exception:
        return None


def load_amps(filepath: str, label: str) -> np.ndarray:
    print(f"  [{label}] {os.path.basename(filepath)}")
    df   = pd.read_csv(filepath)
    amps = [_parse_amp(s) for s in df['csi_data']]
    amps = [a for a in amps if a is not None]
    if not amps:
        raise ValueError(f"Veri yuklenemedi: {filepath}")
    min_len = min(len(a) for a in amps)
    mat = np.stack([a[:min_len] for a in amps])
    print(f"           {mat.shape[0]} paket, {mat.shape[1]} subcarrier (ham)")
    return mat


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Zamansal Bolme  (%70 / %15 / %15)
# ═══════════════════════════════════════════════════════════════════════════════
def temporal_split(mat, tr=0.70, va=0.15):
    n = len(mat)
    return (mat[:int(n * tr)],
            mat[int(n * tr): int(n * (tr + va))],
            mat[int(n * (tr + va)):])


# ═══════════════════════════════════════════════════════════════════════════════
# 3. BNR Subcarrier Secimi (global, sadece train verisinden)
# ═══════════════════════════════════════════════════════════════════════════════
def _hampel_full(x: np.ndarray, hw: int = 50, k: float = 2.5) -> np.ndarray:
    ksize = 2 * hw + 1
    med   = medfilt(x.astype(np.float64), ksize)
    mad   = medfilt(np.abs(x - med), ksize)
    bad   = (mad > 0) & (np.abs(x - med) > k * 1.4826 * mad)
    out   = x.copy(); out[bad] = med[bad]
    return out


def _compute_bnr(sig: np.ndarray) -> float:
    if sig.var() < 1e-6:
        return 0.0
    nperseg = min(len(sig), max(256, int(FS * 15)))
    f, p = welch(sig.astype(np.float64), fs=FS, nperseg=nperseg, scaling='density')
    in_b  = (f >= BNR_LO) & (f <= BNR_HI)
    out_b = ~in_b & (f > 0)
    if in_b.sum() == 0 or out_b.sum() == 0:
        return 0.0
    num = trapezoid(p[in_b],  f[in_b])
    den = trapezoid(p[out_b], f[out_b]) + 1e-12
    return float(num / den)


def select_bnr_subcarriers(breath_train: np.ndarray,
                            n_select: int = TOP_N_SC) -> np.ndarray:
    n_sc     = breath_train.shape[1]
    bnr_vals = np.zeros(n_sc)
    print(f"  BNR hesaplaniyor ({n_sc} aktif SC, {len(breath_train)} paket)...", flush=True)
    for c in range(n_sc):
        sig = breath_train[:, c].astype(np.float64)
        sig = _hampel_full(sig, hw=HAMPEL_HALF_G, k=HAMPEL_NSIG_G)
        sig = savgol_filter(sig, window_length=SG_WIN_G, polyorder=SG_ORD)
        bnr_vals[c] = _compute_bnr(sig)
        if (c + 1) % 30 == 0 or c == n_sc - 1:
            print(f"    [{c+1}/{n_sc}] done", end='\r')
    print()
    top_idx = np.argsort(bnr_vals)[-n_select:][::-1]
    top_bnr = bnr_vals[top_idx]
    print(f"  Top-{n_select} BNR: min={top_bnr.min():.4f}  max={top_bnr.max():.4f}  "
          f"mean={top_bnr.mean():.4f}")
    return top_idx


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Per-Window DSP
# ═══════════════════════════════════════════════════════════════════════════════
def _hampel_col(x: np.ndarray, hw: int = HAMPEL_HALF_W,
                k: float = HAMPEL_NSIG_W) -> np.ndarray:
    ksize = 2 * hw + 1
    med   = medfilt(x.astype(np.float64), ksize)
    mad   = medfilt(np.abs(x - med), ksize)
    bad   = (mad > 0) & (np.abs(x - med) > k * 1.4826 * mad)
    out   = x.copy(); out[bad] = med[bad]
    return out


def dsp_window(win: np.ndarray) -> np.ndarray:
    """
    win: (WINDOW_N, TOP_N_SC) ham amplitud
    -> Hampel(5) + SG(11,3) + Bandpass(0.10-0.60 Hz)
    KRITIK: SG window=11 (NOT 51).
    Kural: SG_WIN < T_breathing * Fs
    25 BPM: T=2.4s -> T*Fs=31.2 -> SG_WIN=11 < 31.2 (ok)
    SG_WIN=51 > 31.2 -> 25 BPM sinyali yok edilir (yanlis!)
    """
    out = win.copy().astype(np.float64)
    for c in range(out.shape[1]):
        out[:, c] = _hampel_col(out[:, c])
        out[:, c] = savgol_filter(out[:, c], window_length=11, polyorder=SG_ORD)
        try:
            out[:, c] = filtfilt(_B, _A, out[:, c])
        except Exception:
            out[:, c] = 0.0
    return out.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# 5a. STFT Spectrogram  [Path 1]
# ═══════════════════════════════════════════════════════════════════════════════
def compute_spectrogram(window_pca: np.ndarray) -> np.ndarray:
    """
    window_pca: (WINDOW_N, N_PCA)
    -> (N_FREQ_BINS, N_FRAMES, N_PCA) dB-magnitude spectrogram, per-sample normalize
    """
    spec = np.zeros(SPEC_SHAPE, dtype=np.float32)
    for k in range(N_PCA):
        _, _, Zxx = scipy_stft(
            window_pca[:, k], fs=FS,
            nperseg=STFT_NPERSEG, noverlap=STFT_NOVERLAP,
            window='hann', boundary=None, padded=False
        )
        mag_db = 20 * np.log10(np.abs(Zxx) + 1e-12)
        n_t    = min(mag_db.shape[1], _N_FRAMES)
        spec[:, :n_t, k] = mag_db[_BAND_MASK, :n_t]
    mu  = spec.mean()
    std = spec.std() + 1e-8
    return ((spec - mu) / std).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# 5b. Zaman-Boyutu Istatistik Ozellikleri  [Path 2]
# ═══════════════════════════════════════════════════════════════════════════════
def compute_window_stats(window_pca: np.ndarray) -> np.ndarray:
    """
    window_pca: (WINDOW_N, N_PCA) -- bandpassed + PCA-transformed

    Her PC icin 6 ozellik:
      1. var        : varyans (amplitud gucu)
      2. rms        : kare-ortalama-kok (enerji)
      3. p2p        : tepe-tepe genlik (dinamik aralik)
      4. dom_bpm    : bant icindeki baskın frekans (BPM)
      5. dom_amp    : baskın frekanstaki PSD genligi (nefes gucu)
      6. spec_ent   : spektral entropi (nefes duzenliligi; dusuk = duzensiz)

    Amac: her iki sinif ~28 BPM iken AMPLITUD ile ayirt et:
      Slow_Breath (derin nefes) -> yuksek var/rms/p2p
      Fast_Breath (yuzeysel)   -> orta var/rms/p2p
      Empty_Room  (nefes yok)  -> cok dusuk var/rms/p2p

    -> (N_STATS_FEATURES=18,) float32
    """
    feats = np.zeros(N_STATS_FEATURES, dtype=np.float32)
    for k in range(N_PCA):
        x = window_pca[:, k].astype(np.float64)

        nperseg_w = min(len(x), 64)
        f_w, psd_w = welch(x, fs=FS, nperseg=nperseg_w)
        band_w = (f_w >= BP_LO) & (f_w <= BP_HI)
        psd_b  = psd_w[band_w]
        f_b    = f_w[band_w]

        var     = float(np.var(x))
        rms     = float(np.sqrt(np.mean(x ** 2)))
        p2p     = float(np.max(x) - np.min(x))
        if len(psd_b) > 0:
            dom_bpm = float(f_b[np.argmax(psd_b)] * 60)
            dom_amp = float(np.max(psd_b))
            psd_norm = psd_b / (psd_b.sum() + 1e-12)
            spec_ent = float(-np.sum(psd_norm * np.log(psd_norm + 1e-12)))
        else:
            dom_bpm = 0.0
            dom_amp = 0.0
            spec_ent = 0.0

        base = k * N_STATS_PER_PC
        feats[base:base + N_STATS_PER_PC] = [var, rms, p2p, dom_bpm, dom_amp, spec_ent]
    return feats


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Sliding Windows
# ═══════════════════════════════════════════════════════════════════════════════
def make_windows(mat: np.ndarray, desc: str = '') -> np.ndarray:
    n      = len(mat)
    starts = list(range(0, n - WINDOW_N + 1, STEP_N))
    wins   = []
    for i, s in enumerate(starts):
        wins.append(dsp_window(mat[s: s + WINDOW_N]))
        if desc and (i % 50 == 0 or i == len(starts) - 1):
            print(f"    {desc}: {i+1}/{len(starts)}", end='\r')
    if desc:
        print()
    if not wins:
        return np.empty((0, WINDOW_N, TOP_N_SC), dtype=np.float32)
    return np.stack(wins)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Veri Seti Olusturma
# ═══════════════════════════════════════════════════════════════════════════════
def build_dataset():
    print("\n" + "=" * 70)
    print(f"CNN+BiLSTM (Dual-Input)  |  BNR Top-{TOP_N_SC} SC  |  {WINDOW_S}s  "
          f"|  STFT{SPEC_SHAPE} + Stats({N_STATS_FEATURES})")
    print("=" * 70)

    # 1. Yukle
    raw: dict = {}
    for lbl, path in FILES.items():
        raw[lbl] = load_amps(path, lbl)

    min_sc = min(r.shape[1] for r in raw.values())
    for lbl in raw:
        raw[lbl] = raw[lbl][:, :min_sc]

    # 2. Zamansal bolme (sizinti yok)
    tr_mat, va_mat, te_mat = {}, {}, {}
    for lbl in CLASS_NAMES:
        tr_mat[lbl], va_mat[lbl], te_mat[lbl] = temporal_split(raw[lbl])
        print(f"  [{lbl}]: train={len(tr_mat[lbl])} val={len(va_mat[lbl])} "
              f"test={len(te_mat[lbl])} paket")

    # 3. Olu SC eleme
    tr_all   = np.concatenate([tr_mat[lbl] for lbl in CLASS_NAMES], axis=0)
    alive    = np.var(tr_all, axis=0) > 1e-6
    n_active = int(alive.sum())
    print(f"\nAktif subcarrier: {n_active} / {min_sc}")
    for lbl in CLASS_NAMES:
        tr_mat[lbl] = tr_mat[lbl][:, alive]
        va_mat[lbl] = va_mat[lbl][:, alive]
        te_mat[lbl] = te_mat[lbl][:, alive]

    # 4. BNR Subcarrier Secimi
    print(f"\nBNR Subcarrier Secimi [BNR_LO={BNR_LO} Hz, BNR_HI={BNR_HI} Hz]...")
    breath_train = np.concatenate(
        [tr_mat['Slow_Breath'], tr_mat['Fast_Breath']], axis=0
    )
    sc_idx = select_bnr_subcarriers(breath_train, n_select=TOP_N_SC)
    print(f"  Secilen SC indeksleri (ilk 10): {sc_idx[:10].tolist()}")

    for lbl in CLASS_NAMES:
        tr_mat[lbl] = tr_mat[lbl][:, sc_idx]
        va_mat[lbl] = va_mat[lbl][:, sc_idx]
        te_mat[lbl] = te_mat[lbl][:, sc_idx]

    # 5. Sliding Windows + Per-Window DSP
    print(f"\nSliding Windows ({WINDOW_S}s, step={STEP_S}s) + DSP...")
    Xw_tr, Xw_va, Xw_te = {}, {}, {}
    for lbl in CLASS_NAMES:
        Xw_tr[lbl] = make_windows(tr_mat[lbl], desc=f'{lbl}/tr')
        Xw_va[lbl] = make_windows(va_mat[lbl], desc=f'{lbl}/va')
        Xw_te[lbl] = make_windows(te_mat[lbl], desc=f'{lbl}/te')
        print(f"  [{lbl}]: train={len(Xw_tr[lbl])} val={len(Xw_va[lbl])} "
              f"test={len(Xw_te[lbl])} pencere")

    # 6. Global PCA(3) -- SADECE train uzerinde fit
    print(f"\nGlobal PCA({N_PCA}) fit (train pencereleri)...")
    tr_flat = np.concatenate(
        [Xw_tr[l].reshape(-1, TOP_N_SC) for l in CLASS_NAMES if len(Xw_tr[l])], axis=0
    )
    scaler = StandardScaler()
    pca    = PCA(n_components=N_PCA, random_state=RANDOM_SEED)
    scaler.fit(tr_flat)
    pca.fit(scaler.transform(tr_flat))
    exp_var = pca.explained_variance_ratio_.sum() * 100
    ev_str  = ', '.join(f'{v*100:.1f}%' for v in pca.explained_variance_ratio_)
    print(f"  Aciklanan varyans: {exp_var:.1f}%  ({ev_str})")

    def apply_pca(wins_dict):
        out = {}
        for lbl in CLASS_NAMES:
            W = wins_dict[lbl]
            if not len(W):
                out[lbl] = np.empty((0, WINDOW_N, N_PCA), dtype=np.float32)
                continue
            N, T, F = W.shape
            out[lbl] = pca.transform(
                scaler.transform(W.reshape(-1, F))
            ).reshape(N, T, N_PCA).astype(np.float32)
        return out

    pc_tr = apply_pca(Xw_tr)
    pc_va = apply_pca(Xw_va)
    pc_te = apply_pca(Xw_te)

    # 7a. STFT Spectrogram  [Path 1]
    print(f"\nSTFT Spectrogram {SPEC_SHAPE} hesaplaniyor...")

    def to_spectrograms(pc_dict):
        Xl, yl = [], []
        for idx, lbl in enumerate(CLASS_NAMES):
            W = pc_dict[lbl]
            if not len(W):
                continue
            specs = np.stack([compute_spectrogram(W[i]) for i in range(len(W))])
            Xl.append(specs)
            yl.append(np.full(len(specs), idx, dtype=np.int64))
        return np.concatenate(Xl), np.concatenate(yl)

    X_tr, y_tr = to_spectrograms(pc_tr)
    X_va, y_va = to_spectrograms(pc_va)
    X_te, y_te = to_spectrograms(pc_te)

    # 7b. Zaman-Istatistik Ozellikleri  [Path 2]
    print(f"Zaman Istatistikleri ({N_STATS_FEATURES} ozellik/pencere) hesaplaniyor...")

    def to_stats(pc_dict):
        Sl = []
        for lbl in CLASS_NAMES:
            W = pc_dict[lbl]
            if not len(W):
                continue
            Sl.append(np.stack([compute_window_stats(W[i]) for i in range(len(W))]))
        return np.concatenate(Sl)

    S_tr_raw = to_stats(pc_tr)
    S_va_raw = to_stats(pc_va)
    S_te_raw = to_stats(pc_te)

    # Stats normalizasyonu (SADECE train uzerinde fit)
    stats_scaler = StandardScaler()
    S_tr = stats_scaler.fit_transform(S_tr_raw).astype(np.float32)
    S_va = stats_scaler.transform(S_va_raw).astype(np.float32)
    S_te = stats_scaler.transform(S_te_raw).astype(np.float32)

    # Ozellik istatistikleri (train)
    print(f"  Stats (train norm): mean={S_tr.mean():.3f}  std={S_tr.std():.3f}")
    per_class_var = {}
    offset = 0
    for idx, lbl in enumerate(CLASS_NAMES):
        cnt = int((y_tr == idx).sum())
        if cnt > 0:
            per_class_var[lbl] = float(S_tr[y_tr == idx, :3].mean(axis=0).mean())
        offset += cnt
    for lbl, v in per_class_var.items():
        print(f"    [{lbl}] ort. amplitude stats (PC1-3 var normalizasyon): {v:.3f}")

    print(f"\nVeri seti:")
    print(f"  Train : {len(X_tr):4d}  {dict(zip(CLASS_NAMES, np.bincount(y_tr).tolist()))}")
    print(f"  Val   : {len(X_va):4d}  {dict(zip(CLASS_NAMES, np.bincount(y_va).tolist()))}")
    print(f"  Test  : {len(X_te):4d}  {dict(zip(CLASS_NAMES, np.bincount(y_te).tolist()))}")

    _plot_avg_spectrogram(pc_tr)
    return X_tr, X_va, X_te, S_tr, S_va, S_te, y_tr, y_va, y_te


def _plot_avg_spectrogram(pc_tr):
    bpm  = _FREQS_STFT[_BAND_MASK] * 60
    fig, axes = plt.subplots(N_PCA, N_CLASSES, figsize=(N_CLASSES * 4, N_PCA * 3))
    for k in range(N_PCA):
        for j, lbl in enumerate(CLASS_NAMES):
            W = pc_tr[lbl]
            if not len(W):
                continue
            specs = np.stack([compute_spectrogram(W[i])[:, :, k] for i in range(len(W))])
            avg   = specs.mean(axis=0)
            ax    = axes[k][j]
            im    = ax.imshow(avg, aspect='auto', origin='lower',
                              extent=[0, _N_FRAMES, bpm[0], bpm[-1]], cmap='inferno')
            ax.set_title(f'{lbl} - PC{k+1}', fontsize=8)
            ax.set_xlabel('Zaman cercevesi')
            ax.set_ylabel('BPM')
            plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f'Ort. STFT Spectrogram (Train | BNR Top-{TOP_N_SC} SC)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(DATA_DIR, 'spectrogram_by_class.png')
    plt.savefig(path, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"  Spectrogram grafigi: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Data Augmentation (Dual-Input)
# ═══════════════════════════════════════════════════════════════════════════════
def augment_data(X: np.ndarray, S: np.ndarray, y: np.ndarray,
                 factor: int = 4) -> tuple:
    """
    X: (N, freq, time, pc) spectrogram
    S: (N, N_STATS_FEATURES) normalize edilmis istatistikler
    y: (N,) etiketler

    Spectrogram augmentasyonu:
      - Gaussian gurultu: sigma=0.10
      - Magnitude olcekleme: 0.85-1.15
      - Zaman maskeleme: 1 rastgele zaman cercevesi sifirlanir (SpecAugment)

    Stats augmentasyonu:
      - Hafif Gaussian gurultu: sigma=0.05 (normalize uzayda)
    """
    rng    = np.random.default_rng(RANDOM_SEED)
    aug_X, aug_S, aug_y = [X], [S], [y]
    for _ in range(factor - 1):
        # Spectrogram: gurultu + olcekleme
        noise = rng.normal(0, 0.10, size=X.shape).astype(np.float32)
        scale = rng.uniform(0.85, 1.15, size=(len(X), 1, 1, 1)).astype(np.float32)
        X_aug = (X + noise) * scale

        # Zaman maskeleme (SpecAugment): 1 rastgele zaman cercevesi sifirla
        t_mask = rng.integers(0, _N_FRAMES, size=len(X))
        X_aug_cp = X_aug.copy()
        for i, t in enumerate(t_mask):
            X_aug_cp[i, :, t, :] = 0.0
        aug_X.append(X_aug_cp)

        # Stats: hafif gurultu (normalize uzayda kucuk perturbation)
        s_noise = rng.normal(0, 0.05, size=S.shape).astype(np.float32)
        aug_S.append(S + s_noise)
        aug_y.append(y.copy())

    Xa  = np.concatenate(aug_X)
    Sa  = np.concatenate(aug_S)
    ya  = np.concatenate(aug_y)
    idx = rng.permutation(len(Xa))
    return Xa[idx], Sa[idx], ya[idx]


# ═══════════════════════════════════════════════════════════════════════════════
# 9a. Keras/TF Modeli: Dual-Input (STFT + Stats)
# ═══════════════════════════════════════════════════════════════════════════════
def build_keras_model():
    """
    [Path 1 - Spectrogram] Giris: (6, 11, 3)
      Conv2D(32) -> BN -> ReLU -> MaxPool(freq 6->3) -> SpatialDrop(0.10)
      Conv2D(64) -> BN -> ReLU -> SpatialDrop(0.10)
      Lambda(freq_avg): (batch, 3, 11, 64) -> (batch, 11, 64)
      BiLSTM(64): -> (batch, 128)

    [Path 2 - Stats] Giris: (18,)
      Dense(32, relu) -> Dropout(0.20)
      Dense(32, relu)
      -> (batch, 32)

    [Merge] Concat([128, 32]) = (batch, 160)
      Dense(64, relu) -> Dropout(0.25)
      Dense(3, softmax)
    """
    reg = keras.regularizers.l2(1e-4)

    # ── Path 1: Spectrogram -> 2D-CNN + BiLSTM ──
    spec_inp = keras.Input(shape=SPEC_SHAPE, name='spectrogram')
    x = layers.Conv2D(32, (3, 3), padding='same', use_bias=False)(spec_inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 1))(x)
    x = layers.SpatialDropout2D(0.10)(x)
    x = layers.Conv2D(64, (3, 3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.SpatialDropout2D(0.10)(x)
    x = layers.Lambda(
        lambda t: tf.reduce_mean(t, axis=1), name='freq_avg'
    )(x)
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=False, dropout=0.10, recurrent_dropout=0.0),
        name='bilstm'
    )(x)
    # x: (batch, 128)

    # ── Path 2: Stats -> Dense ──
    stats_inp = keras.Input(shape=(N_STATS_FEATURES,), name='stats')
    s = layers.Dense(32, activation='relu', kernel_regularizer=reg)(stats_inp)
    s = layers.Dropout(0.20)(s)
    s = layers.Dense(32, activation='relu', kernel_regularizer=reg)(s)
    # s: (batch, 32)

    # ── Merge ──
    merged = layers.Concatenate(name='merge')([x, s])   # (batch, 160)
    merged = layers.Dense(64, activation='relu', kernel_regularizer=reg,
                          name='dense_head')(merged)
    merged = layers.Dropout(0.25)(merged)
    out    = layers.Dense(N_CLASSES, activation='softmax', name='output')(merged)

    return keras.Model([spec_inp, stats_inp], out)


def train_keras(X_tr, y_tr, X_va, y_va, X_te, y_te,
                S_tr, S_va, S_te):
    tf.random.set_seed(RANDOM_SEED)

    print(f"\nAugmentasyon (x4): {len(X_tr)} -> ", end='')
    X_tr_aug, S_tr_aug, y_tr_aug = augment_data(X_tr, S_tr, y_tr, factor=4)
    print(f"{len(X_tr_aug)} ornek")

    model = build_keras_model()
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(3e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy'],
    )
    callbacks = [
        keras.callbacks.EarlyStopping(patience=30, restore_best_weights=True,
                                       monitor='val_accuracy', min_delta=0.005),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=12,
                                           min_lr=1e-6, verbose=0),
    ]
    history = model.fit(
        [X_tr_aug, S_tr_aug], y_tr_aug,
        validation_data=([X_va, S_va], y_va),
        epochs=300, batch_size=64,
        callbacks=callbacks,
        verbose=1,
    )

    mpath = os.path.join(DATA_DIR, 'best_csi_model.keras')
    model.save(mpath)
    print(f"Model kaydedildi: {mpath}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    a1.plot(history.history['accuracy'],     label='Train')
    a1.plot(history.history['val_accuracy'], '--', label='Val')
    a1.set_title('Accuracy'); a1.legend(); a1.grid(alpha=0.3)
    a2.plot(history.history['loss'],     label='Train')
    a2.plot(history.history['val_loss'], '--', label='Val')
    a2.set_title('Loss'); a2.legend(); a2.grid(alpha=0.3)
    plt.suptitle('CNN + BiLSTM (Dual-Input) Egitim Gecmisi', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, 'training_history.png'), dpi=150, bbox_inches='tight')
    plt.close()

    return model


# ═══════════════════════════════════════════════════════════════════════════════
# 9b. PyTorch Fallback: CSIBiLSTM (sadece spectrogram yolu)
# ═══════════════════════════════════════════════════════════════════════════════
if not USE_KERAS:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class SpecDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X.transpose(0, 3, 1, 2), dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.long)

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            return self.X[i], self.y[i]

    class CSIBiLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Sequential(
                nn.Conv2d(N_PCA, 16, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(16), nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 1)),
                nn.Dropout2d(0.20),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32), nn.ReLU(),
                nn.Dropout2d(0.20),
            )
            self.bilstm = nn.LSTM(
                input_size=32, hidden_size=32,
                num_layers=1, batch_first=True,
                bidirectional=True, dropout=0.0,
            )
            self.head = nn.Sequential(
                nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.35),
                nn.Linear(32, N_CLASSES),
            )

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = x.mean(dim=2)
            x = x.permute(0, 2, 1)
            _, (h_n, _) = self.bilstm(x)
            x = torch.cat([h_n[0], h_n[1]], dim=1)
            return self.head(x)

    def _augment_pt(X, y, factor=4):
        rng = np.random.default_rng(RANDOM_SEED)
        aug_X, aug_y = [X], [y]
        for _ in range(factor - 1):
            noise = rng.normal(0, 0.10, size=X.shape).astype(np.float32)
            scale = rng.uniform(0.85, 1.15, size=(len(X), 1, 1, 1)).astype(np.float32)
            aug_X.append((X + noise) * scale)
            aug_y.append(y.copy())
        Xa = np.concatenate(aug_X)
        ya = np.concatenate(aug_y)
        idx = rng.permutation(len(Xa))
        return Xa[idx], ya[idx]

    def _train_ep(model, loader, opt, crit):
        model.train()
        tl, tc = 0.0, 0
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out  = model(X)
            loss = crit(out, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item() * len(y)
            tc += (out.argmax(1) == y).sum().item()
        n = len(loader.dataset)
        return tl / n, tc / n

    @torch.no_grad()
    def _eval_ep(model, loader, crit):
        model.eval()
        tl, tc = 0.0, 0
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            out  = model(X)
            tl  += crit(out, y).item() * len(y)
            tc  += (out.argmax(1) == y).sum().item()
        n = len(loader.dataset)
        return tl / n, tc / n

    @torch.no_grad()
    def _preds_pt(model, loader):
        model.eval()
        return np.concatenate(
            [model(X.to(DEVICE)).argmax(1).cpu().numpy() for X, _ in loader]
        )

    def train_pytorch(X_tr, y_tr, X_va, y_va, X_te, y_te,
                      S_tr=None, S_va=None, S_te=None):
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        print(f"\nAugmentasyon (x4): {len(X_tr)} -> ", end='')
        X_tr_aug, y_tr_aug = _augment_pt(X_tr, y_tr, factor=4)
        print(f"{len(X_tr_aug)} ornek")

        counts = np.bincount(y_tr_aug)
        w      = torch.tensor(
            len(y_tr_aug) / (N_CLASSES * counts), dtype=torch.float32
        ).to(DEVICE)
        crit   = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)

        tr_dl = DataLoader(SpecDataset(X_tr_aug, y_tr_aug), batch_size=32, shuffle=True)
        va_dl = DataLoader(SpecDataset(X_va, y_va), batch_size=64, shuffle=False)
        te_dl = DataLoader(SpecDataset(X_te, y_te), batch_size=64, shuffle=False)

        model = CSIBiLSTM().to(DEVICE)
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nModel: CSIBiLSTM  |  Parametre: {total:,}  |  Device: {DEVICE}")

        opt   = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
        sched = TorchReduceLR(opt, mode='min', factor=0.5, patience=10, min_lr=1e-6)
        mpath = os.path.join(DATA_DIR, 'best_csi_model.pth')
        N_EP, PATIENCE = 200, 25
        best_vl = float('inf'); pat = 0
        hist = {'ta': [], 'va': [], 'tl': [], 'vl': []}

        print("=" * 70)
        print("Egitim  (AdamW + ReduceLR + LabelSmoothing + EarlyStopping)")
        print("=" * 70)
        for ep in range(1, N_EP + 1):
            tl, ta = _train_ep(model, tr_dl, opt, crit)
            vl, va = _eval_ep(model, va_dl, crit)
            sched.step(vl)
            hist['tl'].append(tl); hist['vl'].append(vl)
            hist['ta'].append(ta); hist['va'].append(va)
            if ep % 5 == 0 or ep <= 10:
                print(f"Ep {ep:3d}/{N_EP}  tr={tl:.4f}/{ta:.4f}  vl={vl:.4f}/{va:.4f}")
            if vl < best_vl - 0.001:
                best_vl = vl; pat = 0; torch.save(model.state_dict(), mpath)
            else:
                pat += 1
                if pat >= PATIENCE:
                    print(f"\nEarlyStopping: epoch {ep}"); break

        model.load_state_dict(torch.load(mpath, map_location=DEVICE))

        fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
        a1.plot(hist['ta'], label='Train', lw=2)
        a1.plot(hist['va'], '--', label='Val', lw=2)
        a1.set_title('Accuracy'); a1.legend(); a1.grid(alpha=0.3)
        a2.plot(hist['tl'], label='Train', lw=2)
        a2.plot(hist['vl'], '--', label='Val', lw=2)
        a2.set_title('Loss'); a2.legend(); a2.grid(alpha=0.3)
        plt.suptitle('CNN + BiLSTM Egitim Gecmisi (PyTorch)', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(DATA_DIR, 'training_history.png'), dpi=150, bbox_inches='tight')
        plt.close()

        return model, te_dl


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Degerlendirme
# ═══════════════════════════════════════════════════════════════════════════════
def evaluate(y_true, y_pred):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_true, y_pred,    average='weighted', zero_division=0)
    f1   = f1_score(y_true, y_pred,        average='weighted', zero_division=0)

    print("\n" + "=" * 70)
    print(f"Test Sonuclari  --  CNN + BiLSTM Dual-Input  (BNR Top-{TOP_N_SC} SC)")
    print("=" * 70)
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(classification_report(y_true, y_pred,
                                target_names=CLASS_NAMES, zero_division=0))

    cm   = confusion_matrix(y_true, y_pred)
    cm_n = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(cm,   annot=True, fmt='d',   cmap='Blues',  ax=ax[0],
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    sns.heatmap(cm_n, annot=True, fmt='.1%', cmap='Greens', ax=ax[1],
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    for a in ax:
        a.set_ylabel('Gercek')
        a.set_xlabel('Tahmin')
    fig.suptitle(
        f'CNN + BiLSTM Dual-Input  |  BNR Top-{TOP_N_SC} SC  |  Acc:{acc:.2%}  F1:{f1:.4f}',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    path = os.path.join(DATA_DIR, 'confusion_matrix.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Karmasiklik matrisi: {path}")
    return acc, f1


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Ana Akis
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    X_tr, X_va, X_te, S_tr, S_va, S_te, y_tr, y_va, y_te = build_dataset()

    if USE_KERAS:
        model  = train_keras(X_tr, y_tr, X_va, y_va, X_te, y_te,
                             S_tr, S_va, S_te)
        y_pred = np.argmax(model.predict([X_te, S_te], verbose=0), axis=1)
    else:
        model, te_dl = train_pytorch(X_tr, y_tr, X_va, y_va, X_te, y_te,
                                     S_tr, S_va, S_te)
        y_pred       = _preds_pt(model, te_dl)

    evaluate(y_te, y_pred)

    ext = 'keras' if USE_KERAS else 'pth'
    print(f"\nCikti dosyalari:")
    print(f"  confusion_matrix.png")
    print(f"  training_history.png")
    print(f"  spectrogram_by_class.png")
    print(f"  best_csi_model.{ext}")
    print("=" * 70 + "\nTAMAMLANDI.\n" + "=" * 70)


if __name__ == '__main__':
    main()
