#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSI 1D-CNN + BiLSTM  (Ham PCA Zaman Serisi Girisi)
===================================================
STFT spectrogram yerine dogrudan PCA zaman serisini (260, 3) girdi olarak kullanir.

Avantaj:
  - Genlik (amplitude) bilgisi korunur  -- STFT'de kayboluyor
  - Nefes dalga formu sekli (morfology) yakalanir
  - Frekans + genlik + sekil: uc boyutlu bilgi

Dusunce: her iki nefes sinifi ~28 BPM olsa da derin nefes vs yuzeysel nefes
morfological farkliligi zaman serisinde gorunur:
  - Slow_Breath (uzun nefes) -> derin -> buyuk tepe-cukur amplitud
  - Fast_Breath (25bpm)      -> yuzeysel -> kucuk amplitud
  - Empty_Room               -> nefes yok -> cok kucuk ve duzensiz varyasyon

Pipeline:
  BNR Top-20 SC secimi -> DSP (Hampel+SG+BP) -> PCA(3)
  -> Normalize (z-score)
  -> 1D-CNN (yerel temporal ozellikler)
  -> BiLSTM (global temporal dinamikler)
  -> Softmax(3)

Karsilastirma: csi_classifier.py (STFT + Stats Dual-Input) -> %50.82 test accuracy
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

from scipy.signal import butter, filtfilt, savgol_filter, medfilt, welch
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
# Sabitler  (csi_classifier.py ile ayni DSP/BNR parametreleri)
# ═══════════════════════════════════════════════════════════════════════════════
FS          = 13
WINDOW_S    = 20
STEP_S      = 1
WINDOW_N    = WINDOW_S * FS    # = 260
STEP_N      = STEP_S   * FS   # = 13

TOP_N_SC    = 20
BNR_LO      = 0.16
BNR_HI      = 0.60

HAMPEL_HALF_G = 50
HAMPEL_NSIG_G = 2.5
SG_WIN_G      = 11
SG_ORD        = 3

BP_LO       = 0.10
BP_HI       = 0.60
HAMPEL_HALF_W = 5
HAMPEL_NSIG_W = 3.0

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

_nyq = FS / 2.0
_B, _A = butter(4, [BP_LO / _nyq, BP_HI / _nyq], btype='band')

print(f"\n1D-CNN + BiLSTM  |  Giris: Ham PCA Zaman Serisi ({WINDOW_N}, {N_PCA})")
print(f"DSP: Hampel({HAMPEL_HALF_W}) + SG(11,3) + BP({BP_LO}-{BP_HI} Hz)")
print(f"BNR Top-{TOP_N_SC} SC  |  {WINDOW_S}s pencere  |  {STEP_S}s adim")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CSI Yukleme
# ═══════════════════════════════════════════════════════════════════════════════
def _parse_amp(csi_str):
    try:
        raw  = np.array(ast.literal_eval(str(csi_str).strip()), dtype=np.float32)
        vals = raw[4:]
        n    = (len(vals) // 2) * 2
        return np.sqrt(vals[0:n:2] ** 2 + vals[1:n:2] ** 2)
    except Exception:
        return None


def load_amps(filepath, label):
    print(f"  [{label}] {os.path.basename(filepath)}")
    df   = pd.read_csv(filepath)
    amps = [_parse_amp(s) for s in df['csi_data']]
    amps = [a for a in amps if a is not None]
    min_len = min(len(a) for a in amps)
    mat = np.stack([a[:min_len] for a in amps])
    print(f"           {mat.shape[0]} paket, {mat.shape[1]} subcarrier")
    return mat


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Zamansal Bolme (%70 / %15 / %15)
# ═══════════════════════════════════════════════════════════════════════════════
def temporal_split(mat, tr=0.70, va=0.15):
    n = len(mat)
    return (mat[:int(n * tr)],
            mat[int(n * tr): int(n * (tr + va))],
            mat[int(n * (tr + va)):])


# ═══════════════════════════════════════════════════════════════════════════════
# 3. BNR Subcarrier Secimi
# ═══════════════════════════════════════════════════════════════════════════════
def _hampel_full(x, hw=50, k=2.5):
    ksize = 2 * hw + 1
    med   = medfilt(x.astype(np.float64), ksize)
    mad   = medfilt(np.abs(x - med), ksize)
    bad   = (mad > 0) & (np.abs(x - med) > k * 1.4826 * mad)
    out   = x.copy(); out[bad] = med[bad]
    return out


def _compute_bnr(sig):
    if sig.var() < 1e-6:
        return 0.0
    nperseg = min(len(sig), max(256, int(FS * 15)))
    f, p = welch(sig.astype(np.float64), fs=FS, nperseg=nperseg, scaling='density')
    in_b  = (f >= BNR_LO) & (f <= BNR_HI)
    out_b = ~in_b & (f > 0)
    if in_b.sum() == 0 or out_b.sum() == 0:
        return 0.0
    return float(trapezoid(p[in_b], f[in_b]) / (trapezoid(p[out_b], f[out_b]) + 1e-12))


def select_bnr_subcarriers(breath_train, n_select=TOP_N_SC):
    n_sc     = breath_train.shape[1]
    bnr_vals = np.zeros(n_sc)
    print(f"  BNR hesaplaniyor ({n_sc} SC, {len(breath_train)} paket)...", flush=True)
    for c in range(n_sc):
        sig = _hampel_full(breath_train[:, c].astype(np.float64))
        sig = savgol_filter(sig, window_length=SG_WIN_G, polyorder=SG_ORD)
        bnr_vals[c] = _compute_bnr(sig)
        if (c + 1) % 30 == 0 or c == n_sc - 1:
            print(f"    [{c+1}/{n_sc}] done", end='\r')
    print()
    top_idx = np.argsort(bnr_vals)[-n_select:][::-1]
    print(f"  Top-{n_select} BNR: min={bnr_vals[top_idx].min():.4f}  "
          f"max={bnr_vals[top_idx].max():.4f}")
    return top_idx


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Per-Window DSP
# ═══════════════════════════════════════════════════════════════════════════════
def _hampel_col(x, hw=HAMPEL_HALF_W, k=HAMPEL_NSIG_W):
    ksize = 2 * hw + 1
    med   = medfilt(x.astype(np.float64), ksize)
    mad   = medfilt(np.abs(x - med), ksize)
    bad   = (mad > 0) & (np.abs(x - med) > k * 1.4826 * mad)
    out   = x.copy(); out[bad] = med[bad]
    return out


def dsp_window(win):
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
# 5. Sliding Windows
# ═══════════════════════════════════════════════════════════════════════════════
def make_windows(mat, desc=''):
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
# 6. Per-Pencere z-score normalizasyon
# ═══════════════════════════════════════════════════════════════════════════════
def normalize_per_window(X):
    """
    X: (N, WINDOW_N, N_PCA)
    Her pencere ve her PC icin: z-score normalizasyon
    Amac: amplitud scale degisimini azalt, waveform sekline odaklan
    NOT: Bu normalizasyon amplitud bilgisini KALDIRIR.
    """
    mu  = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True) + 1e-8
    return ((X - mu) / std).astype(np.float32)


def normalize_global(X_tr, X_va, X_te):
    """
    Global z-score: amplitude bilgisini KORUR (cross-class differences).
    Sadece train verisi uzerinde fit edilir.
    """
    mu  = X_tr.mean(axis=(0, 1), keepdims=True)
    std = X_tr.std(axis=(0, 1), keepdims=True) + 1e-8
    return (
        ((X_tr - mu) / std).astype(np.float32),
        ((X_va - mu) / std).astype(np.float32),
        ((X_te - mu) / std).astype(np.float32),
        mu, std
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Veri Seti Olusturma
# ═══════════════════════════════════════════════════════════════════════════════
def build_dataset():
    print("\n" + "=" * 70)
    print(f"1D-CNN+BiLSTM  |  Ham PCA  |  BNR Top-{TOP_N_SC} SC  |  {WINDOW_S}s pencere")
    print("=" * 70)

    # 1. Yukle
    raw = {}
    for lbl, path in FILES.items():
        raw[lbl] = load_amps(path, lbl)

    min_sc = min(r.shape[1] for r in raw.values())
    for lbl in raw:
        raw[lbl] = raw[lbl][:, :min_sc]

    # 2. Zamansal bolme
    tr_mat, va_mat, te_mat = {}, {}, {}
    for lbl in CLASS_NAMES:
        tr_mat[lbl], va_mat[lbl], te_mat[lbl] = temporal_split(raw[lbl])
        print(f"  [{lbl}]: train={len(tr_mat[lbl])} val={len(va_mat[lbl])} "
              f"test={len(te_mat[lbl])} paket")

    # 3. Olu SC eleme
    tr_all = np.concatenate([tr_mat[lbl] for lbl in CLASS_NAMES], axis=0)
    alive  = np.var(tr_all, axis=0) > 1e-6
    print(f"\nAktif subcarrier: {alive.sum()} / {min_sc}")
    for lbl in CLASS_NAMES:
        tr_mat[lbl] = tr_mat[lbl][:, alive]
        va_mat[lbl] = va_mat[lbl][:, alive]
        te_mat[lbl] = te_mat[lbl][:, alive]

    # 4. BNR Subcarrier Secimi
    print(f"\nBNR Subcarrier Secimi...")
    breath_tr = np.concatenate([tr_mat['Slow_Breath'], tr_mat['Fast_Breath']], axis=0)
    sc_idx    = select_bnr_subcarriers(breath_tr, n_select=TOP_N_SC)
    for lbl in CLASS_NAMES:
        tr_mat[lbl] = tr_mat[lbl][:, sc_idx]
        va_mat[lbl] = va_mat[lbl][:, sc_idx]
        te_mat[lbl] = te_mat[lbl][:, sc_idx]

    # 5. Sliding Windows + DSP
    print(f"\nSliding Windows ({WINDOW_S}s, step={STEP_S}s) + DSP...")
    Xw_tr, Xw_va, Xw_te = {}, {}, {}
    for lbl in CLASS_NAMES:
        Xw_tr[lbl] = make_windows(tr_mat[lbl], desc=f'{lbl}/tr')
        Xw_va[lbl] = make_windows(va_mat[lbl], desc=f'{lbl}/va')
        Xw_te[lbl] = make_windows(te_mat[lbl], desc=f'{lbl}/te')
        print(f"  [{lbl}]: train={len(Xw_tr[lbl])} val={len(Xw_va[lbl])} "
              f"test={len(Xw_te[lbl])} pencere")

    # 6. Global PCA(3)
    print(f"\nGlobal PCA({N_PCA}) fit...")
    tr_flat = np.concatenate(
        [Xw_tr[l].reshape(-1, TOP_N_SC) for l in CLASS_NAMES if len(Xw_tr[l])], axis=0
    )
    scaler = StandardScaler()
    pca    = PCA(n_components=N_PCA, random_state=RANDOM_SEED)
    scaler.fit(tr_flat)
    pca.fit(scaler.transform(tr_flat))
    ev = pca.explained_variance_ratio_
    print(f"  Aciklanan varyans: {ev.sum()*100:.1f}%  "
          f"({', '.join(f'{v*100:.1f}%' for v in ev)})")

    def to_pca(wins_dict):
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

    pc_tr = to_pca(Xw_tr)
    pc_va = to_pca(Xw_va)
    pc_te = to_pca(Xw_te)

    # 7. Stack into arrays with labels
    def stack_with_labels(pc_dict):
        Xl, yl = [], []
        for idx, lbl in enumerate(CLASS_NAMES):
            W = pc_dict[lbl]
            if not len(W):
                continue
            Xl.append(W)
            yl.append(np.full(len(W), idx, dtype=np.int64))
        return np.concatenate(Xl), np.concatenate(yl)

    X_tr, y_tr = stack_with_labels(pc_tr)
    X_va, y_va = stack_with_labels(pc_va)
    X_te, y_te = stack_with_labels(pc_te)

    # 8. Global normalizasyon (amplitude bilgisini KORU)
    X_tr, X_va, X_te, _mu, _std = normalize_global(X_tr, X_va, X_te)

    # Per-class amplitude tani
    print(f"\nSinif bazi amplitude analizi (global norm. PCA varyans):")
    for idx, lbl in enumerate(CLASS_NAMES):
        mask = y_tr == idx
        if mask.sum() > 0:
            var_val = float(X_tr[mask].var(axis=1).mean())
            rms_val = float(np.sqrt((X_tr[mask]**2).mean(axis=1).mean()))
            print(f"  [{lbl:12s}] var={var_val:.4f}  rms={rms_val:.4f}")

    print(f"\nVeri seti (sekil: {X_tr.shape[1:]}):")
    print(f"  Train : {len(X_tr):4d}  {dict(zip(CLASS_NAMES, np.bincount(y_tr).tolist()))}")
    print(f"  Val   : {len(X_va):4d}  {dict(zip(CLASS_NAMES, np.bincount(y_va).tolist()))}")
    print(f"  Test  : {len(X_te):4d}  {dict(zip(CLASS_NAMES, np.bincount(y_te).tolist()))}")

    _plot_waveforms(pc_tr)
    return X_tr, X_va, X_te, y_tr, y_va, y_te


def _plot_waveforms(pc_tr):
    fig, axes = plt.subplots(N_PCA, N_CLASSES, figsize=(N_CLASSES * 5, N_PCA * 3))
    t_axis = np.arange(WINDOW_N) / FS
    for k in range(N_PCA):
        for j, lbl in enumerate(CLASS_NAMES):
            W = pc_tr[lbl]
            if not len(W):
                continue
            sample = W[len(W) // 2, :, k]
            ax = axes[k][j]
            ax.plot(t_axis, sample, lw=0.8, alpha=0.8)
            ax.set_title(f'{lbl} - PC{k+1}  (ort.)', fontsize=8)
            ax.set_xlabel('Zaman (s)')
            ax.set_ylabel('Amplitud')
            ax.grid(alpha=0.3)
    fig.suptitle(f'Ham PCA Zaman Serisi Ornekleri (Train | BNR Top-{TOP_N_SC} SC)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(DATA_DIR, 'waveform_by_class.png')
    plt.savefig(path, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"  Waveform grafigi: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Data Augmentation
# ═══════════════════════════════════════════════════════════════════════════════
def augment_data(X, y, factor=4):
    """
    X: (N, WINDOW_N, N_PCA) global-normalize edilmis zaman serisi
    Augmentasyon:
      - Gaussian gurultu: sigma=0.05 (zaman serisi icin daha kucuk)
      - Amplitude olcekleme: 0.85-1.15 (amplitude bilgisini biraz degistir)
      - Zaman kayma (shift): +/-2 sample
    """
    rng = np.random.default_rng(RANDOM_SEED)
    aug_X, aug_y = [X], [y]
    for _ in range(factor - 1):
        # Gurultu + olcekleme
        noise = rng.normal(0, 0.05, size=X.shape).astype(np.float32)
        scale = rng.uniform(0.88, 1.12, size=(len(X), 1, 1)).astype(np.float32)
        X_aug = (X + noise) * scale

        # Zaman kayma
        shift = rng.integers(-3, 4, size=len(X))
        X_aug_shifted = np.zeros_like(X_aug)
        for i, s in enumerate(shift):
            if s > 0:
                X_aug_shifted[i, s:, :] = X_aug[i, :-s, :]
            elif s < 0:
                X_aug_shifted[i, :s, :] = X_aug[i, -s:, :]
            else:
                X_aug_shifted[i] = X_aug[i]

        aug_X.append(X_aug_shifted)
        aug_y.append(y.copy())

    Xa  = np.concatenate(aug_X)
    ya  = np.concatenate(aug_y)
    idx = rng.permutation(len(Xa))
    return Xa[idx], ya[idx]


# ═══════════════════════════════════════════════════════════════════════════════
# 9a. Keras: 1D-CNN + BiLSTM
# ═══════════════════════════════════════════════════════════════════════════════
def build_keras_model():
    """
    Giris: (WINDOW_N=260, N_PCA=3) global-normalize PCA zaman serisi

    [1D-CNN Block 1] Conv1D(32, k=13) + BN + ReLU + MaxPool(2) -> (130, 32)
      13 sample = 1 saniye ~ tek nefes periyodunun 1/2.4 kadar kucuk fragment

    [1D-CNN Block 2] Conv1D(64, k=7) + BN + ReLU + MaxPool(2) -> (65, 64)
      7 sample = 0.54 saniye ~ lokal nefes deseni

    [1D-CNN Block 3] Conv1D(128, k=5) + BN + ReLU + MaxPool(2) -> (32, 128)
      Daha genis ozellikler

    [BiLSTM(64)] Zamansal dinamikler -> (batch, 128)
    [Dense(64)] + Dropout(0.25) -> [Softmax(3)]

    Toplam parametre: ~200K
    """
    reg = keras.regularizers.l2(1e-4)
    inp = keras.Input(shape=(WINDOW_N, N_PCA), name='timeseries')

    # Block 1: yerel nefes deseni (1s alici alan)
    x = layers.Conv1D(32, kernel_size=13, padding='same', use_bias=False)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)    # 260 -> 130
    x = layers.Dropout(0.10)(x)

    # Block 2: orta olcekli desen (0.54s)
    x = layers.Conv1D(64, kernel_size=7, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)    # 130 -> 65
    x = layers.Dropout(0.10)(x)

    # Block 3: uzun erimli nefes dalga deseni
    x = layers.Conv1D(128, kernel_size=5, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)    # 65 -> 32
    x = layers.Dropout(0.10)(x)

    # BiLSTM: zamansal nefes dinamikleri
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=False, dropout=0.10, recurrent_dropout=0.0),
        name='bilstm'
    )(x)
    # x: (batch, 128)

    x   = layers.Dense(64, activation='relu', kernel_regularizer=reg, name='head')(x)
    x   = layers.Dropout(0.25)(x)
    out = layers.Dense(N_CLASSES, activation='softmax', name='output')(x)

    return keras.Model(inp, out)


def train_keras(X_tr, y_tr, X_va, y_va, X_te, y_te):
    tf.random.set_seed(RANDOM_SEED)

    print(f"\nAugmentasyon (x4): {len(X_tr)} -> ", end='')
    X_tr_aug, y_tr_aug = augment_data(X_tr, y_tr, factor=4)
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
        X_tr_aug, y_tr_aug,
        validation_data=(X_va, y_va),
        epochs=300, batch_size=64,
        callbacks=callbacks,
        verbose=1,
    )

    mpath = os.path.join(DATA_DIR, 'best_1d_model.keras')
    model.save(mpath)
    print(f"Model kaydedildi: {mpath}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    a1.plot(history.history['accuracy'],     label='Train')
    a1.plot(history.history['val_accuracy'], '--', label='Val')
    a1.set_title('Accuracy (1D-CNN+BiLSTM)'); a1.legend(); a1.grid(alpha=0.3)
    a2.plot(history.history['loss'],     label='Train')
    a2.plot(history.history['val_loss'], '--', label='Val')
    a2.set_title('Loss'); a2.legend(); a2.grid(alpha=0.3)
    plt.suptitle('1D-CNN + BiLSTM Egitim Gecmisi (Ham PCA)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, 'training_history_1d.png'), dpi=150, bbox_inches='tight')
    plt.close()

    return model


# ═══════════════════════════════════════════════════════════════════════════════
# 9b. PyTorch Fallback
# ═══════════════════════════════════════════════════════════════════════════════
if not USE_KERAS:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class TSDataset(Dataset):
        def __init__(self, X, y):
            # X: (N, WINDOW_N, N_PCA) -> PyTorch: (N, N_PCA, WINDOW_N)
            self.X = torch.tensor(X.transpose(0, 2, 1), dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.long)

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            return self.X[i], self.y[i]

    class BiLSTM1D(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(N_PCA, 32, kernel_size=13, padding=6, bias=False),
                nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
                nn.Dropout(0.10),
                nn.Conv1d(32, 64, kernel_size=7, padding=3, bias=False),
                nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
                nn.Dropout(0.10),
                nn.Conv1d(64, 128, kernel_size=5, padding=2, bias=False),
                nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
                nn.Dropout(0.10),
            )
            self.bilstm = nn.LSTM(
                128, 64, num_layers=1, batch_first=True,
                bidirectional=True, dropout=0.0
            )
            self.head = nn.Sequential(
                nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.25),
                nn.Linear(64, N_CLASSES)
            )

        def forward(self, x):
            x = self.conv(x)           # (B, 128, T')
            x = x.permute(0, 2, 1)    # (B, T', 128)
            _, (h, _) = self.bilstm(x)
            x = torch.cat([h[0], h[1]], dim=1)
            return self.head(x)

    def _train_aug_pt(X, y, factor=4):
        rng = np.random.default_rng(RANDOM_SEED)
        aug_X, aug_y = [X], [y]
        for _ in range(factor - 1):
            noise = rng.normal(0, 0.05, size=X.shape).astype(np.float32)
            scale = rng.uniform(0.88, 1.12, size=(len(X), 1, 1)).astype(np.float32)
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
            out = model(X); loss = crit(out, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item() * len(y)
            tc += (out.argmax(1) == y).sum().item()
        return tl / len(loader.dataset), tc / len(loader.dataset)

    @torch.no_grad()
    def _eval_ep(model, loader, crit):
        model.eval()
        tl, tc = 0.0, 0
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            out = model(X)
            tl += crit(out, y).item() * len(y)
            tc += (out.argmax(1) == y).sum().item()
        return tl / len(loader.dataset), tc / len(loader.dataset)

    @torch.no_grad()
    def _preds_pt(model, loader):
        model.eval()
        return np.concatenate([model(X.to(DEVICE)).argmax(1).cpu().numpy() for X, _ in loader])

    def train_pytorch(X_tr, y_tr, X_va, y_va, X_te, y_te):
        torch.manual_seed(RANDOM_SEED)
        print(f"\nAugmentasyon (x4): {len(X_tr)} -> ", end='')
        X_aug, y_aug = _train_aug_pt(X_tr, y_tr, factor=4)
        print(f"{len(X_aug)} ornek")

        crit  = nn.CrossEntropyLoss(label_smoothing=0.05)
        tr_dl = DataLoader(TSDataset(X_aug, y_aug), batch_size=32, shuffle=True)
        va_dl = DataLoader(TSDataset(X_va, y_va), batch_size=64)
        te_dl = DataLoader(TSDataset(X_te, y_te), batch_size=64)

        model = BiLSTM1D().to(DEVICE)
        opt   = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
        sched = TorchReduceLR(opt, mode='min', factor=0.5, patience=10, min_lr=1e-6)
        mpath = os.path.join(DATA_DIR, 'best_1d_model.pth')
        best_vl = float('inf'); pat = 0

        for ep in range(1, 201):
            tl, ta = _train_ep(model, tr_dl, opt, crit)
            vl, va = _eval_ep(model, va_dl, crit)
            sched.step(vl)
            if ep % 5 == 0 or ep <= 10:
                print(f"Ep {ep:3d}  tr={ta:.4f}  vl={va:.4f}")
            if vl < best_vl - 0.001:
                best_vl = vl; pat = 0; torch.save(model.state_dict(), mpath)
            else:
                pat += 1
                if pat >= 25:
                    print(f"EarlyStopping: epoch {ep}"); break

        model.load_state_dict(torch.load(mpath, map_location=DEVICE))
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
    print(f"Test Sonuclari  --  1D-CNN + BiLSTM  (Ham PCA | BNR Top-{TOP_N_SC} SC)")
    print("=" * 70)
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(classification_report(y_true, y_pred,
                                target_names=CLASS_NAMES, zero_division=0))

    # STFT modeli ile karsilastirma
    baseline_stft = 0.5082
    print(f"  STFT+Stats Dual-Input (baseline): {baseline_stft*100:.2f}%")
    delta = acc - baseline_stft
    sign  = '+' if delta >= 0 else ''
    print(f"  1D-CNN degisim: {sign}{delta*100:.2f}%")

    cm   = confusion_matrix(y_true, y_pred)
    cm_n = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(cm,   annot=True, fmt='d',   cmap='Blues',  ax=ax[0],
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    sns.heatmap(cm_n, annot=True, fmt='.1%', cmap='Greens', ax=ax[1],
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    for a in ax:
        a.set_ylabel('Gercek'); a.set_xlabel('Tahmin')
    fig.suptitle(
        f'1D-CNN+BiLSTM (Ham PCA)  |  BNR Top-{TOP_N_SC} SC  |  Acc:{acc:.2%}  F1:{f1:.4f}',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    path = os.path.join(DATA_DIR, 'confusion_matrix_1d.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Karmasiklik matrisi: {path}")
    return acc, f1


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Ana Akis
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    X_tr, X_va, X_te, y_tr, y_va, y_te = build_dataset()

    if USE_KERAS:
        model  = train_keras(X_tr, y_tr, X_va, y_va, X_te, y_te)
        y_pred = np.argmax(model.predict(X_te, verbose=0), axis=1)
    else:
        model, te_dl = train_pytorch(X_tr, y_tr, X_va, y_va, X_te, y_te)
        y_pred       = _preds_pt(model, te_dl)

    evaluate(y_te, y_pred)

    ext = 'keras' if USE_KERAS else 'pth'
    print(f"\nCikti dosyalari:")
    print(f"  confusion_matrix_1d.png")
    print(f"  training_history_1d.png")
    print(f"  waveform_by_class.png")
    print(f"  best_1d_model.{ext}")
    print("=" * 70 + "\nTAMAMLANDI.\n" + "=" * 70)


if __name__ == '__main__':
    main()
