# -*- coding: utf-8 -*-
"""
CSI Breath Analysis - ESP32 Wi-Fi
Pipeline: Trim -> Hampel -> Savitzky-Golay -> Bandpass -> PCA -> FFT / Welch
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (saves file without display)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ───────────────────────────────────────────────────────────
# PARAMETRELER
# ───────────────────────────────────────────────────────────
TRIM_SEC      = 5       # baştan ve sondan kırpılacak saniye
BP_LOW        = 0.05    # Hz – bandpass alt kesim
BP_HIGH       = 0.40    # Hz – bandpass üst kesim
BP_ORDER      = 4       # Butterworth mertebesi
SG_WINDOW     = 51      # Savitzky-Golay pencere uzunluğu (tek sayı)
SG_ORDER      = 3       # Savitzky-Golay polinom mertebesi
HAMPEL_HALF   = 10      # Hampel filtresi yarı-pencere (örnek sayısı)
HAMPEL_NSIG   = 3.0     # Hampel filtresi n-sigma eşiği

DATA_DIR = Path(r"c:/Users/X1 Carbon 8 i5 LTE/Desktop/ehb/ehb440/experiment")
FILES = [
    "csi_data_empty_room.csv",
    "csi_data_5bpm_breath.csv",
    "csi_data_10bpm_breath.csv",
]
LABELS = {
    "csi_data_empty_room.csv"  : "Dosya 1 — Boş Oda (Referans)",
    "csi_data_5bpm_breath.csv" : "Dosya 2 — Nefes Verisi A",
    "csi_data_10bpm_breath.csv": "Dosya 3 — Nefes Verisi B",
}

# ───────────────────────────────────────────────────────────
# YARDIMCI FONKSİYONLAR
# ───────────────────────────────────────────────────────────

def parse_csi_amplitude(csi_str: str) -> np.ndarray:
    """
    CSI veri dizisini parse et, her alt-taşıyıcının genliğini döndür.
    Format: [I0,Q0, I1,Q1, ... I127,Q127]  (256 int8 değer)
    İlk 2 çift (4 değer) anormal (pilot / eğitim sembolü) → atlanır.
    """
    vals = np.fromstring(csi_str.strip("[] "), dtype=np.int16, sep=",")
    # İlk 4 değeri atla (75,-80,4,0 gibi anormal çiftler)
    vals = vals[4:]
    I = vals[0::2].astype(np.float32)
    Q = vals[1::2].astype(np.float32)
    return np.sqrt(I**2 + Q**2)


def hampel_filter(x: np.ndarray, half_win: int = 10, n_sig: float = 3.0) -> np.ndarray:
    """Hampel filtresi — kayan medyan ile aykırı değerleri değiştirir."""
    n = len(x)
    out = x.copy()
    k = 1.4826  # MAD → std tutarlılık katsayısı
    for i in range(n):
        lo = max(0, i - half_win)
        hi = min(n, i + half_win + 1)
        win = x[lo:hi]
        med = np.median(win)
        mad = np.median(np.abs(win - med))
        thr = n_sig * k * mad
        if thr > 0 and abs(x[i] - med) > thr:
            out[i] = med
    return out


def apply_bandpass(x: np.ndarray, fs: float,
                   low: float = BP_LOW, high: float = BP_HIGH,
                   order: int = BP_ORDER) -> np.ndarray:
    """Sıfır fazlı Butterworth bant geçiren filtre."""
    nyq = fs / 2.0
    b, a = signal.butter(order, [low / nyq, high / nyq], btype="band")
    return signal.filtfilt(b, a, x)


def welch_psd(x: np.ndarray, fs: float,
              low: float = BP_LOW, high: float = BP_HIGH):
    """Welch yöntemiyle PSD, [low, high] Hz aralığını döndürür."""
    # nperseg: iyi frekans çözünürlüğü için en az 2 × (1/low) örnek
    nperseg = min(len(x), max(256, int(2 * fs / low)))
    f, p = signal.welch(x, fs=fs, nperseg=nperseg,
                        noverlap=nperseg // 2, window="hann")
    mask = (f >= low) & (f <= high)
    return f[mask], p[mask]


def dominant_bpm(f: np.ndarray, p: np.ndarray):
    """En yüksek PSD tepesinin frekansını ve BPM'ini döndür."""
    idx = np.argmax(p)
    return f[idx], f[idx] * 60.0


# ───────────────────────────────────────────────────────────
# ANA İŞLEM DÖNGÜSÜ
# ───────────────────────────────────────────────────────────
results = {}

for fname in FILES:
    fpath = DATA_DIR / fname
    print(f"\n{'='*62}")
    print(f"  {fname}")
    print(f"{'='*62}")

    # 1. VERİ OKUMA
    df = pd.read_csv(fpath, parse_dates=["timestamp"])
    df = (df.dropna(subset=["csi_data"])
            .sort_values("timestamp")
            .reset_index(drop=True))
    t0 = df["timestamp"].iloc[0]
    df["t"] = (df["timestamp"] - t0).dt.total_seconds()

    dt_med = df["t"].diff().median()
    Fs = 1.0 / dt_med
    print(f"  Toplam örnek : {len(df)}  |  Süre: {df['t'].iloc[-1]:.1f} s  |  Fs ≈ {Fs:.2f} Hz")

    # 2. KIRPMA (trim)
    mask = (df["t"] >= TRIM_SEC) & (df["t"] <= df["t"].iloc[-1] - TRIM_SEC)
    df = df[mask].reset_index(drop=True)
    df["t"] = df["t"] - df["t"].iloc[0]
    print(f"  Kırpma sonrası: {len(df)} örnek  |  {df['t'].iloc[-1]:.1f} s")

    # 3. CSI GENLİKLERİNİ PARSE ET  →  [N_örnekler × N_taşıyıcı]
    amp_matrix = np.stack(df["csi_data"].map(parse_csi_amplitude).values)

    # Sıfır taşıyıcıları (null subcarrier) kaldır
    nonzero_mask = np.any(amp_matrix > 0.5, axis=0)
    amp_matrix = amp_matrix[:, nonzero_mask]
    print(f"  Aktif alt-taşıyıcı: {amp_matrix.shape[1]}")

    # 4. GÜRÜLTÜ TEMİZLEME — Hampel + Savitzky-Golay
    n_samples, n_sc = amp_matrix.shape
    denoised = amp_matrix.astype(np.float64).copy()

    # Savitzky-Golay penceresi örnek sayısından büyük olamaz
    sg_win = SG_WINDOW
    if sg_win >= n_samples:
        sg_win = n_samples - 1 if (n_samples - 1) % 2 == 1 else n_samples - 2

    print(f"  Hampel + S-G filtresi uygulanıyor ({n_sc} taşıyıcı)…")
    for j in range(n_sc):
        col = denoised[:, j]
        col = hampel_filter(col, half_win=HAMPEL_HALF, n_sig=HAMPEL_NSIG)
        col = signal.savgol_filter(col, window_length=sg_win, polyorder=SG_ORDER)
        denoised[:, j] = col

    # 5. BANT GEÇİREN FİLTRE (her taşıyıcıya ayrı ayrı)
    print(f"  Bandpass (0.05–0.40 Hz) uygulanıyor…")
    bp_matrix = np.zeros_like(denoised)
    for j in range(n_sc):
        bp_matrix[:, j] = apply_bandpass(denoised[:, j], Fs)

    # 6. PCA — 1. Temel Bileşen
    X_sc = StandardScaler().fit_transform(bp_matrix)
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(X_sc)[:, 0]
    explained_var = pca.explained_variance_ratio_[0] * 100
    print(f"  PC1 varyans açıklama oranı: {explained_var:.1f}%")

    # PC1'e de bandpass uygula (emin olmak için)
    pc1_bp = apply_bandpass(pc1, Fs)

    # 7. WELCH PSD + DOMINANT TEPE
    wf, wp = welch_psd(pc1_bp, Fs)
    peak_hz, peak_bpm = dominant_bpm(wf, wp)
    print(f"  ► Dominant tepe : {peak_hz:.4f} Hz  →  {peak_bpm:.1f} BPM")

    # FFT (karşılaştırma için)
    N   = len(pc1_bp)
    win = np.hanning(N)
    ff  = fftfreq(N, d=1.0 / Fs)
    sp  = np.abs(fft(pc1_bp * win))
    pos = (ff > 0) & (ff >= BP_LOW) & (ff <= BP_HIGH)
    fft_hz  = ff[pos][np.argmax(sp[pos])]
    fft_bpm = fft_hz * 60.0
    print(f"  ► FFT dominant  : {fft_hz:.4f} Hz  →  {fft_bpm:.1f} BPM")

    results[fname] = {
        "t"         : df["t"].values,
        "pc1"       : pc1_bp,
        "Fs"        : Fs,
        "wf"        : wf,
        "wp"        : wp,
        "peak_hz"   : peak_hz,
        "peak_bpm"  : peak_bpm,
        "fft_bpm"   : fft_bpm,
        "exp_var"   : explained_var,
    }


# ───────────────────────────────────────────────────────────
# GÖRSELLEŞTİRME
# ───────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 13))
fig.patch.set_facecolor("#0D1117")

fig.suptitle(
    "CSI Tabanlı Nefes Analizi — ESP32 Wi-Fi (Tek Anten)\n"
    "Pipeline: Kırpma (5 s) → Hampel → Savitzky-Golay → "
    "Bandpass (0.05–0.40 Hz) → PCA (PC1) → Welch PSD",
    fontsize=13, fontweight="bold", color="white", y=0.99,
)

gs = gridspec.GridSpec(len(FILES), 2, figure=fig, hspace=0.50, wspace=0.30,
                       left=0.06, right=0.97, top=0.93, bottom=0.06)

C_TIME  = "#4FC3F7"
C_FREQ  = "#EF5350"
C_PEAK  = "#FFB300"
C_BG    = "#161B22"
C_GRID  = "#30363D"
C_5BPM  = "#69F0AE"
C_10BPM = "#CE93D8"

for row_i, fname in enumerate(FILES):
    r   = results[fname]
    lbl = LABELS[fname]
    Fs  = r["Fs"]

    # ── Zaman serisi ──────────────────────────────────────
    ax_t = fig.add_subplot(gs[row_i, 0])
    ax_t.set_facecolor(C_BG)
    ax_t.plot(r["t"], r["pc1"], color=C_TIME, linewidth=0.8, alpha=0.9)
    ax_t.set_title(
        f"{lbl}\nPC1 Nefes Sinyali   (Fs ≈ {Fs:.1f} Hz,  PC1 varyans: {r['exp_var']:.1f}%)",
        fontsize=9, fontweight="bold", color="white", pad=5,
    )
    ax_t.set_xlabel("Zaman (saniye)", fontsize=8, color="#AAAAAA")
    ax_t.set_ylabel("Genlik (a.u.)", fontsize=8, color="#AAAAAA")
    ax_t.tick_params(colors="#AAAAAA", labelsize=7)
    for spine in ax_t.spines.values():
        spine.set_edgecolor(C_GRID)
    ax_t.grid(True, color=C_GRID, linewidth=0.5, alpha=0.7)

    # ── Welch PSD ─────────────────────────────────────────
    ax_f = fig.add_subplot(gs[row_i, 1])
    ax_f.set_facecolor(C_BG)

    bpm_axis = r["wf"] * 60.0
    ax_f.plot(bpm_axis, r["wp"], color=C_FREQ, linewidth=1.3, alpha=0.9)

    # Referans çizgileri
    ax_f.axvline(5,  color=C_5BPM,  linestyle=":", linewidth=1.2, alpha=0.7, label="5 BPM ref")
    ax_f.axvline(10, color=C_10BPM, linestyle=":", linewidth=1.2, alpha=0.7, label="10 BPM ref")

    # Dominant tepe
    peak_bpm_x = r["peak_hz"] * 60.0
    peak_y     = r["wp"][np.argmax(r["wp"])]
    ax_f.axvline(peak_bpm_x, color=C_PEAK, linestyle="--", linewidth=1.5, alpha=0.85)
    ax_f.plot(peak_bpm_x, peak_y, "o", color=C_PEAK, markersize=9, zorder=6,
              label=f"Peak: {r['peak_bpm']:.1f} BPM")
    ax_f.annotate(
        f" {r['peak_bpm']:.1f} BPM\n ({r['peak_hz']:.3f} Hz)",
        xy=(peak_bpm_x, peak_y),
        xytext=(peak_bpm_x + 1.5, peak_y * 0.85),
        fontsize=8, color=C_PEAK, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_PEAK, lw=1.0),
    )

    ax_f.set_title(
        f"Güç Spektral Yoğunluğu (Welch)\n"
        f"Dominant Tepe: {r['peak_bpm']:.1f} BPM  |  FFT: {r['fft_bpm']:.1f} BPM",
        fontsize=9, fontweight="bold", color="white", pad=5,
    )
    ax_f.set_xlabel("Frekans (BPM)", fontsize=8, color="#AAAAAA")
    ax_f.set_ylabel("Güç (a.u.²/Hz)", fontsize=8, color="#AAAAAA")
    ax_f.set_xlim([BP_LOW * 60, BP_HIGH * 60])
    ax_f.legend(fontsize=7.5, facecolor=C_BG, labelcolor="white", edgecolor=C_GRID)
    ax_f.tick_params(colors="#AAAAAA", labelsize=7)
    for spine in ax_f.spines.values():
        spine.set_edgecolor(C_GRID)
    ax_f.grid(True, color=C_GRID, linewidth=0.5, alpha=0.7)

out_path = DATA_DIR / "csi_breath_analysis.png"
plt.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\nGrafik kaydedildi: {out_path}")

# ───────────────────────────────────────────────────────────
# SONUÇ ÖZETİ
# ───────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  SONUC OZETI")
print("=" * 62)
for fname in FILES:
    r    = results[fname]
    bpm  = r["peak_bpm"]
    tag  = "≈ 5 BPM" if abs(bpm - 5) < abs(bpm - 10) else "≈ 10 BPM"
    if "empty" in fname:
        tag = "(boş oda / referans)"
    print(f"\n  {LABELS[fname]}")
    print(f"    Fs            : {r['Fs']:.2f} Hz")
    print(f"    PC1 varyans   : {r['exp_var']:.1f}%")
    print(f"    Welch dominant: {r['peak_bpm']:.1f} BPM  ({r['peak_hz']:.4f} Hz)  → {tag}")
    print(f"    FFT dominant  : {r['fft_bpm']:.1f} BPM")
print("\n" + "=" * 62)
