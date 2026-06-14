# -*- coding: utf-8 -*-
"""
RespirFi-tarzı CSI Analizi  —  Önce BNR Eleme, Sonra PCA
=========================================================
Anahtar değişiklik: BNR alt sınırı 0.10 → 0.16 Hz
  0.16 Hz = 9.6 BPM  →  HVAC/fan 8-10 BPM DIŞARIDA kalır
  0.41 Hz = 25 BPM   →  hedef sinyal IÇERIDE kalır

Pipeline:
  Trim → Hampel(half=50) → S-G(w=11)
  → BNR[0.16–0.60 Hz] per subcarrier → Top-20 seç
  → Bandpass(0.10–0.60 Hz) → PCA(3) → VN-SampEn → FFT BPM (+ Welch referans)
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from scipy import signal
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# PARAMETRELER
# ============================================================
TRIM_SEC      = 5

# Bandpass (sinyal filtreleme için): 0.10–0.60 Hz
BP_LOW        = 0.10
BP_HIGH       = 0.60
BP_ORDER      = 4

# BNR hesaplama bandı (subcarrier seçimi için): 0.16–0.60 Hz
# 0.16 Hz = 9.6 BPM → HVAC/fan (8-10 BPM) bu bandın DIŞINDA kalır
BNR_LO        = 0.16
BNR_HI        = 0.60

# Savitzky-Golay: window=11 < T_25bpm=29 örnek → 25 BPM periyodunu korur
SG_WINDOW     = 11
SG_ORDER      = 3

# Hampel: toplam pencere ≈ 100 örnek → hızlı tepeler outlier sayılmaz
HAMPEL_HALF   = 50
HAMPEL_NSIG   = 2.5

TOP_N_SC      = 20          # BNR'ye göre seçilecek en iyi subcarrier sayısı
N_PCS         = 3
SAMPEN_M      = 3
SAMPEN_R_COEF = 0.1
SAMPEN_N_MAX  = 400
FFT_NFFT      = 4096        # Sıfır-dolgu FFT → yüksek frekans çözünürlüğü

DATA_DIR = Path(r"c:/Users/X1 Carbon 8 i5 LTE/Desktop/ehb/ehb440/experiment")
FILES = [
    ("csi_data_empty_my_room.csv",  "Dosya 1 — Boş Oda (Referans)"),
    ("csi_data_long_breath.csv",    "Dosya 2 — Uzun Nefes (BPM bilinmiyor)"),
    ("csi_data_25bpm_breath.csv",   "Dosya 3 — 25 BPM Kontrollü Nefes"),
]

# ============================================================
# YARDIMCI FONKSİYONLAR
# ============================================================

def parse_csi_amplitude(csi_str: str) -> np.ndarray:
    vals = np.fromstring(str(csi_str).strip("[] "), dtype=np.int16, sep=",")
    vals = vals[4:]
    return np.sqrt(vals[0::2].astype(np.float32)**2 +
                   vals[1::2].astype(np.float32)**2)


def hampel_filter(x: np.ndarray, half_win: int = HAMPEL_HALF,
                  n_sig: float = HAMPEL_NSIG):
    """Outlier sil. (temizlenmiş, outlier_mask) döner."""
    k      = 1.4826
    padded = np.pad(x, half_win, mode="edge")
    wins   = sliding_window_view(padded, 2 * half_win + 1)
    meds   = np.median(wins, axis=1)
    mads   = np.median(np.abs(wins - meds[:, None]), axis=1)
    out    = x.copy()
    mask   = np.abs(x - meds) > n_sig * k * mads
    out[mask] = meds[mask]
    return out, mask


def bandpass_filter(x: np.ndarray, fs: float) -> np.ndarray:
    nyq  = fs / 2.0
    high = min(BP_HIGH, nyq * 0.99)
    b, a = signal.butter(BP_ORDER, [BP_LOW / nyq, high / nyq], btype="band")
    return signal.filtfilt(b, a, x)


def compute_bnr(x: np.ndarray, fs: float) -> float:
    """
    BNR = (BNR_LO–BNR_HI bandındaki PSD enerjisi) / (dışarıdaki PSD enerjisi)
    BNR_LO=0.16 Hz → HVAC 8-10 BPM dışarıda, 25 BPM içeride.
    """
    nperseg = min(len(x), max(256, int(fs * 15)))
    f, p    = signal.welch(x, fs=fs, nperseg=nperseg)
    in_b    = (f >= BNR_LO) & (f <= BNR_HI)
    out_b   = ~in_b & (f > 0)
    breath  = float(np.trapezoid(p[in_b],  f[in_b]))  if in_b.any()  else 0.0
    noise   = float(np.trapezoid(p[out_b], f[out_b])) if out_b.any() else 1e-10
    return breath / max(noise, 1e-10)


def dominant_bpm_fft(x: np.ndarray, fs: float,
                     nfft: int = FFT_NFFT) -> tuple:
    """
    Sıfır-dolgulu FFT ile dominant BPM hesaplar.
    Dönüş: (peak_bpm, peak_hz, fft_freqs_bpm, fft_mag_normalized)
    """
    n_use  = max(len(x), nfft)
    freqs  = np.fft.rfftfreq(n_use, d=1.0 / fs)
    mag    = np.abs(np.fft.rfft(x, n=n_use))
    mask   = (freqs >= BP_LOW) & (freqs <= BP_HIGH)
    if not np.any(mask):
        return 0.0, 0.0, np.array([]), np.array([])
    f_band = freqs[mask]
    m_band = mag[mask]
    idx    = int(np.argmax(m_band))
    peak_hz  = float(f_band[idx])
    peak_bpm = peak_hz * 60.0
    norm     = m_band / (m_band.max() + 1e-12)
    return peak_bpm, peak_hz, f_band * 60.0, norm


def sample_entropy(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if len(x) > SAMPEN_N_MAX:
        x = signal.resample(x, SAMPEN_N_MAX)
    r = SAMPEN_R_COEF * np.std(x, ddof=1)
    if r == 0:
        return np.inf
    N = len(x)

    def _count(dim):
        nT = N - dim
        if nT <= 1:
            return 0
        T = sliding_window_view(x, dim)[:nT]
        total = 0
        for i in range(nT):
            dist = np.max(np.abs(T - T[i]), axis=1)
            total += int(np.sum(dist < r)) - 1
        return total

    B, A = _count(SAMPEN_M), _count(SAMPEN_M + 1)
    return float(-np.log(A / B)) if B > 0 and A > 0 else np.inf


def welch_psd(x: np.ndarray, fs: float):
    """(f_band, p_band, peak_hz, peak_bpm) döner — referans görselleştirme."""
    nperseg = min(len(x), max(512, int(2 * fs / BP_LOW)))
    f, p    = signal.welch(x, fs=fs, nperseg=nperseg,
                           noverlap=nperseg // 2, window="hann")
    mask    = (f >= BP_LOW) & (f <= BP_HIGH)
    f_m, p_m = f[mask], p[mask]
    if len(p_m) == 0:
        return f_m, p_m, 0.0, 0.0
    idx = int(np.argmax(p_m))
    return f_m, p_m, float(f_m[idx]), float(f_m[idx] * 60.0)


def vn_sampen_select(pcs: np.ndarray, fs: float,
                     entropies: list, ev: np.ndarray):
    scores, bpms = [], []
    for k in range(len(entropies)):
        _, _, _, pbpm = welch_psd(pcs[:, k], fs)
        se  = entropies[k] if np.isfinite(entropies[k]) else 10.0
        var = max(ev[k] / 100.0, 1e-6)
        scores.append(se / var)
        bpms.append(pbpm)
    return int(np.argmin(scores)), bpms, scores


# ============================================================
# ANA ANALİZ DÖNGÜSÜ
# ============================================================
results = {}

for fname, label in FILES:
    fpath = DATA_DIR / fname
    print(f"\n{'='*72}")
    print(f"  {label}")
    print(f"{'='*72}")

    df = pd.read_csv(fpath, parse_dates=["timestamp"])
    df = (df.dropna(subset=["csi_data"])
            .sort_values("timestamp")
            .reset_index(drop=True))
    df["t"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
    Fs = 1.0 / df["t"].diff().median()
    print(f"  Örnekler: {len(df)}  Süre: {df['t'].iloc[-1]:.1f}s  Fs: {Fs:.2f} Hz")

    # 1. Trim
    mask_t = (df["t"] >= TRIM_SEC) & (df["t"] <= df["t"].iloc[-1] - TRIM_SEC)
    df     = df[mask_t].reset_index(drop=True)
    df["t"] = df["t"] - df["t"].iloc[0]
    print(f"  Kırpma sonrası: {len(df)} örnekler  {df['t'].iloc[-1]:.1f}s")

    # 2. CSI parse
    amp = np.stack(df["csi_data"].map(parse_csi_amplitude).values)
    amp = amp[:, np.any(amp > 0.5, axis=0)]
    n_s, n_sc = amp.shape
    print(f"  Aktif alt taşıyıcılar: {n_sc}")

    # Ham sinyal (görsel için en yüksek varyanslı subcarrier)
    raw_sub  = int(np.argmax(np.var(amp, axis=0)))
    raw_plot = amp[:, raw_sub].copy()
    _, out_mask_plot = hampel_filter(raw_plot.astype(np.float64))
    n_outliers = int(out_mask_plot.sum())
    print(f"  Outlier tespiti: {n_outliers} nokta ({100*n_outliers/n_s:.1f}%)")

    # ── ADIM 3: Hampel + S-G — tüm subcarrier'lara ──────────────────────────
    sg_w = SG_WINDOW
    while sg_w >= n_s:
        sg_w -= 2
    if sg_w % 2 == 0:
        sg_w -= 1
    sg_w = max(sg_w, SG_ORDER + 2)

    print(f"  Hampel(half={HAMPEL_HALF}) + S-G(w={sg_w}) uygulanıyor ({n_sc} subcarrier)...")
    denoised = np.empty_like(amp, dtype=np.float64)
    hamp_plot_col = None
    for j in range(n_sc):
        col, _ = hampel_filter(amp[:, j].astype(np.float64))
        denoised[:, j] = signal.savgol_filter(col, sg_w, SG_ORDER)
        if j == raw_sub:
            hamp_plot_col = denoised[:, j].copy()

    # ── ADIM 4: BNR[0.16–0.60 Hz] ile subcarrier eleme ──────────────────────
    print(f"  BNR[{BNR_LO}–{BNR_HI} Hz] hesaplanıyor ({n_sc} subcarrier)...")
    bnr_vals = np.array([compute_bnr(denoised[:, j], Fs) for j in range(n_sc)])

    # BNR dağılım istatistikleri
    top_n   = min(TOP_N_SC, n_sc)
    top_idx = np.argsort(bnr_vals)[-top_n:][::-1]   # büyükten küçüğe
    top_bnr = bnr_vals[top_idx]

    # Hangi subcarrier'lar 25 BPM bölgesinde dominant? (tanısal)
    n_above_hvac = int(np.sum(bnr_vals > np.median(bnr_vals)))
    print(f"  BNR istatistik  —  "
          f"min={bnr_vals.min():.3f}  median={np.median(bnr_vals):.3f}  "
          f"max={bnr_vals.max():.3f}")
    print(f"  Top-{top_n} BNR: [{top_bnr[-1]:.3f} – {top_bnr[0]:.3f}]")
    print(f"  Seçilen subcarrier indeksleri (global): {list(top_idx[:10])}...")

    # ── ADIM 5: Sadece top-20 ile devam ──────────────────────────────────────
    selected = denoised[:, top_idx]   # (N, top_n)

    # ── ADIM 6: Bandpass 0.10–0.60 Hz ────────────────────────────────────────
    print(f"  Bandpass Butterworth({BP_LOW}–{BP_HIGH} Hz)...")
    bp_mat = np.stack([bandpass_filter(selected[:, j], Fs)
                       for j in range(top_n)], axis=1)

    # ── ADIM 7: PCA → N_PCS bileşen ──────────────────────────────────────────
    X_sc = StandardScaler().fit_transform(bp_mat)
    pca  = PCA(n_components=min(N_PCS, top_n))
    pcs  = pca.fit_transform(X_sc)
    ev   = pca.explained_variance_ratio_ * 100
    print(f"  PCA(top-{top_n} subcarrier): " +
          "  ".join(f"PC{k+1}={ev[k]:.1f}%" for k in range(len(ev))))

    # ── ADIM 8: SampEn + VN-SampEn seçimi ────────────────────────────────────
    print(f"  SampEn hesaplanıyor...")
    entropies = []
    for k in range(pcs.shape[1]):
        se = sample_entropy(pcs[:, k])
        _, _, _, dom_bpm = welch_psd(pcs[:, k], Fs)
        entropies.append(se)
        se_str = f"{se:.4f}" if np.isfinite(se) else "inf"
        print(f"    PC{k+1}: SampEn={se_str}  Welch={dom_bpm:.1f} BPM  (var={ev[k]:.1f}%)")

    best_k, pc_bpms, vn_scores = vn_sampen_select(pcs, Fs, entropies, ev)
    print(f"  VN-SampEn: " +
          "  ".join(f"PC{k+1}={vn_scores[k]:.3f}" for k in range(len(vn_scores))))
    breath_sig = pcs[:, best_k]

    # ── ADIM 9: FFT BPM (sıfır-dolgulu, yüksek çözünürlük) ──────────────────
    fft_bpm, fft_hz, fft_f_bpm, fft_mag = dominant_bpm_fft(breath_sig, Fs)

    # Welch BPM (referans karşılaştırma)
    wf, wp, w_hz, w_bpm = welch_psd(breath_sig, Fs)

    se_best = entropies[best_k]
    se_str  = f"{se_best:.4f}" if np.isfinite(se_best) else "inf"
    print(f"\n  *** PC{best_k+1} seçildi  SampEn={se_str}  "
          f"VN-score={vn_scores[best_k]:.3f}")
    print(f"  *** FFT BPM  = {fft_bpm:.2f} BPM  ({fft_hz:.4f} Hz)  "
          f"[sıfır-dolgulu FFT, Δf={Fs/FFT_NFFT*60:.2f} BPM]")
    print(f"  *** Welch BPM= {w_bpm:.1f} BPM  (referans)")

    results[fname] = {
        "t":          df["t"].values,
        "raw":        raw_plot,
        "hamp":       hamp_plot_col,
        "out_mask":   out_mask_plot,
        "breath":     breath_sig,
        "Fs":         Fs,
        # Welch (referans PSD görselleştirme)
        "wf": wf, "wp": wp,
        "w_bpm": w_bpm, "w_hz": w_hz,
        # FFT (ana BPM)
        "fft_bpm":    fft_bpm,
        "fft_hz":     fft_hz,
        "fft_f_bpm":  fft_f_bpm,
        "fft_mag":    fft_mag,
        # Meta
        "best_k":     best_k,
        "ev":         ev,
        "entropies":  entropies,
        "vn_scores":  vn_scores,
        "label":      label,
        "n_outliers": n_outliers,
        "top_bnr":    top_bnr[0],
        "bnr_vals":   bnr_vals,
        "top_idx":    top_idx,
    }


# ============================================================
# ÖZET TABLO
# ============================================================
W = 88
print(f"\n{'='*W}")
print(f"  {'ÖZET TABLO':^{W-4}}")
print(f"{'='*W}")
hdr = (f"  {'Dosya':<32}  {'Fs':>5}  {'Out':>5}  "
       f"{'BNR_top':>7}  {'PC':>3}  {'SampEn':>7}  "
       f"{'FFT BPM':>8}  {'Welch':>6}")
print(hdr)
print(f"  {'-'*(W-4)}")
for fname, label in FILES:
    r  = results[fname]
    k  = r["best_k"]
    se = r["entropies"][k]
    se_s = f"{se:.3f}" if np.isfinite(se) else "  inf"
    print(f"  {label:<32}  {r['Fs']:>5.2f}  {r['n_outliers']:>5}  "
          f"{r['top_bnr']:>7.3f}  PC{k+1}  "
          f"{se_s:>7}  "
          f"{r['fft_bpm']:>8.2f}  {r['w_bpm']:>6.1f}")
print(f"{'='*W}")
print(f"  Hedef: 25 BPM = 0.4167 Hz")
print(f"  BNR bandı: {BNR_LO}–{BNR_HI} Hz  (HVAC 8-10 BPM dışarıda)")
print(f"  FFT Δf: {results[FILES[0][0]]['Fs']/FFT_NFFT*60:.3f} BPM çözünürlük")


# ============================================================
# GÖRSELLEŞTİRME — 3 satır × 3 sütun
# Sol: sinyal | Orta: BNR dağılımı | Sağ: FFT BPM spektrumu
# ============================================================
DARK = "#0D1117"; BG = "#161B22"; GRID = "#30363D"
C_RAW   = "#FF6B35"
C_HAMP  = "#FFD700"
C_CLEAN = "#4FC3F7"
C_OUTL  = "#FF0055"
C_FFT   = "#58A6FF"   # mavi — FFT spektrumu
C_WELCH = "#EF5350"   # kırmızı — Welch referans
C_PEAK  = "#FFB300"
C_BNR_H = "#39D353"   # yeşil — seçilen subcarrier'lar
C_BNR_L = "#484F58"   # gri   — elenen subcarrier'lar

fig = plt.figure(figsize=(26, 17))
fig.patch.set_facecolor(DARK)
fig.suptitle(
    "RespirFi-Tarzı CSI Analizi  —  Önce BNR[0.16–0.60 Hz] Eleme, Sonra PCA\n"
    f"Hampel(half={HAMPEL_HALF}) → S-G(w={SG_WINDOW}) → "
    f"BNR top-{TOP_N_SC} → Bandpass(0.10–0.60 Hz) → PCA(3) → VN-SampEn → FFT BPM\n"
    f"Hedef: 25 BPM = 0.4167 Hz  |  BNR alt kesimi 0.16 Hz → HVAC 8-10 BPM elenmiş",
    fontsize=10, fontweight="bold", color="white", y=0.999
)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.65, wspace=0.30,
                       left=0.05, right=0.97, top=0.920, bottom=0.04)

def znorm(x):
    s = np.std(x)
    return (x - np.mean(x)) / (s if s > 0 else 1.0)

for ri, (fname, _) in enumerate(FILES):
    r      = results[fname]
    t      = r["t"]
    k      = r["best_k"]
    se     = r["entropies"][k]
    se_str = f"{se:.3f}" if np.isfinite(se) else "inf"
    n_sc_total = len(r["bnr_vals"])

    # ── Sol: Sinyal katmanları ────────────────────────────────────────────────
    ax_t = fig.add_subplot(gs[ri, 0])
    ax_t.set_facecolor(BG)
    raw_z    = znorm(r["raw"])
    hamp_z   = znorm(r["hamp"]) if r["hamp"] is not None else raw_z
    breath_z = znorm(r["breath"])

    ax_t.plot(t, raw_z,    color=C_RAW,   lw=0.5, alpha=0.40,
              label="Ham CSI")
    ax_t.plot(t, hamp_z,   color=C_HAMP,  lw=0.8, alpha=0.60,
              label=f"Hampel+S-G ({r['n_outliers']} outlier)")
    ax_t.plot(t, breath_z, color=C_CLEAN, lw=1.8, alpha=0.95,
              label=f"Nefes (PC{k+1}, SE={se_str})")
    if r["out_mask"].any():
        ax_t.scatter(t[r["out_mask"]], raw_z[r["out_mask"]],
                     color=C_OUTL, s=12, zorder=6, alpha=0.80,
                     label=f"Outlier ({r['out_mask'].sum()})")

    ax_t.set_title(
        f"{r['label']}\n"
        f"FFT={r['fft_bpm']:.1f} BPM  |  PC{k+1} var={r['ev'][k]:.1f}%  |  "
        f"BNR_top={r['top_bnr']:.3f}",
        fontsize=8.5, fontweight="bold", color="white", pad=4
    )
    ax_t.set_xlabel("Zaman (s)", fontsize=7.5, color="#AAAAAA")
    ax_t.set_ylabel("Normalize Genlik", fontsize=7.5, color="#AAAAAA")
    ax_t.tick_params(colors="#AAAAAA", labelsize=7)
    ax_t.legend(fontsize=6, facecolor=BG, labelcolor="white",
                edgecolor=GRID, loc="upper right", ncol=2)
    ax_t.grid(True, color=GRID, lw=0.5, alpha=0.6)
    [sp.set_edgecolor(GRID) for sp in ax_t.spines.values()]

    # ── Orta: BNR dağılımı — tüm subcarrier'lar ──────────────────────────────
    ax_b = fig.add_subplot(gs[ri, 1])
    ax_b.set_facecolor(BG)

    bnr_all  = r["bnr_vals"]
    colors_b = np.array([C_BNR_L] * n_sc_total)
    colors_b[r["top_idx"]] = C_BNR_H
    sort_ord  = np.argsort(bnr_all)[::-1]
    bars = ax_b.bar(np.arange(n_sc_total),
                    bnr_all[sort_ord],
                    color=colors_b[sort_ord], width=1.0, alpha=0.9)

    ax_b.axvline(TOP_N_SC - 0.5, color="#FF6B35", ls="--", lw=1.5, alpha=0.9,
                 label=f"Top-{TOP_N_SC} kesme")
    ax_b.set_title(
        f"BNR[{BNR_LO}–{BNR_HI} Hz] — {n_sc_total} Subcarrier\n"
        f"Yeşil: seçilen top-{TOP_N_SC}  |  Gri: elenen",
        fontsize=8.5, fontweight="bold", color="white", pad=4
    )
    ax_b.set_xlabel("Subcarrier sıralaması (BNR büyükten küçüğe)", fontsize=7.5,
                    color="#AAAAAA")
    ax_b.set_ylabel("BNR değeri", fontsize=7.5, color="#AAAAAA")
    ax_b.tick_params(colors="#AAAAAA", labelsize=7)
    ax_b.legend(fontsize=7, facecolor=BG, labelcolor="white", edgecolor=GRID)
    ax_b.grid(True, axis="y", color=GRID, lw=0.5, alpha=0.6)
    [sp.set_edgecolor(GRID) for sp in ax_b.spines.values()]

    # ── Sağ: FFT + Welch PSD spektrumu ────────────────────────────────────────
    ax_f = fig.add_subplot(gs[ri, 2])
    ax_f.set_facecolor(BG)

    # Welch PSD (kırmızı, arka plan referans)
    if len(r["wp"]) > 0:
        wp_norm = r["wp"] / (r["wp"].max() + 1e-12)
        ax_f.plot(r["wf"] * 60.0, wp_norm, color=C_WELCH, lw=1.2,
                  alpha=0.55, label="Welch PSD (norm.)")

    # FFT (mavi, ana)
    if len(r["fft_mag"]) > 0:
        ax_f.plot(r["fft_f_bpm"], r["fft_mag"], color=C_FFT, lw=1.4,
                  alpha=0.90, label="FFT (sıfır-dolgulu, norm.)")

    # Tepe işaretçisi
    ax_f.axvline(r["fft_bpm"], color=C_PEAK, ls="--", lw=2.0, alpha=0.95)
    ax_f.annotate(
        f"  {r['fft_bpm']:.1f} BPM\n  ({r['fft_hz']:.3f} Hz)",
        xy=(r["fft_bpm"], 1.0), xytext=(min(r["fft_bpm"] + 1.5, 33), 0.80),
        fontsize=9, color=C_PEAK, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_PEAK, lw=1.2)
    )

    # Referans çizgileri
    for ref_bpm, col, tag in [
        (12, "#69F0AE", "12 BPM (normal)"),
        (25, "#CE93D8", "25 BPM (hedef)"),
    ]:
        ax_f.axvline(ref_bpm, color=col, ls=":", lw=1.8, alpha=0.85, label=tag)

    ax_f.set_title(
        f"FFT Spektrumu  →  {r['fft_bpm']:.1f} BPM ({r['fft_hz']:.3f} Hz)\n"
        f"Welch ref: {r['w_bpm']:.1f} BPM  |  "
        f"FFT Δf={r['Fs']/FFT_NFFT*60:.2f} BPM çözünürlük",
        fontsize=8.5, fontweight="bold", color="white", pad=4
    )
    ax_f.set_xlabel("Frekans (BPM)", fontsize=7.5, color="#AAAAAA")
    ax_f.set_ylabel("Normalize Genlik", fontsize=7.5, color="#AAAAAA")
    ax_f.set_xlim([BP_LOW * 60, BP_HIGH * 60])
    ax_f.set_ylim([0, 1.15])
    ax_f.legend(fontsize=7, facecolor=BG, labelcolor="white",
                edgecolor=GRID, loc="upper right")
    ax_f.tick_params(colors="#AAAAAA", labelsize=7)
    ax_f.grid(True, color=GRID, lw=0.5, alpha=0.6)
    [sp.set_edgecolor(GRID) for sp in ax_f.spines.values()]

out_path = DATA_DIR / "csi_respirfi_analysis.png"
plt.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor=DARK)
print(f"\nGrafik kaydedildi: {out_path}")
