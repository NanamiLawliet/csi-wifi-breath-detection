# -*- coding: utf-8 -*-
"""
CSI Advanced Breath Analysis
Pipeline: Trim -> Hampel -> S-G -> BNR Selection -> Bandpass -> PC-SampEn -> BPM + I:E

PC selection uses a HYBRID criterion:
  score = peak_spectral_power / (SampEn + eps)
Rationale: environmental oscillations can have very low SampEn (high regularity)
but LOW absolute power compared to the actual breathing component after BNR
filtering. The hybrid score rewards BOTH regularity AND spectral dominance.
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
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
# PARAMETERS
# ============================================================
TRIM_SEC      = 5
BP_LOW        = 0.05       # Hz
BP_HIGH       = 0.40       # Hz
BP_ORDER      = 4
SG_WINDOW     = 51
SG_ORDER      = 3
HAMPEL_HALF   = 100        # total window = 200 samples
HAMPEL_NSIG   = 2.0
TOP_N_SC      = 15         # top BNR subcarriers
N_PCS         = 3
SAMPEN_M      = 3
SAMPEN_R_COEF = 0.1        # r = 0.1 * std(x)
SAMPEN_N_MAX  = 400        # resample limit before SampEn

DATA_DIR = Path(r"c:/Users/X1 Carbon 8 i5 LTE/Desktop/ehb/ehb440/experiment")
FILES = [
    "csi_data_empty_room.csv",
    "csi_data_5bpm_breath.csv",
    "csi_data_10bpm_breath.csv",
]
LABELS = {
    "csi_data_empty_room.csv"  : "Dosya 1 - Bos Oda (Referans)",
    "csi_data_5bpm_breath.csv" : "Dosya 2 - Nefes Verisi A",
    "csi_data_10bpm_breath.csv": "Dosya 3 - Nefes Verisi B",
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def parse_csi_amplitude(csi_str: str) -> np.ndarray:
    vals = np.fromstring(csi_str.strip("[] "), dtype=np.int16, sep=",")
    vals = vals[4:]                            # skip 2 pilot/header pairs
    return np.sqrt(vals[0::2].astype(np.float32)**2 +
                   vals[1::2].astype(np.float32)**2)


def hampel_filter(x: np.ndarray, half_win: int = 100,
                  n_sig: float = 2.0) -> np.ndarray:
    """Vectorised Hampel filter (window = 2*half_win+1 samples)."""
    k      = 1.4826
    padded = np.pad(x, half_win, mode="edge")
    wins   = sliding_window_view(padded, 2 * half_win + 1)
    meds   = np.median(wins, axis=1)
    mads   = np.median(np.abs(wins - meds[:, None]), axis=1)
    out    = x.copy()
    mask   = np.abs(x - meds) > n_sig * k * mads
    out[mask] = meds[mask]
    return out


def compute_bnr(x: np.ndarray, fs: float,
                low: float = BP_LOW, high: float = BP_HIGH) -> float:
    """Breathing-to-Noise Ratio via Welch PSD."""
    nperseg = min(len(x), max(128, int(fs * 20)))
    f, p   = signal.welch(x, fs=fs, nperseg=nperseg)
    in_b   = (f >= low) & (f <= high)
    out_b  = ~in_b & (f > 0)
    breath = np.trapezoid(p[in_b],  f[in_b])  if in_b.any()  else 0.0
    noise  = np.trapezoid(p[out_b], f[out_b]) if out_b.any() else 1e-10
    return breath / max(noise, 1e-10)


def bandpass(x: np.ndarray, fs: float,
             low: float = BP_LOW, high: float = BP_HIGH,
             order: int = BP_ORDER) -> np.ndarray:
    nyq  = fs / 2.0
    b, a = signal.butter(order, [low / nyq, high / nyq], btype="band")
    return signal.filtfilt(b, a, x)


def sample_entropy(x: np.ndarray, m: int = 3,
                   r_coef: float = 0.1, n_max: int = 400) -> float:
    """SampEn with r = r_coef * std(x). Downsamples to n_max for speed."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) > n_max:
        x = signal.resample(x, n_max)
    r = r_coef * np.std(x, ddof=1)
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

    B, A = _count(m), _count(m + 1)
    return float(-np.log(A / B)) if B > 0 and A > 0 else np.inf


def welch_psd(x: np.ndarray, fs: float,
              low: float = BP_LOW, high: float = BP_HIGH):
    """Return (f_band, p_band, peak_hz, peak_bpm)."""
    nperseg = min(len(x), max(512, int(2 * fs / low)))
    f, p   = signal.welch(x, fs=fs, nperseg=nperseg,
                          noverlap=nperseg // 2, window="hann")
    mask   = (f >= low) & (f <= high)
    f_m, p_m = f[mask], p[mask]
    if len(p_m) == 0:
        return f_m, p_m, 0.0, 0.0
    idx = np.argmax(p_m)
    return f_m, p_m, f_m[idx], f_m[idx] * 60.0


def select_best_pc(pcs: np.ndarray, fs: float,
                   entropies: list, ev: np.ndarray):
    """
    Variance-Normalised SampEn criterion (PC-SampEn extended).

    Criterion: minimise  SampEn / explained_variance_fraction
    Rationale: a very-regular environmental oscillation (low SampEn) that
    only explains a small fraction of the total variance should be penalised
    relative to a moderately-regular but variance-dominant breathing component.
    This matches the PC-SampEn spirit while being robust to periodic
    environmental artefacts.

    score_to_minimise[k] = SampEn[k] / (ev[k] / 100)
    """
    scores = []
    bpms   = []
    ppows  = []
    for k in range(len(entropies)):
        _, wp, _, peak_bpm = welch_psd(pcs[:, k], fs)
        peak_power = float(np.max(wp)) if len(wp) > 0 else 0.0
        se  = entropies[k] if np.isfinite(entropies[k]) else 10.0
        var = max(ev[k] / 100.0, 1e-6)
        scores.append(se / var)    # lower = better
        bpms.append(peak_bpm)
        ppows.append(peak_power)
    return int(np.argmin(scores)), bpms, ppows, scores


def compute_ie_ratio(sig: np.ndarray, fs: float, bpm_est: float):
    """Returns (ie_ratio, peak_indices, trough_indices)."""
    if bpm_est <= 0:
        return None, np.array([]), np.array([])
    min_dist = max(int(fs * 60 / bpm_est / 3), 5)
    std_s    = np.std(sig)
    peaks,   _ = find_peaks( sig, distance=min_dist, prominence=0.05 * std_s)
    troughs, _ = find_peaks(-sig, distance=min_dist, prominence=0.05 * std_s)

    if len(peaks) < 2 or len(troughs) < 2:
        return None, peaks, troughs

    t      = np.arange(len(sig)) / fs
    events = ([(t[p], "pk") for p in peaks] +
              [(t[tr], "tr") for tr in troughs])
    events.sort(key=lambda e: e[0])

    inhale, exhale = [], []
    for i in range(len(events) - 1):
        t0, ty0 = events[i];  t1, ty1 = events[i + 1]
        dur = t1 - t0
        if ty0 == "tr" and ty1 == "pk":
            inhale.append(dur)
        elif ty0 == "pk" and ty1 == "tr":
            exhale.append(dur)

    if not inhale or not exhale:
        return None, peaks, troughs
    return float(np.mean(inhale) / np.mean(exhale)), peaks, troughs


# ============================================================
# MAIN ANALYSIS LOOP
# ============================================================
results = {}

for fname in FILES:
    fpath = DATA_DIR / fname
    print(f"\n{'='*65}")
    print(f"  {fname}")
    print(f"{'='*65}")

    # 1. Load + trim
    df = pd.read_csv(fpath, parse_dates=["timestamp"])
    df = (df.dropna(subset=["csi_data"])
            .sort_values("timestamp")
            .reset_index(drop=True))
    df["t"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
    Fs = 1.0 / df["t"].diff().median()
    print(f"  Samples: {len(df)}  Duration: {df['t'].iloc[-1]:.1f}s  Fs: {Fs:.2f} Hz")

    mask = (df["t"] >= TRIM_SEC) & (df["t"] <= df["t"].iloc[-1] - TRIM_SEC)
    df   = df[mask].reset_index(drop=True)
    df["t"] = df["t"] - df["t"].iloc[0]
    print(f"  After trim: {len(df)} samples  {df['t'].iloc[-1]:.1f}s")

    # 2. Parse CSI amplitudes
    amp = np.stack(df["csi_data"].map(parse_csi_amplitude).values)
    amp = amp[:, np.any(amp > 0.5, axis=0)]
    n_s, n_sc = amp.shape
    print(f"  Active subcarriers: {n_sc}")

    # 2b. Hampel + Savitzky-Golay per subcarrier
    sg_w = SG_WINDOW if SG_WINDOW < n_s else (n_s-1 if (n_s-1)%2 else n_s-2)
    print(f"  Hampel(half={HAMPEL_HALF}, n_sig={HAMPEL_NSIG}) + "
          f"S-G(w={sg_w}, poly={SG_ORDER}) ...")
    denoised = np.empty_like(amp, dtype=np.float64)
    for j in range(n_sc):
        col = hampel_filter(amp[:, j].astype(np.float64),
                            half_win=HAMPEL_HALF, n_sig=HAMPEL_NSIG)
        denoised[:, j] = signal.savgol_filter(col, sg_w, SG_ORDER)

    # 3. BNR subcarrier selection (computed on denoised, pre-bandpass signal)
    print(f"  BNR calculation ({n_sc} subcarriers) ...")
    bnr_vals = np.array([compute_bnr(denoised[:, j], Fs) for j in range(n_sc)])
    top_idx  = np.argsort(bnr_vals)[-TOP_N_SC:][::-1]
    top_bnr  = bnr_vals[top_idx]
    print(f"  Top {TOP_N_SC} selected  BNR=[{top_bnr[-1]:.2f}..{top_bnr[0]:.2f}]  "
          f"(median all: {np.median(bnr_vals):.2f})")

    # 4. Bandpass on selected subcarriers
    selected  = denoised[:, top_idx]
    bp_matrix = np.stack([bandpass(selected[:, j], Fs) for j in range(TOP_N_SC)],
                         axis=1)

    # 5. PCA -> 3 components
    X_sc = StandardScaler().fit_transform(bp_matrix)
    pca  = PCA(n_components=N_PCS)
    pcs  = pca.fit_transform(X_sc)                    # (N, 3)
    ev   = pca.explained_variance_ratio_ * 100
    print(f"  PCA explained variance: "
          f"PC1={ev[0]:.1f}%  PC2={ev[1]:.1f}%  PC3={ev[2]:.1f}%")

    # 5b. SampEn per component
    print(f"  SampEn (m={SAMPEN_M}, r={SAMPEN_R_COEF}*std, max_n={SAMPEN_N_MAX}) ...")
    entropies = []
    pc_bpms_for_diag = []
    for k in range(N_PCS):
        se  = sample_entropy(pcs[:, k], m=SAMPEN_M,
                             r_coef=SAMPEN_R_COEF, n_max=SAMPEN_N_MAX)
        _, _, _, dom_bpm = welch_psd(pcs[:, k], Fs)
        entropies.append(se)
        pc_bpms_for_diag.append(dom_bpm)
        print(f"    PC{k+1}: SampEn={se:.4f}  Welch_dom={dom_bpm:.1f} BPM  "
              f"(var={ev[k]:.1f}%)")

    # Variance-Normalised SampEn selection
    best_k, pc_bpms, pc_ppows, scores = select_best_pc(pcs, Fs, entropies, ev)
    print(f"  VN-SampEn scores (lower=better): " +
          "  ".join(f"PC{k+1}={scores[k]:.3f}" for k in range(N_PCS)))
    print(f"  => Selected: PC{best_k+1}  "
          f"SampEn={entropies[best_k]:.4f}  "
          f"PeakPower={pc_ppows[best_k]:.4f}  "
          f"Welch={pc_bpms[best_k]:.1f} BPM")

    breath_sig = bandpass(pcs[:, best_k], Fs)         # final cleanup

    # 6. BPM detection
    wf, wp, peak_hz, peak_bpm = welch_psd(breath_sig, Fs)

    N_s    = len(breath_sig)
    ff     = fftfreq(N_s, d=1.0/Fs)
    sp     = np.abs(fft(breath_sig * np.hanning(N_s)))
    pos    = (ff > 0) & (ff >= BP_LOW) & (ff <= BP_HIGH)
    fft_hz  = ff[pos][np.argmax(sp[pos])]
    fft_bpm = fft_hz * 60.0
    print(f"  Welch BPM: {peak_bpm:.1f}  FFT BPM: {fft_bpm:.1f}")

    # 6b. I:E ratio
    ie_ratio, peaks, troughs = compute_ie_ratio(breath_sig, Fs, peak_bpm)
    t_arr = df["t"].values
    ie_str = f"{ie_ratio:.3f}" if ie_ratio is not None else "N/A"
    print(f"  I:E ratio: {ie_str}  "
          f"(peaks={len(peaks)}, troughs={len(troughs)})")

    results[fname] = {
        "t": t_arr, "sig": breath_sig, "Fs": Fs,
        "wf": wf, "wp": wp,
        "peak_hz": peak_hz, "peak_bpm": peak_bpm, "fft_bpm": fft_bpm,
        "peaks": peaks, "troughs": troughs,
        "ie": ie_ratio, "ie_str": ie_str,
        "best_k": best_k,
        "entropies": entropies,
        "pc_bpms": pc_bpms,
        "pc_ppows": pc_ppows,
        "scores": scores,
        "ev": ev,
        "bnr_max": top_bnr[0], "bnr_med": np.median(bnr_vals),
    }

# ============================================================
# VISUALIZATION
# ============================================================
DARK = "#0D1117"; BG = "#161B22"; GRID = "#30363D"
C_SIG = "#4FC3F7"; C_PSD = "#EF5350"; C_PEAK = "#FFB300"
C_PK  = "#FF5722"; C_TR  = "#66BB6A"
C_5   = "#69F0AE"; C_10  = "#CE93D8"

fig = plt.figure(figsize=(19, 14))
fig.patch.set_facecolor(DARK)
fig.suptitle(
    "CSI Breath Analysis  |  ESP32 Wi-Fi (Single Antenna)\n"
    "BNR Subcarrier Selection + PC-SampEn Isolation + I:E Ratio",
    fontsize=13, fontweight="bold", color="white", y=0.997,
)
gs = gridspec.GridSpec(len(FILES), 2, figure=fig,
                       hspace=0.52, wspace=0.30,
                       left=0.06, right=0.97, top=0.935, bottom=0.05)

for ri, fname in enumerate(FILES):
    r   = results[fname]
    lbl = LABELS[fname]
    k   = r["best_k"]
    t   = r["t"]; sig = r["sig"]
    peaks = r["peaks"]; troughs = r["troughs"]

    title_t = (
        f"{lbl}\n"
        f"PC{k+1} selected  SampEn={r['entropies'][k]:.4f}  |  "
        f"Welch={r['peak_bpm']:.1f} BPM  |  I:E={r['ie_str']}"
    )
    title_f = (
        f"Welch PSD  |  Peak: {r['peak_bpm']:.1f} BPM ({r['peak_hz']:.3f} Hz)  "
        f"|  FFT: {r['fft_bpm']:.1f} BPM"
    )

    # Time domain
    ax_t = fig.add_subplot(gs[ri, 0])
    ax_t.set_facecolor(BG)
    ax_t.plot(t, sig, color=C_SIG, lw=0.8, alpha=0.9, zorder=2)
    if len(peaks):
        ax_t.scatter(t[peaks], sig[peaks], color=C_PK, s=40, zorder=5,
                     marker="^", label=f"Peak (Inhale end, n={len(peaks)})")
    if len(troughs):
        ax_t.scatter(t[troughs], sig[troughs], color=C_TR, s=40, zorder=5,
                     marker="v", label=f"Trough (Exhale end, n={len(troughs)})")
    ax_t.set_title(title_t, fontsize=8.5, fontweight="bold", color="white", pad=4)
    ax_t.set_xlabel("Time (s)", fontsize=8, color="#AAAAAA")
    ax_t.set_ylabel("Amplitude (a.u.)", fontsize=8, color="#AAAAAA")
    ax_t.tick_params(colors="#AAAAAA", labelsize=7)
    [sp.set_edgecolor(GRID) for sp in ax_t.spines.values()]
    ax_t.grid(True, color=GRID, lw=0.5, alpha=0.6)
    if len(peaks) or len(troughs):
        ax_t.legend(fontsize=6.5, facecolor=BG, labelcolor="white",
                    edgecolor=GRID, loc="upper right")

    # Frequency domain
    ax_f = fig.add_subplot(gs[ri, 1])
    ax_f.set_facecolor(BG)
    bpm_ax = r["wf"] * 60.0
    ax_f.plot(bpm_ax, r["wp"], color=C_PSD, lw=1.3, alpha=0.9)
    ax_f.axvline(5,  color=C_5,    ls=":", lw=1.2, alpha=0.7, label="5 BPM ref")
    ax_f.axvline(10, color=C_10,   ls=":", lw=1.2, alpha=0.7, label="10 BPM ref")
    pbx = r["peak_hz"] * 60.0
    pby = r["wp"][np.argmax(r["wp"])]
    ax_f.axvline(pbx, color=C_PEAK, ls="--", lw=1.5, alpha=0.85)
    ax_f.plot(pbx, pby, "o", color=C_PEAK, ms=9, zorder=6,
              label=f"Peak: {r['peak_bpm']:.1f} BPM")
    ax_f.annotate(
        f" {r['peak_bpm']:.1f} BPM\n({r['peak_hz']:.3f} Hz)",
        xy=(pbx, pby), xytext=(pbx + 1.5, pby * 0.82),
        fontsize=7.5, color=C_PEAK, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_PEAK, lw=1.0),
    )
    ax_f.set_title(title_f, fontsize=8.5, fontweight="bold", color="white", pad=4)
    ax_f.set_xlabel("Frequency (BPM)", fontsize=8, color="#AAAAAA")
    ax_f.set_ylabel("Power (a.u.^2/Hz)", fontsize=8, color="#AAAAAA")
    ax_f.set_xlim([BP_LOW * 60, BP_HIGH * 60])
    ax_f.legend(fontsize=7, facecolor=BG, labelcolor="white", edgecolor=GRID)
    ax_f.tick_params(colors="#AAAAAA", labelsize=7)
    [sp.set_edgecolor(GRID) for sp in ax_f.spines.values()]
    ax_f.grid(True, color=GRID, lw=0.5, alpha=0.6)

out_path = DATA_DIR / "csi_advanced_analysis.png"
plt.savefig(str(out_path), dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print(f"\nPlot saved: {out_path}")

# ============================================================
# SUMMARY TABLE
# ============================================================
W = 88
print(f"\n{'='*W}")
print(f"  {'SUMMARY TABLE':^{W-4}}")
print(f"{'='*W}")
hdr = (f"  {'File':<26}  {'Fs':>5}  {'BNR_top':>7}  {'PC':>3}  "
       f"{'SampEn':>7}  {'Welch BPM':>9}  {'FFT BPM':>7}  {'I:E':>6}")
print(hdr)
print(f"  {'-'*(W-4)}")
for fname in FILES:
    r = results[fname]
    k = r["best_k"]
    row = (f"  {LABELS[fname]:<26}  "
           f"{r['Fs']:>5.2f}  "
           f"{r['bnr_max']:>7.2f}  "
           f"PC{k+1:>1}  "
           f"{r['entropies'][k]:>7.4f}  "
           f"{r['peak_bpm']:>9.1f}  "
           f"{r['fft_bpm']:>7.1f}  "
           f"{r['ie_str']:>6}")
    print(row)
print(f"{'='*W}")

print("\n  All-PC SampEn diagnostics:")
for fname in FILES:
    r = results[fname]
    diag = "  ".join(
        f"PC{k+1}: SE={r['entropies'][k]:.3f} @ {r['pc_bpms'][k]:.1f}BPM"
        for k in range(N_PCS))
    print(f"    {LABELS[fname]}: {diag}")

print(f"\n  Classification:")
breath_files = [f for f in FILES if "empty" not in f]
for fname in sorted(breath_files, key=lambda f: results[f]["peak_bpm"]):
    r   = results[fname]
    bpm = r["peak_bpm"]
    tag = "5 BPM" if abs(bpm - 5) <= abs(bpm - 10) else "10 BPM"
    print(f"    {LABELS[fname]}: {bpm:.1f} BPM  =>  {tag} recording")
print(f"{'='*W}")
