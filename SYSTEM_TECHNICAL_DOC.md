# Wi-Fi CSI Based Dual-Modality Respiration Monitoring System
## Technical System Documentation

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Hardware Architecture](#2-hardware-architecture)
3. [Data Collection Layer](#3-data-collection-layer)
4. [Signal Processing Pipeline](#4-signal-processing-pipeline)
5. [Subcarrier Selection Method — BNR Filtering](#5-subcarrier-selection-method--bnr-filtering)
6. [PCA and Component Selection — VN-SampEn](#6-pca-and-component-selection--vn-sampen)
7. [BPM Estimation — Zero-Padded FFT](#7-bpm-estimation--zero-padded-fft)
8. [BNR Presence Gate (Coherence Gate)](#8-bnr-presence-gate-coherence-gate)
9. [Camera Modality — MediaPipe Pose](#9-camera-modality--mediapipe-pose)
10. [Dual-Modality Fusion](#10-dual-modality-fusion)
11. [Real-Time System Architecture](#11-real-time-system-architecture)
12. [Offline Analysis Results](#12-offline-analysis-results)
13. [Key Findings and Limitations](#13-key-findings-and-limitations)
14. [Algorithm Parameter Reference Table](#14-algorithm-parameter-reference-table)

---

## 1. System Overview

This system simultaneously processes Wi-Fi CSI (Channel State Information) signals and RGB camera images in a home environment to measure a person's respiration rate (BPM — Breaths Per Minute) without any physical contact.

### Key Specifications

| Feature | Value |
|---------|-------|
| Modality | Wi-Fi CSI (802.11n) + RGB Camera |
| Hardware | 2 × ESP32 (ESP-NOW protocol) |
| Sampling Rate (CSI) | ~11–13 Hz |
| Sampling Rate (Camera) | 30 FPS |
| Target BPM Range | 6–36 BPM (0.1–0.6 Hz) |
| Processing Latency | ~1 second (sliding window) |
| Reference Method | RespirFi (subcarrier filtering + PCA + FFT) |

### Core Problem

In indoor environments, Wi-Fi CSI signals reflect not only respiratory movement but also HVAC systems, fans, and other periodic environmental sources. The **BNR-based subcarrier filtering** method developed here systematically eliminates these noise sources before they enter the respiration estimation stage.

---

## 2. Hardware Architecture

### 2.1 ESP32 Dual-Antenna Setup

```
┌─────────────────────────────────────────────────────────────────┐
│                         Room / Environment                       │
│                                                                 │
│   ┌──────────────┐    ESP-NOW (Wi-Fi 802.11n)   ┌───────────┐  │
│   │  TX ESP32    │ ─────────────────────────►  │ RX ESP32  │  │
│   │  (COM9)      │    CSI packets (~13 Hz)      │  (COM10)  │  │
│   │              │                              │           │  │
│   │ USB: NONE    │     [Person breathing]        │ USB: YES  │  │
│   │ not connected│         ↕ RF path             │ to PC     │  │
│   │ to PC        │                              │           │  │
│   └──────────────┘                              └─────┬─────┘  │
│                                                       │        │
└───────────────────────────────────────────────────────┼────────┘
                                                        │ USB-Serial
                                                        ▼
                                                   ┌──────────┐
                                                   │    PC    │
                                                   │ (Python) │
                                                   └──────────┘
```

### 2.2 Device Roles

**TX ESP32 (Transmitter)**
- Role: Continuously broadcasts Wi-Fi CSI packets
- Connection: NO USB connection to PC; runs on independent power
- Protocol: ESP-NOW (802.11n, 40 MHz channel width)
- Config: `PORT_TX = "COM9"` — reference only, PC does not connect

**RX ESP32 (Receiver)**
- Role: Receives CSI packets from TX, forwards to PC via USB-Serial
- Connection: `PORT_RX = "COM10"`, `BAUD = 115200 bps`
- Output format: `CSI_START{...JSON...}CSI_END` one packet per line

### 2.3 CSI Data Format

Each line contains the following JSON structure:

```json
{
  "csi_data": [I0, Q0, I1, Q1, ..., I127, Q127],
  "rssi": -65,
  "noise_floor": -95
}
```

**Amplitude calculation** (per subcarrier):
```
A_k = √(I_k² + Q_k²)
```

- First 4 elements of `csi_data` array are skipped (invalid pilot subcarriers)
- Remaining 252 elements → 126 pairs → 126 subcarrier amplitudes
- Active subcarriers (variance > 0.5): typically 115–116

### 2.4 Ideal Placement (Fresnel Zone)

```
[TX ESP32] ←─────── 1–3 meters ───────→ [RX ESP32]
                        ↑
                   Person here
                 (on the RF path)
```

**Critical note:** TX and RX ESP32 devices **must not be placed side by side**. CSI variation is maximized when the person is on the TX-RX line of sight, within the first Fresnel ellipsoid. Side-by-side placement (the constraint in this study) yields BNR < 1.0.

---

## 3. Data Collection Layer

### 3.1 Serial Port Reading

`SerialThread` (QThread) continuously reads and parses the serial port:

```python
ser = serial.Serial(RX_PORT, RX_BAUD, timeout=1.0)
line = ser.readline().decode('ascii', errors='ignore').strip()
amps = parse_csi_amplitudes(line)   # → np.ndarray (n_sub,) or None
```

`parse_csi_amplitudes()`:
1. Extract JSON block with `CSI_START{...}CSI_END` regex pattern
2. Parse `csi_data` list
3. `vals[4:]` — skip pilot subcarriers
4. Amplitude from I/Q pairs: `sqrt(I² + Q²)`
5. Return `np.ndarray(dtype=float32)`

### 3.2 Circular Buffer

Incoming CSI packets are stored in `collections.deque(maxlen=BUF_MAX)`:

```
BUF_MAX = WINDOW_S × 20 = 15 × 20 = 300 packets
```

Each packet is a `np.ndarray(n_sub,)` containing all subcarrier amplitudes. The processing window slides every second (`PROC_INTERVAL = 1000 ms`).

---

## 4. Signal Processing Pipeline

### 4.1 General Flow

```
Raw CSI Matrix (N × n_sub)
        │
        ▼
[1] Dead Subcarrier Removal
    variance < 1e-6 → remove
        │
        ▼
[2] Hampel Filter (per subcarrier)
    half_win = 50 samples (~4.2 s)
    k = 1.4826, nsig = 2.5
        │
        ▼
[3] Savitzky-Golay Filter (per subcarrier)
    window = 11 samples (~0.93 s), poly = 3
        │
        ▼
[4] BNR Calculation (per subcarrier)
    band = [0.16 Hz – 0.60 Hz]
        │
        ▼
[5] Top-20 Subcarrier Selection
    BNR descending → top 20
        │
        ▼
[6] Butterworth Bandpass (per subcarrier)
    [0.10 Hz – 0.60 Hz], order = 4, filtfilt
        │
        ▼
[7] StandardScaler + PCA (n_components=3)
        │
        ▼
[8] Best PC Selection via VN-SampEn
        │
        ▼
[9] Zero-Padded FFT (N_FFT = 4096)
    → peak_bpm, confidence
        │
        ▼
    BPM Output
```

### 4.2 Hampel Filter

**Purpose:** Cleans sudden spikes in the CSI stream (RF packet loss, multipath fading jumps) using median-based outlier detection.

**Algorithm:**
1. For each sample `x[i]`, take window `[i-half_win, i+half_win]`
2. Compute window median `m` and MAD (Median Absolute Deviation):
   ```
   MAD = median(|x_window - m|)
   ```
3. Outlier criterion:
   ```
   |x[i] - m| > nsig × 1.4826 × MAD
   ```
4. If outlier: `x[i] ← m`

**Parameters:**
- `half_win = 50` → total window ≈ 101 samples (~8.6 s @ 11.8 Hz)
- `nsig = 2.5` → ~1.5% false positive rate under Gaussian assumption

**Why half_win = 50?**  
For a 25 BPM signal, period ≈ 29 samples. With `half_win = 50`, ~3.5 full periods fit in the window; the median approximates the DC level and peaks do not exceed the threshold. A narrower window (half_win < 14) could mark a single peak as an outlier.

### 4.3 Savitzky-Golay Filter

**Purpose:** Suppresses high-frequency noise (RF jitter, antenna movement) via polynomial fitting while preserving the shape of the breath signal (peak and trough timing).

**Mathematical basis:** A polynomial of degree `poly` is fitted to data in the window; the center point is replaced by the polynomial value.

**Critical parameter selection:**

| Window (samples) | Duration @ 12 Hz | Ratio to 25 BPM period | Result |
|-----------------|------------------|------------------------|--------|
| 51 samples | 4.25 s | **1.77 × period** | Oversmoothing — 25 BPM eliminated |
| 11 samples | 0.92 s | **0.38 × period** | Period preserved ✓ |

**Rule:** `SG_WINDOW < T_breathing × Fs`  
For 25 BPM: `T = 2.4 s`, `Fs = 12 Hz` → `T × Fs = 29 samples` → `SG_WINDOW = 11 < 29` ✓

---

## 5. Subcarrier Selection Method — BNR Filtering

The **core technical contribution** of this work. While traditional methods feed all subcarriers into PCA, this approach performs quality filtering first.

### 5.1 BNR (Breathing-to-Noise Ratio) Definition

For a subcarrier signal `x`, BNR is:

```
          ∫[BNR_LO to BNR_HI] S_xx(f) df
BNR(x) = ─────────────────────────────────
          ∫[0 to ∞, outside] S_xx(f) df
```

Where `S_xx(f)` is the Power Spectral Density estimated via Welch's method.

**Calculation:**
```python
nperseg = min(N, max(256, int(Fs × 15)))
f, p    = welch(x, fs=Fs, nperseg=nperseg)
in_b    = (f >= BNR_LO) & (f <= BNR_HI)
out_b   = ~in_b & (f > 0)
BNR     = trapezoid(p[in_b], f[in_b]) / trapezoid(p[out_b], f[out_b])
```

### 5.2 Critical Importance of BNR Band Selection

**Traditional approach (failed):** `BNR_LO = 0.10 Hz`  
→ HVAC/fan (8–10 BPM = 0.133–0.167 Hz) **included in the breath band**  
→ BNR rates HVAC-dominant subcarriers highly  
→ Top-N selected subcarriers are HVAC-dominated  
→ PCA extracts HVAC signal  
→ 8–10 BPM detected instead of 25 BPM

**Proposed approach (successful):** `BNR_LO = 0.16 Hz`  
→ HVAC (0.133–0.150 Hz) **falls OUTSIDE the breath band**  
→ 25 BPM (0.417 Hz) is inside the breath band  
→ Subcarriers carrying 25 BPM get high BNR  
→ Top-20 selected subcarriers carry breath information  
→ PCA extracts respiration signal  
→ ~28 BPM detected (close to 25 BPM target)

```
Frequency axis:
0    0.10  0.133  0.16   0.167  ...   0.417   0.60 Hz
│     │      │     │       │          │        │
│     │    HVAC   BNR_LO  HVAC       25 BPM  BNR_HI
│     │   (8 BPM)         (10 BPM)           
│     │
│     BP_LOW (for bandpass filter)
│
DC

← BP filter band: 0.10 Hz ──────────────────── 0.60 Hz →
         ← BNR calculation band: 0.16 Hz ──────── 0.60 Hz →
                               ↑
                     HVAC excluded from this cutoff
```

### 5.3 Subcarrier Distribution Analysis

From measured data (25 BPM recording, 115 active subcarriers):

| Dominant frequency region | Subcarrier count | BNR_old (0.10 Hz) | BNR_new (0.16 Hz) |
|--------------------------|------------------|-------------------|-------------------|
| 8–10 BPM (HVAC) | 104 | High (0.38–0.40) | Low (< 0.20) |
| 25–28 BPM (breath) | 9 | Low (< 0.30) | High (0.32–0.40) |
| Ambiguous | 2 | — | — |

This inversion is the direct effect of changing BNR_LO.

### 5.4 Top-N Selection

BNR values are sorted descending and the first `TOP_N_SC = 20` subcarriers are selected:

```python
top_idx = np.argsort(bnr_vals)[-TOP_N_SC:][::-1]
selected = denoised[:, top_idx]   # (N, 20)
```

The remaining `n_sc - 20` subcarriers (typically 95–96) are **completely removed from the PCA matrix**.

---

## 6. PCA and Component Selection — VN-SampEn

### 6.1 PCA Application

The selected 20 subcarriers are bandpass filtered and normalized with StandardScaler, then `n_components = 3` PCA is applied:

```python
X_sc = StandardScaler().fit_transform(bp_mat)   # (N, 20)
pca  = PCA(n_components=3)
pcs  = pca.fit_transform(X_sc)                   # (N, 3)
ev   = pca.explained_variance_ratio_ × 100
```

**Typical results (25 BPM file, new method):**
- PC1: 54.3% variance
- PC2: 19.8% variance
- PC3: 6.8% variance

Comparison (old method, all subcarriers):
- PC1: 90.0% variance → single component dominates (HVAC)

The drop in PC1 variance (90% → 54%) indicates multiple meaningful signal components and demonstrates the method's subcarrier separation power.

### 6.2 VN-SampEn (Variance-Normalized Sample Entropy)

**Problem:** Minimum SampEn criterion selects HVAC.  
HVAC is highly periodic → low SampEn → incorrectly selected as the "best" signal.

**Solution — VN-SampEn score:**

```
VN-score(k) = SampEn(PC_k) / (ev_k / 100)
```

| Case | SampEn | Variance | VN-score | Result |
|------|--------|----------|----------|--------|
| HVAC (fan) | Low (0.3) | Low (3%) | High (10.0) | Rejected |
| Breath | Medium (1.8) | High (54%) | Low (3.3) | **Selected** |

Minimum VN-score → selected PC.

**Sample Entropy calculation:**
```python
# m=3, r=0.1×std(x), downsampling n_max=400
B = count_templates(x, m)      # count of m-dimensional matching templates
A = count_templates(x, m+1)    # count of (m+1)-dimensional matching templates
SampEn = -log(A / B)
```

- `SampEn = 0` → completely repetitive (periodic) signal
- `SampEn = ∞` → completely random (noise)
- Breath: intermediate values (1.5–3.0)

---

## 7. BPM Estimation — Zero-Padded FFT

### 7.1 Limitation of Welch's Method

Welch PSD resolution:

```
Δf = Fs / nperseg
```

`nperseg = 512`, `Fs = 11.8 Hz`:
```
Δf = 11.8 / 512 = 0.023 Hz = 1.38 BPM
```

With less data (`nperseg = 118`):
```
Δf = 11.8 / 118 = 0.10 Hz = 6.0 BPM → always outputs 6 BPM!
```

For this reason, Welch is kept only as a reference in the real-time system; the primary BPM estimation uses FFT.

### 7.2 Zero-Padded FFT

```python
N_FFT = 4096   # zero-padding size
freqs = np.fft.rfftfreq(N_FFT, d=1.0/Fs)
mag   = np.abs(np.fft.rfft(x, n=N_FFT))
mask  = (freqs >= BP_LOW) & (freqs <= BP_HIGH)
peak_hz  = freqs[mask][argmax(mag[mask])]
peak_bpm = peak_hz × 60.0
```

**Effective resolution:**
```
Δf = Fs / N_FFT = 11.8 / 4096 = 0.00288 Hz = 0.173 BPM
```

This resolution clearly distinguishes the difference between 25 BPM (0.4167 Hz) and 28 BPM (0.4667 Hz) (~0.05 Hz = 3 BPM).

**Note:** Zero-padding does not improve frequency resolution — it improves **interpolation precision**. True frequency resolution is bounded by `Fs/N` (N = original data length), but peak position detection becomes much more accurate.

### 7.3 Confidence Score

```python
prominence = peak_val / mean(mag[mask])
confidence = clip((prominence - 1.0) / 4.0, 0.0, 1.0)
```

- `prominence < 1.5` → weak signal (confidence = 0)
- `prominence > 5.0` → strong signal (confidence = 1)

---

## 8. BNR Presence Gate (Coherence Gate)

### 8.1 Purpose

Prevents FFT computation when no one is in the room or when the CSI signal is too weak, and shows a meaningful warning to the user.

### 8.2 Operation

```python
BNR_GATE_THRESH = 1.5

# Check max_bnr returned from CSI pipeline:
if max_bnr < BNR_GATE_THRESH:
    bpm = -1.0   # special sentinel value
    status = f"Empty Room / Weak Signal  BNR={max_bnr:.2f}"
    return  # FFT not computed
else:
    bpm = dominant_bpm_fft(pc1, fs)
```

**BNR interpretation table:**

| BNR Range | Interpretation | System Decision |
|-----------|---------------|-----------------|
| < 0.5 | Very weak signal | Empty room / blind spot |
| 0.5–1.5 | Weak signal | Gate closed, BPM not shown |
| 1.5–5.0 | Normal signal | Gate open, BPM computed |
| 5.0–20.0 | Strong signal | Ideal Fresnel position |
| > 20.0 | Very strong | Perfect LoS |

**Measured values (this study, side-by-side ESP32):**
- All recordings: BNR_max ≈ 0.28–0.40 → below gate threshold
- Expected in ideal LoS: BNR 5–20

### 8.3 GUI Integration

When BNR gate activates:
- `bpm = -1.0` signal sent to GUI
- `_on_csi_result()` method catches this value
- Label: `"Wi-Fi CSI  --  0 BPM  (Empty Room / Weak Signal)"` (red border)
- Status bar: `"CSI: Empty Room / Weak Signal  BNR=0.38 (threshold=1.5)"`

---

## 9. Camera Modality — MediaPipe Pose

### 9.1 Respiration Detection via Shoulder Movement

The camera pipeline extracts a breath waveform by tracking the person's chest/shoulder movement.

**MediaPipe Pose Landmarker (Tasks API v0.10.35):**
```python
opts = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="pose_landmarker_lite.task"),
    running_mode=RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
```

**Tracked landmarks:**
- Landmark 11: Left shoulder
- Landmark 12: Right shoulder
- `shoulder_y = (lm11.y + lm12.y) / 2` — normalized Y coordinate

**Why the Y axis?** When inhaling, the chest rises; the Y coordinate decreases. Inversion: `raw_y = -(samples - mean)`.

### 9.2 Camera Pipeline

```
Raw shoulder-Y time series
        │
[1] DC removal (mean subtraction)
        │
[2] Hampel filter (half_win = max(5, Fs//2))
        │
[3] Savitzky-Golay (window = min(51, N//6 | 1), poly=3)
        │
[4] Butterworth bandpass (0.05–0.40 Hz, order=4)
        │
[5] FFT BPM (dominant_bpm_fft)
        │
    Camera BPM
```

---

## 10. Dual-Modality Fusion

### 10.1 Spectral Geometric Mean

The FFT spectrum of each modality is interpolated onto a common 200-point frequency grid (`_FUSE_FREQS = linspace(0.05, 0.40, 200)`) and normalized (peak = 1):

```python
fused_spec = sqrt(cam_spec × csi_spec)
```

**Rationale:** The geometric mean is maximized when both signals are strong at the same frequency; if one is weak, the value drops. This automatically suppresses false detections.

### 10.2 Agreement Score

```python
agreement = max(0.0, 1.0 - |cam_bpm - csi_bpm| / 6.0)
```

- `agreement = 1.0` → two modalities fully aligned (< 1 BPM difference)
- `agreement = 0.0` → 6 BPM or more disagreement

The agreement score determines the color coding of the fused BPM label in the GUI:
- Purple (bright): agreement > 0.70
- Purple (medium): agreement > 0.40
- Purple (dim): agreement ≤ 0.40

### 10.3 BNR Gate and Fusion Interaction

If the CSI signal fails the BNR gate, `FusedBreathWorker` produces no output:

```python
_, _, csi_signal, csi_fs, max_bnr = csi_pipeline(csi_frames, self._pca)
if max_bnr < BNR_GATE_THRESH:
    return  # no fusion, fused BPM not updated
```

In this case, the fused BPM label continues to show the last valid value.

---

## 11. Real-Time System Architecture

### 11.1 QThread Structure

```
Main Thread (GUI)
├── MainWindow (PyQt5)
│   ├── BPM labels (×3: Camera, CSI, Fused)
│   ├── 5 pyqtgraph plots (raw CSI, SG, PC1, raw shoulder, filtered)
│   └── Status bar (last status message)
│
├── CameraThread (QThread)           ← captures camera frames
│   └── frame_ready → MainWindow._on_frame()
│
├── SerialThread (QThread)           ← reads CSI from COM10
│   └── buf_updated → CSI buffer counter updated
│
├── CameraBreathWorker (QThread)     ← camera breath processing
│   ├── QTimer (1000 ms)
│   └── result → MainWindow._on_cam_result()
│
├── CSIProcessingWorker (QThread)    ← CSI breath processing
│   ├── QTimer (1000 ms)
│   └── result → MainWindow._on_csi_result()
│
└── FusedBreathWorker (QThread)      ← dual-modality fusion
    ├── QTimer (1000 ms)
    └── result → MainWindow._on_fused_result()
```

### 11.2 Mutex Synchronization

```python
# CSI buffer
self._csi_buf  = collections.deque(maxlen=BUF_MAX)
self._csi_lock = QMutex()

# Camera buffer
self._cam_buf  = collections.deque(maxlen=CAM_BUF_MAX)
self._cam_lock = QMutex()

# Safe access
with QMutexLocker(self._csi_lock):
    frames = list(self._csi_buf)
```

### 11.3 Signal Types

```python
# CSIProcessingWorker
result = pyqtSignal(object, object, object, float, float)
#                   raw_mean sg_mean pc1    bpm   fs
# bpm = -1.0 → BNR gate activated

# CameraBreathWorker
result = pyqtSignal(object, object, float, float)
#                   raw_y  filtered bpm   fs

# FusedBreathWorker
result = pyqtSignal(float, float, float, float)
#                   fused  cam    csi    agreement
```

### 11.4 Minimum Data Requirements

```python
MIN_CSI_SAMP = FS_NOMINAL × 5 = 13 × 5 = 65 packets (~5 s)
MIN_CAM_SAMP = 75 frames (~2.5 s @ 30 FPS)
```

Processing does not start until these thresholds are reached.

---

## 12. Offline Analysis Results

### 12.1 Dataset

| File | Duration | Fs (Hz) | Active SC | Content |
|------|----------|---------|-----------|---------|
| `csi_data_empty_my_room.csv` | 581 s | 11.71 | 115 | Empty room (reference) |
| `csi_data_long_breath.csv` | 1243 s | 11.78 | 116 | Long breathing (unknown rate) |
| `csi_data_25bpm_breath.csv` | 346 s | 11.82 | 115 | Controlled 25 BPM breathing |

**Recording conditions:** TX and RX ESP32 placed **side by side** (no LoS). Person breathed in front of ESP32s but not on the RF path.

### 12.2 Effect of BNR Band Change

| Method | Empty Room | Long Breath | 25 BPM |
|--------|-----------|-------------|--------|
| Naive (all SC, SG=51) | 11.0 BPM | 8.3 BPM | 9.7 BPM |
| SG=11 fix | 22.0 BPM | 23.5 BPM | 16.6 BPM |
| **BNR[0.16 Hz] + top-20** | **16.3 BPM** | **28.4 BPM** | **28.1 BPM** |
| Target | ~0 (noise) | Unknown | 25.0 BPM |

### 12.3 25 BPM File Detailed Analysis

**Top-20 subcarrier PC analysis with BNR[0.16–0.60 Hz]:**

```
PCA:  PC1=54.3% variance  PC2=19.8% variance  PC3=6.8% variance
      PC1: Welch=29.1 BPM   SampEn=2.615
      PC2: Welch=16.6 BPM   SampEn=2.442
      PC3: Welch=15.2 BPM   SampEn=2.639

VN-SampEn: PC1=4.82  PC2=12.32  PC3=39.01
→ PC1 selected (minimum VN-score)

FFT BPM = 28.07 BPM (0.4678 Hz)
Welch BPM = 29.1 BPM
```

**Interpretive note:** 28.07 BPM shows ~3 BPM deviation from the 25 BPM target. The empty room recording showed no strong signal in this frequency region (empty room: 16.3 BPM), supporting attribution of the 28 BPM signal to a real respiratory source. Possible causes of deviation: (a) actual respiration rate shifted from 25 to 28 BPM during recording, (b) partial signal distortion due to lack of LoS.

### 12.4 Long Breath File BPM Estimation

Despite the long breath file being labeled "unknown":
- Measured FFT BPM: **28.4 BPM**
- Different from empty room → not an environmental source
- Person was likely breathing at ~28 BPM

### 12.5 SampEn = ∞ Case

In earlier analysis of the 25 BPM file (before BNR band change), PC1 SampEn = ∞ was observed:
- `B = 0`: no m-dimensional template matches → signal is either too noisy or too uniform
- Workaround: `SampEn = inf` → treated as `10.0` in code
- This value can distort VN-SampEn score computation

---

## 13. Key Findings and Limitations

### 13.1 Technical Contributions

1. **BNR band cutoff (0.16 Hz):** Systematically eliminates HVAC/fan noise (8–10 BPM) at the subcarrier selection stage. Previous approaches included this frequency in the breath band.

2. **VN-SampEn:** Variance-normalized sample entropy prevents HVAC from being incorrectly selected as the best PC due to its low SampEn. Traditional minimum-SampEn selection failed in this scenario.

3. **Zero-padded FFT:** Overcomes the resolution limitation of Welch's method (especially Δf = 6 BPM on short data windows).

4. **BNR Presence Gate:** Prevents meaningless BPM output in empty room and weak signal conditions by blocking FFT computation when BNR < 1.5.

5. **Dual-modality geometric fusion:** If both modalities are strong at the same frequency, the peak grows; if one is weak, it drops. This improves reliability compared to single-modality systems.

### 13.2 Limitations and Open Problems

| Problem | Cause | Suggested Solution |
|---------|-------|-------------------|
| BNR_max < 1.0 (all recordings) | TX-RX side by side, outside Fresnel zone | Position person on TX-RX line of sight |
| 28 BPM instead of 25 BPM | No LoS → weak signal, drift | Target BNR ≥ 5 with ideal LoS |
| HVAC pollutes 22–23 BPM region | Some subcarriers still carry HVAC | Compute BNR_LO dynamically |
| SampEn computation slow (n_max=400) | O(N²) complexity | Use fast SampEn (CEEMDAN-based) |
| Welch resolution ≥ 1.4 BPM | `nperseg` limit | Use zero-padded FFT only |

### 13.3 Fresnel Zone Effect

The **first Fresnel ellipsoid** is critical for Wi-Fi CSI based respiration detection:

- **Case 1 (Ideal):** Person on TX-RX line of sight, inside Fresnel zone → high BNR (5–20)
- **Case 2 (Partial):** Person at edge of Fresnel zone → medium BNR (1–5)
- **Case 3 (This study):** TX-RX side by side, person outside Fresnel zone → low BNR (< 1)

First Fresnel ellipsoid radius (at center point):

```
r₁ = sqrt(λ × d / 4)
```

`λ = 0.125 m` (2.4 GHz Wi-Fi), `d = 2 m` (TX-RX distance):
```
r₁ = sqrt(0.125 × 2 / 4) = 0.25 m
```

The person must remain within this 25 cm radius ellipsoid.

---

## 14. Algorithm Parameter Reference Table

### Offline Analysis (`analyze_new_csi.py`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `TRIM_SEC` | 5 s | Unstable region at start and end |
| `BP_LOW` | 0.10 Hz | Bandpass filter lower bound |
| `BP_HIGH` | 0.60 Hz | Bandpass filter upper bound (25 BPM = 0.417 Hz included) |
| `BP_ORDER` | 4 | Butterworth filter order |
| `BNR_LO` | **0.16 Hz** | BNR calculation lower bound (HVAC excluded) |
| `BNR_HI` | 0.60 Hz | BNR calculation upper bound |
| `SG_WINDOW` | **11 samples** | Savitzky-Golay window (< T_25bpm = 29 samples) |
| `SG_ORDER` | 3 | Polynomial degree |
| `HAMPEL_HALF` | 50 samples | Hampel half-window (~4.2 s @ 12 Hz) |
| `HAMPEL_NSIG` | 2.5 | Outlier threshold (MAD multiplier) |
| `TOP_N_SC` | **20** | Number of subcarriers selected by BNR |
| `N_PCS` | 3 | PCA component count |
| `SAMPEN_M` | 3 | Sample Entropy embedding dimension |
| `SAMPEN_R_COEF` | 0.1 | SampEn tolerance coefficient (r = 0.1 × std) |
| `SAMPEN_N_MAX` | 400 | Max samples for SampEn speed |
| `FFT_NFFT` | 4096 | Zero-padded FFT points |

### Real-Time System (`csi_dual_monitor.py`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `RX_PORT` | COM10 | Receiver ESP32 serial port |
| `RX_BAUD` | 115200 | Serial port baud rate |
| `FS_NOMINAL` | 13 Hz | Nominal CSI sampling rate |
| `WINDOW_S` | 15 s | Processing window length |
| `BREATH_LO` | 0.05 Hz | Bandpass lower bound (real-time) |
| `BREATH_HI` | 0.40 Hz | Bandpass upper bound (real-time) |
| `PROC_INTERVAL` | 1000 ms | Processing trigger period |
| `BNR_GATE_THRESH` | **1.5** | Gate threshold (BPM not shown below this) |
| `BUF_MAX` | 300 packets | CSI circular buffer size |
| `MIN_CSI_SAMP` | 65 packets | Minimum packets to start processing |

---

## References

- **RespirFi:** Reference architecture for Wi-Fi CSI based respiration monitoring; subcarrier selection + PCA approach
- **ESP-NOW:** Espressif's connectionless Wi-Fi protocol; uses standard 802.11n infrastructure for CSI telemetry
- **Welch PSD:** P.D. Welch, 1967 — PSD estimation with overlapping windowed FFT
- **Sample Entropy:** Richman & Moorman, 2000 — complexity measurement in biomedical time series
- **MediaPipe Pose Landmarker:** Google, 2023 — 33 keypoints, Tasks API v0.10.x

---

*Created: 2026-06-13 | System: 2× ESP32 + CSI + MediaPipe | Platform: Windows 11, Python 3.x, PyQt5*
