# Dual-Modality Wi-Fi CSI Breath Monitoring System — System Documentation

## Table of Contents
1. [Overview](#overview)
2. [Hardware Architecture](#hardware-architecture)
3. [Software Architecture](#software-architecture)
4. [Data Flow: End-to-End](#data-flow-end-to-end)
5. [Module 1: Camera Thread](#module-1-camera-thread)
6. [Module 2: Serial Port Thread](#module-2-serial-port-thread)
7. [Module 3: Signal Processing Worker](#module-3-signal-processing-worker)
8. [Module 4: Graphical Interface](#module-4-graphical-interface)
9. [CSI Data: Format and Parsing](#csi-data-format-and-parsing)
10. [Signal Processing Pipeline](#signal-processing-pipeline)
11. [Thread Safety](#thread-safety)
12. [System Parameters](#system-parameters)
13. [Installation and Setup](#installation-and-setup)
14. [Troubleshooting](#troubleshooting)

---

## Overview

This system measures a person's respiration rate in real time by analyzing **Channel State Information (CSI)** from the Wi-Fi signal between two ESP32 microcontrollers. The camera is used only as a visual reference; respiration estimation is performed **exclusively from the CSI signal**.

```
[TX ESP32] ──ESP-NOW──► [RX ESP32] ──USB Serial──► [PC: Python GUI]
                              ↑
                        Person sitting between
                        the two devices breathes
                        → CSI fluctuates
```

---

## Hardware Architecture

### Devices

| Device | Role | Connection | MAC Address |
|--------|------|------------|-------------|
| TX ESP32 | Transmitter (Master) | Standalone (no USB) | `08:D1:F9:F6:7C:EC` |
| RX ESP32 | Receiver (Slave) | USB → PC Serial Port | `68:FE:71:0B:A4:00` |

### Physical Layout

```
[TX ESP32]  ←— ~1 meter —→  [PERSON]  ←— ~1 meter —→  [RX ESP32]
                                ↑
                        Person sits here
                        (breathing in and out)
```

The distance between TX and RX is ~1 meter. When the person sits in the middle, respiratory movement causes the chest/abdomen to expand and contract, slightly modifying the radio path between the two devices and modulating the CSI.

### Firmware Settings (main.c)

```c
#define ESPNOW_CHANNEL      1
#define SEND_INTERVAL_MS    50      // 20 Hz nominal → ~13-14 Hz measured on PC
#define CSI_MAX_DATA_LEN    256

wifi_csi_config_t csi_cfg = {
    .lltf_en         = true,   // Legacy LTF: 52 subcarriers
    .htltf_en        = true,   // HT LTF:    56 subcarriers
    .stbc_htltf2_en  = true,
    .ltf_merge_en    = true,
    .channel_filter_en = false,
    .manu_scale      = false,
    .shift           = 0,
};
```

- **Channel width:** 20 MHz (`WIFI_SECOND_CHAN_NONE`)
- **PHY mode:** HT20 (`WIFI_PHY_RATE_MCS0_LGI`)
- **CSI callback:** Processes only packets where TX MAC matches local MAC

---

## Software Architecture

### Thread Diagram

```
Main Thread (Qt Event Loop)
│
├─► CameraThread  ─────────────────────────────────────────────────
│   QThread.run() loop                                            │
│   • OpenCV webcam capture (~30 FPS)                            │
│   • MediaPipe Pose skeleton detection                          │
│   • emit frame_ready(QImage) signal                            │
│         └──── queued signal ───────────────► _on_frame() slot  │
│                                              GUI updated        │
│                                                                  │
├─► SerialThread  ─────────────────────────────────────────────────
│   QThread.run() loop                                            │
│   • pyserial.readline() blocking call                           │
│   • Parse CSI line → amplitude vector                          │
│   • QMutexLocker → append to deque                             │
│   • emit buf_updated(int) signal                               │
│         └──── queued signal ───────────────► buffer counter     │
│                                                                  │
├─► ProcessingThread (QThread + QObject Worker)  ─────────────────
│   QTimer (1000 ms) → ProcessingWorker.process()                 │
│   • deque snapshot (with QMutex)                               │
│   • Hampel → S-G → Bandpass → Scaler → PCA → BP → FFT        │
│   • emit result(ndarray, float, float) signal                  │
│         └──── queued signal ───────────────► _on_result() slot  │
│                                              BPM + plot updated
│
└─► GUI (MainWindow)
    • Control bar (port, baud, camera)
    • Left: camera image (QLabel)
    • Right: BPM label (large font) + pyqtgraph waveform
    • Debug strip (raw line / parse status)
```

### Signal/Slot Communication Table

| Signal | Source Thread | Target Thread | Payload |
|--------|--------------|---------------|---------|
| `frame_ready` | CameraThread | Main Thread | `QImage` (RGB frame) |
| `buf_updated` | SerialThread | Main Thread | `int` (buffer size) |
| `status` | SerialThread | Main Thread | `str` (status message) |
| `debug` | SerialThread | Main Thread | `str` (raw line) |
| `result` | ProcessingThread | Main Thread | `ndarray, float, float` |
| `status` | ProcessingWorker | Main Thread | `str` (processing status) |

All cross-thread signals are automatically handled by Qt as **queued connections** → the GUI never freezes.

---

## Data Flow: End-to-End

```
TX ESP32
│  ESP-NOW packet (every 50 ms)
▼
RX ESP32  ──── wifi_csi_cb() triggered
│  • MAC filter: only packets from TX
│  • Copy to csi_event_t struct
│  • Send to FreeRTOS Queue (from ISR)
▼
csi_processing_task()
│  • Read from Queue
│  • Write to serial port in JSON format:
│    CSI_START{...,"csi_data":[v0..v255]}CSI_END
▼
USB Serial (115200 baud)
▼
Python: SerialThread.run()
│  • readline() → read line
│  • call parse_csi_amplitudes()
│    ├── find CSI_START{...}CSI_END with regex
│    ├── parse with json.loads()
│    ├── csi_data[4:] → skip first 4 header values
│    ├── imag = vals[0::2],  real = vals[1::2]
│    └── amplitude = sqrt(imag² + real²)  → 126 floats
│  • append to deque with QMutexLocker (maxlen=300)
▼
ProcessingWorker.process()  (every 1 second)
│  • deque snapshot → (N, 126) matrix
│  • remove null subcarrier columns (var < 1e-6)
│  • Hampel filter (per column)
│  • Savitzky-Golay (per column)
│  • Butterworth bandpass 0.05–0.40 Hz (per column)
│  • StandardScaler (normalize each column)
│  • PCA → PC1 vector (1D breath waveform)
│  • bandpass on PC1 (extra cleanup)
│  • FFT → dominant frequency → BPM
▼
GUI (Main Thread)
│  • Large BPM label updated
│  • Breath waveform drawn in pyqtgraph
▼
User
```

---

## Module 1: Camera Thread

**Class:** `CameraThread(QThread)`

### Purpose
Visual reference only. Does **not participate** in respiration estimation.

### Workflow

```python
while self._alive:
    ok, frame = cap.read()         # ~30 FPS
    ─► Process MediaPipe Pose
       ─► Draw skeleton (cv2.line / cv2.circle)
    ─► Convert BGR → RGB
    ─► Create QImage and emit
    self.msleep(33)                # ~30 FPS
```

### MediaPipe Version Compatibility

The system automatically tries two different MediaPipe APIs:

```
1. Tasks API (mediapipe >= 0.10)
   └── PoseLandmarkerOptions + VIDEO mode
   └── Model: pose_landmarker_lite.task (~5 MB, auto-downloaded)
   └── Landmark drawing: manual with cv2 (_draw_pose_cv2)

2. Solutions API (mediapipe <= 0.9, legacy)
   └── mp.solutions.pose.Pose()
   └── mp.solutions.drawing_utils.draw_landmarks()
```

### Skeleton Connections

MediaPipe defines 33 body keypoints, connected by 32 lines (shoulders, arms, legs, torso, face outline).

---

## Module 2: Serial Port Thread

**Class:** `SerialThread(QThread)`

### Safety Features

| Condition | Behavior |
|-----------|----------|
| Port cannot open | Emit error message, wait 3 s, retry |
| Read error | Disconnect, wait 3 s, reconnect |
| Corrupt line | `continue` — does not crash |
| JSON parse error | Returns `None`, line is skipped |
| Line too short | Skip (len < 8) |

### Auto-Reconnect

```python
while self._alive:
    try:
        ser = serial.Serial(port, baud, timeout=1.0)
        while self._alive:
            line = ser.readline()
            parse → deque
    except SerialException:
        msleep(3000)    # wait 3 s → loop reconnects
```

### Debug Output

The first 5 raw lines and first 3 successful parses are shown in the debug strip in the UI, making it immediately visible whether the baud rate is correct and whether the format matches.

---

## Module 3: Signal Processing Worker

**Class:** `ProcessingWorker(QObject)` → moved to `QThread`

### Why QObject + QThread (not a QThread subclass)?

Qt best practice: `QObject.moveToThread()` runs the worker inside the `QThread`'s event loop.  
`QTimer` is also moved to the same thread → `QTimer.start()` and `stop()` are called from the correct thread.

```python
self._proc_thr = QThread()
self._proc_wrk = ProcessingWorker(...)
self._proc_wrk.moveToThread(self._proc_thr)

self._proc_tmr = QTimer()
self._proc_tmr.moveToThread(self._proc_thr)
self._proc_thr.started.connect(self._proc_tmr.start)   # timer starts in its own thread
self._proc_thr.start()
```

### Stopping (cross-thread killTimer guard)

```python
# WRONG: self._proc_tmr.stop()  → call from different thread → Qt warning
# CORRECT:
QMetaObject.invokeMethod(self._proc_tmr, "stop", Qt.QueuedConnection)
```

With `QueuedConnection`, the `stop()` call is queued into the timer's own event loop.

---

## Module 4: Graphical Interface

### Components

```
MainWindow (1440 × 820)
│
├── Control Bar
│   ├── RX Port selector (QComboBox, editable — type "COM10" directly)
│   ├── ↺ Refresh ports button
│   ├── Baud rate selector (115200 / 921600 / 460800 / 230400)
│   ├── Camera index (QSpinBox, 0-5)
│   └── ▶ Start / ■ Stop button
│
├── Debug Strip (single line, small font)
│   └── Raw line / parse status / error messages
│
└── QSplitter (horizontal, 500:940)
    │
    ├── Left: Camera Panel
    │   └── QLabel ← QPixmap ← QImage ← CameraThread
    │
    └── Right: Analysis Panel
        ├── BPM Label (46pt font, green/red)
        ├── pyqtgraph PlotWidget (filtered PC1 waveform)
        └── Footer (buffer counter, estimated FS)
```

### BPM Color Coding

| BPM Range | Color | Meaning |
|-----------|-------|---------|
| 3 – 24 BPM | Green (`#39d353`) | Within normal respiration band |
| < 3 or > 24 BPM | Red (`#f85149`) | Out of band (may be noise) |
| Cannot compute | Green, `— BPM` | Not enough data yet |

---

## CSI Data: Format and Parsing

### Firmware Output (main.c, lines 263–283)

```
CSI_START{"rssi":-12,"rate":11,"channel":1,"bandwidth":20,
          "data_length":256,"esp_timestamp":2322828743,
          "csi_data":[75,-80,4,0,-24,-3,-23,-3,...,-42,5]}CSI_END
```

### csi_data Array Structure (20 MHz, LLTF + HT-LTF)

```
Index   | Content
--------|----------------------------------------------------------
0-3     | Header / Pilot: [75, -80, 4, 0] → ALWAYS SKIPPED
4-53    | LLTF Negative subcarriers (-26..-2): 25 I/Q pairs
54-75   | Null subcarriers (DC, guard band): 11 zero pairs
76-131  | LLTF Positive subcarriers (+2..+26): 28 I/Q pairs
132-133 | Separator [-1, -1]
134-253 | HT-LTF subcarriers: 60 I/Q pairs
254-255 | Trailing bytes
```

### I/Q to Amplitude

```
int8 array:  [imag0, real0, imag1, real1, ..., imag125, real125]
                                                                ↑
                                           indices: 0,1,2,3,...,251

imag = vals[0::2]   → [imag0, imag1, ..., imag125]
real = vals[1::2]   → [real0, real1, ..., real125]
amp  = sqrt(imag² + real²)   → 126 float32 values
```

### Null Subcarrier Filtering

Zero-variance columns are removed before PCA:

```python
active = np.where(np.var(mat, axis=0) > 1e-6)[0]
mat    = mat[:, active]    # only active columns remain
```

Typically ~52 active subcarriers remain (LLTF negative + positive subcarriers).

### Why Are the First 4 Values Skipped?

The ESP32 CSI buffer always contains abnormal values such as `[75, -80, 4, 0]` at the beginning. These belong to high-amplitude header/pilot symbols and are unrelated to the respiration signal. The same skip is applied in offline analysis (`analyze_csi.py`, lines 57–58):

```python
vals = vals[4:]   # Skip first 4 values (2 I/Q pairs)
```

---

## Signal Processing Pipeline

Triggered every 1 second. Input: last 15 seconds of buffer.

### Step 0: Snapshot and Pre-check

```python
with QMutexLocker(lock):
    frames = list(deque)          # Thread-safe copy

mat = np.array(frames)            # (N_packets × 126) matrix
fs  = N_packets / WINDOW_S       # Estimated real sampling rate
```

### Step 1: Null Column Removal

Zero columns corresponding to the DC component and guard bands are removed.

### Step 2a: Hampel Filter

**Purpose:** Detect instantaneous outliers (spikes) from RF interference and replace with the median.

```
Parameters:
  half_win = 10   → ±10 sample window
  k        = 3.0  → median ± 3 × 1.4826 × MAD threshold

Applied independently to each subcarrier.
```

**Why needed:** ESP32 occasionally produces sudden amplitude spikes in a single packet (channel complexity, retransmission). These spikes corrupt Savitzky-Golay; Hampel removes them first.

### Step 2b: Savitzky-Golay Filter

**Purpose:** Smooth the signal using polynomial least-squares fitting.

```
Parameters:
  window_length = min(51, n_samp//4 × 2 + 1)   → odd number, ≤ 51
  polyorder     = 3

Applied independently to each subcarrier.
```

**Why needed:** Removes remaining high-frequency noise after Hampel. Applied **before** the bandpass filter because a rough signal can distort the transient behavior of the Butterworth filter.

### Step 2c: Butterworth Bandpass Filter

**Purpose:** Pass only the respiration frequency band (0.05 – 0.40 Hz).

```
Parameters:
  order  = 4
  low    = 0.05 Hz  → 3 BPM  (needed to cover slow breathing)
  high   = 0.40 Hz  → 24 BPM
  method = filtfilt (zero-phase, two-pass)
```

**Why zero-phase:** `filtfilt` applies forward and backward → no phase shift in the breath waveform → peaks correspond to actual time.

**Applied separately to each subcarrier.** This ensures correct bandpass even when different subcarriers have different noise profiles.

### Step 3: StandardScaler + PCA

**Purpose:** Extract the dominant component from the shared variance of ~52 active subcarriers.

```python
X_sc = StandardScaler().fit_transform(mat)
# Each column: mean=0, std=1  →  equalize subcarriers of different strengths

pca  = PCA(n_components=1)
pc1  = pca.fit_transform(X_sc)[:, 0]
```

**Why StandardScaler:** Some subcarriers have average amplitudes 5–10× larger than others. Without scaling, PCA selects these dominant channels, which don't always carry respiration information. The scaler gives all channels equal weight.

**Why PCA(1):** Respiratory motion modulates all subcarriers in phase and similarly. PC1 represents this common modulation. Noise, however, is distributed randomly across channels → stays in PC2, PC3, ...

### Step 3b: Bandpass on PC1 Again

```python
pc1 = butter_bandpass(pc1, fs)
```

A very small amount of out-of-band energy may remain in PC1 after PCA (numerical error, subcarrier selection effect). This step is identical to offline analysis (`analyze_csi.py`, line 176).

### Step 3c: Sign Consistency

```python
if abs(pc1.min()) > abs(pc1.max()):
    pc1 = -pc1
```

PCA sign is ambiguous (PC or -PC are equivalent). This line ensures positive peaks correspond to inhalation (expansion).

### Step 4: FFT → Dominant Frequency → BPM

```python
n     = len(pc1)
freqs = np.fft.rfftfreq(n, d=1.0 / fs)
mag   = abs(rfft(pc1 * hanning(n)))      # Hann window reduces spectral leakage

mask        = (freqs >= 0.05) & (freqs <= 0.40)
dominant_hz = freqs[mask][argmax(mag[mask])]
bpm         = dominant_hz * 60.0
```

**Why Hann window:** The rectangular window (rfft default) causes spectral leakage at edges. The Hann window suppresses this → more accurate frequency detection.

**Frequency resolution:**
```
Δf = fs / N = 13.5 Hz / (15 s × 13.5 Hz) ≈ 0.067 Hz ≈ 4 BPM
```
With a 15-second window, BPM precision is ±4 BPM. A longer window (30 s) gives finer results but increases latency.

---

## Thread Safety

### Shared Resource: `deque`

```
SerialThread      → append()
ProcessingWorker  → list(deque)  [snapshot]
```

Both accesses are protected by `QMutexLocker`:

```python
with QMutexLocker(self._lock):
    self._buf.append(amps)          # SerialThread

with QMutexLocker(self._lock):
    frames = list(self._buf)        # ProcessingWorker
    n      = len(self._buf)
```

`QMutexLocker` is a context manager; it automatically releases the lock at the end of the block. Lock hold time is minimal (only the duration of `append` or `list()`).

### GUI Thread Safety

In Qt, **only the main thread can update GUI widgets.** All cross-thread signal connections are set up with `Qt.AutoConnection` (default), which automatically becomes `QueuedConnection` when the two threads differ. As a result:

- `CameraThread` → `frame_ready` → `_on_frame` slot in main thread
- `SerialThread` → `buf_updated` → lambda in main thread
- `ProcessingWorker` → `result` → `_on_result` slot in main thread

None of them block each other.

---

## System Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| `FS_NOMINAL` | 13 Hz | CSV timestamp analysis (~71 ms interval) |
| `WINDOW_S` | 15 seconds | Standard breath signal window length |
| `BUF_MAX` | 300 slots | 15 s × 20 Hz (safe upper limit) |
| `BREATH_LO` | 0.05 Hz (3 BPM) | Covers very slow breathing trials |
| `BREATH_HI` | 0.40 Hz (24 BPM) | Upper limit for normal activity |
| `PROC_INTERVAL` | 1000 ms | BPM updated every 1 second |
| `MIN_SAMPLES` | 65 packets (~5 s) | Minimum data to start first computation |
| Hampel half_win | 10 | ±10 samples = ±0.74 seconds |
| S-G window | ≤51 (adaptive) | Capped at N/4 |
| S-G polyorder | 3 | Cubic polynomial smoothing |
| Butterworth order | 4 | -80 dB/decade roll-off |
| PCA component | 1 (PC1) | Respiration modulation dominant component |

---

## Installation and Setup

### Required Python Packages

```bash
pip install pyqt5 pyqtgraph pyserial opencv-python mediapipe numpy scipy scikit-learn
```

### RX ESP32 Connection

1. Connect RX ESP32 to PC via USB
2. Find the correct COM port in Device Manager (e.g., `COM10`)
3. Baud rate: **115200** (ESP32 default `printf` speed)
   - If no data arrives, try **921600** (some firmware uses higher baud)

### Running

```bash
cd Desktop/ehb/ehb440/experiment
python csi_dual_monitor.py
```

### Startup Steps

1. Application opens → console lists available serial ports
2. Select COM port from **RX Port** dropdown (or type it directly)
3. Select **Baud** rate (start with 115200)
4. Click **▶ Start**
5. Debug strip should show `RAW[1]: CSI_START{...}CSI_END`
6. After ~5 seconds `Waiting for data… 65/65` completes and BPM calculation begins

---

## Troubleshooting

### Debug Strip Shows "NO CSI_START"

```
Cause  : Wrong baud rate or wrong port
Fix    : Try 115200 → if it doesn't work, try 921600
         Check "Available serial ports:" list in console
```

### Debug Strip Shows "Parse error"

```
Cause  : Partial line read (buffer overflow or USB latency)
Fix    : Usually temporary, resolves after a few packets
         Restart RX ESP32
```

### "No active subcarriers"

```
Cause  : All subcarriers are zero → ESP32 cannot record CSI
Fix    : Verify TX ESP32 is running and on the same channel
         Check CSI diagnostic task on RX ESP32 console
```

### "Waiting for data" Message Persists

```
Cause  : Buffer not full yet (MIN_SAMPLES = 65 packets ~ 5 seconds)
Fix    : Normal wait time. First BPM arrives after ~5-7 seconds.
```

### BPM Result Unreasonable (e.g., always 3 or 24 BPM)

```
Cause  : FFT sticking to the edge of the band → noisy signal
Fix    : Person should sit exactly between the two ESP32 devices
         No large moving objects in the environment
         Sit still for the 15-second window
```

### Camera Image Does Not Open

```
Cause  : Wrong camera index or MediaPipe model cannot be downloaded
Fix    : Try camera index 0, 1, 2
         Check internet connection (needed for model download)
         Check [MediaPipe] error message in console
         CSI analysis continues to work without a camera
```

### `QObject::killTimer` Warning

This warning has been fixed. The timer is stopped from its own thread using `QMetaObject.invokeMethod(timer, "stop", Qt.QueuedConnection)`.

---

*Documentation derived from `csi_dual_monitor.py` source code and `main.c` firmware.*
