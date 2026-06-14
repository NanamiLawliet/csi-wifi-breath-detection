#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual-Modal Wi-Fi CSI + Camera Breath Tracking System

Models:
  Camera   : Shoulder-Y movement -> Hampel -> SG -> Bandpass -> FFT -> BPM
  Wi-Fi CSI: ESP32 subcarrier    -> Hampel -> SG -> BP -> PCA -> FFT -> BPM
  Fused    : Geometric-mean spectrum -> consensus frequency -> BPM
               (product of both signal spectra; if both agree on a frequency
                the magnitude is high, if one disagrees it drops -> better accuracy)
"""

import json, os, re, sys, time, traceback, collections, urllib.request
from typing import Optional, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter, welch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────────────────────
#  HARDWARE SETTINGS  <-  only modify this block
# ─────────────────────────────────────────────────────────────────────────────
RX_PORT      = "COM11"
RX_BAUD      = 115200
CAMERA_INDEX = 0
# ─────────────────────────────────────────────────────────────────────────────

try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False

_MP_API  = None
_mp_pose = None
_mp_draw = None
MP_OK    = False

_HERE      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, "pose_landmarker_lite.task")
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)

if CV2_OK:
    try:
        import mediapipe as _mp_lib
        from mediapipe.tasks import python         as _mp_tasks
        from mediapipe.tasks.python import vision as _mp_vision
        _MP_API = 'tasks'
        MP_OK   = True
    except Exception:
        try:
            import mediapipe as _mp_lib
            _mp_pose = _mp_lib.solutions.pose
            _mp_draw = _mp_lib.solutions.drawing_utils
            _MP_API  = 'solutions'
            MP_OK    = True
        except Exception as _e:
            print(f"[MediaPipe] Failed to load: {_e}")

try:
    import serial
    SERIAL_OK = True
except ImportError:
    SERIAL_OK = False

import pyqtgraph as pg
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QHBoxLayout, QVBoxLayout, QLabel, QGroupBox,
)
from PyQt5.QtCore import (
    QThread, pyqtSignal, QTimer, Qt, QMetaObject,
    QMutex, QMutexLocker, QObject,
)
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline constants
# ─────────────────────────────────────────────────────────────────────────────
FS_NOMINAL    = 13
WINDOW_S      = 15
BUF_MAX       = WINDOW_S * 20
BREATH_LO     = 0.05
BREATH_HI     = 0.40
PROC_INTERVAL = 1000

MIN_CSI_SAMP  = FS_NOMINAL * 5
CAM_BUF_MAX   = WINDOW_S * 30
MIN_CAM_SAMP  = 75

BNR_GATE_THRESH = 1.5   # below → Empty Room / Weak Signal

_CHEST_IDX = {0, 11, 12}
_POSE_CONN = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),(11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),
    (27,29),(28,30),(29,31),(30,32),(27,31),(28,32),
]

# Common frequency grid for spectral fusion (200 points)
_FUSE_FREQS = np.linspace(BREATH_LO, BREATH_HI, 200)


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────
def _ensure_model() -> bool:
    if os.path.exists(MODEL_PATH):
        return True
    print("[MediaPipe] Downloading model...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        return True
    except Exception as e:
        print(f"[MediaPipe] Could not download: {e}")
        return False


def _draw_pose_and_extract(frame: np.ndarray, landmarks) -> Optional[float]:
    h, w = frame.shape[:2]
    pts: dict = {}
    for i, lm in enumerate(landmarks):
        if getattr(lm, 'visibility', 1.0) < 0.35:
            continue
        pts[i] = (int(lm.x * w), int(lm.y * h))

    for a, b in _POSE_CONN:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], (0, 130, 255), 2, cv2.LINE_AA)

    for i, pt in pts.items():
        if i not in _CHEST_IDX:
            cv2.circle(frame, pt, 4, (0, 200, 80), -1, cv2.LINE_AA)
    for i in _CHEST_IDX:
        if i in pts:
            cv2.circle(frame, pts[i], 11, (0, 220, 255), -1, cv2.LINE_AA)

    shoulder_y: Optional[float] = None
    if 11 in pts and 12 in pts:
        lm11, lm12 = landmarks[11], landmarks[12]
        if getattr(lm11, 'visibility', 1.0) > 0.3 and \
           getattr(lm12, 'visibility', 1.0) > 0.3:
            shoulder_y = float((lm11.y + lm12.y) / 2.0)

        mx = (pts[11][0] + pts[12][0]) // 2
        my = (pts[11][1] + pts[12][1]) // 2
        cv2.line(frame, pts[11], pts[12], (0, 220, 255), 3, cv2.LINE_AA)
        cv2.circle(frame, (mx, my), 9, (0, 220, 255), -1, cv2.LINE_AA)
        cv2.putText(frame, "BREATH TRACKING", (mx - 62, my - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2,
                    cv2.LINE_AA)
    return shoulder_y


# ─────────────────────────────────────────────────────────────────────────────
# DSP primitifler
# ─────────────────────────────────────────────────────────────────────────────
def hampel_filter(x: np.ndarray, half_win: int = 10, k: float = 3.0) \
        -> np.ndarray:
    x = x.copy()
    n = len(x)
    for i in range(n):
        lo, hi = max(0, i - half_win), min(n, i + half_win + 1)
        w   = x[lo:hi]
        med = np.median(w)
        mad = np.median(np.abs(w - med))
        if mad > 0 and abs(x[i] - med) > k * 1.4826 * mad:
            x[i] = med
    return x


def butter_bandpass(x: np.ndarray, fs: float) -> np.ndarray:
    nyq = fs / 2.0
    lo  = max(BREATH_LO / nyq, 1e-6)
    hi  = min(BREATH_HI / nyq, 0.9999)
    if lo >= hi:
        return x
    try:
        b, a = butter(4, [lo, hi], btype='band')
        return filtfilt(b, a, x)
    except Exception:
        return x


def compute_bnr(x: np.ndarray, fs: float) -> float:
    """Breathing-to-Noise Ratio: breath band PSD energy / out-of-band PSD energy."""
    nperseg = min(len(x), max(64, int(fs * 8)))
    f, p    = welch(x, fs=fs, nperseg=nperseg)
    in_b    = (f >= BREATH_LO) & (f <= BREATH_HI)
    out_b   = ~in_b & (f > 0)
    breath  = float(np.trapezoid(p[in_b],  f[in_b]))  if in_b.any()  else 0.0
    noise   = float(np.trapezoid(p[out_b], f[out_b])) if out_b.any() else 1e-10
    return breath / max(noise, 1e-10)


def dominant_bpm_fft(x: np.ndarray, fs: float) -> float:
    n     = len(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mag   = np.abs(np.fft.rfft(x * np.hanning(n)))
    mask  = (freqs >= BREATH_LO) & (freqs <= BREATH_HI)
    if not np.any(mask):
        return 0.0
    return float(freqs[mask][np.argmax(mag[mask])]) * 60.0


def parse_csi_amplitudes(line: str) -> Optional[np.ndarray]:
    try:
        m = re.search(r'CSI_START(\{.*\})CSI_END', line)
        if not m:
            return None
        obj  = json.loads(m.group(1))
        raw  = obj.get('csi_data')
        if not raw or len(raw) < 8:
            return None
        vals = np.array(raw[4:], dtype=np.float32)
        n    = (len(vals) // 2) * 2
        return np.sqrt(vals[0:n:2] ** 2 + vals[1:n:2] ** 2)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline helper functions  (used by both worker and fused)
# ─────────────────────────────────────────────────────────────────────────────
def cam_pipeline(samples: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Returns: (raw_y, filtered_y, fs)
    raw_y: inverted, DC-removed shoulder Y
    filtered_y: breath waveform after Hampel + SG + Bandpass
    """
    n  = len(samples)
    fs = float(np.clip(n / WINDOW_S, 5.0, 60.0))

    raw_y = -(samples - samples.mean())

    sig    = hampel_filter(raw_y, half_win=max(5, int(fs // 2)))
    sg_win = min(51, max(5, (n // 6) | 1))
    if n > sg_win:
        sig = savgol_filter(sig, sg_win, polyorder=3)

    return raw_y, butter_bandpass(sig, fs), fs


def csi_pipeline(frames: list, pca: PCA) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Returns: (raw_mean, sg_mean, pc1, fs)
    Raises: ValueError -- if no active subcarriers
    """
    min_len = min(len(f) for f in frames)
    mat     = np.array([f[:min_len] for f in frames], dtype=np.float64)
    n_samp  = mat.shape[0]
    fs      = float(np.clip(n_samp / WINDOW_S, 5.0, 200.0))

    active = np.where(np.var(mat, axis=0) > 1e-6)[0]
    if len(active) < 4:
        raise ValueError("no active subcarriers")
    mat   = mat[:, active]
    n_sub = mat.shape[1]

    raw_mean = mat.mean(axis=1).copy()

    for col in range(n_sub):
        mat[:, col] = hampel_filter(mat[:, col], half_win=10)

    sg_win = min(51, max(5, (n_samp // 4) | 1))
    if n_samp > sg_win:
        for col in range(n_sub):
            mat[:, col] = savgol_filter(mat[:, col], sg_win, polyorder=3)

    sg_mean = mat.mean(axis=1).copy()

    for col in range(n_sub):
        mat[:, col] = butter_bandpass(mat[:, col], fs)

    # BNR gate: compute BNR for each subcarrier → take the best
    bnr_vals = [compute_bnr(mat[:, col], fs) for col in range(n_sub)]
    max_bnr  = float(max(bnr_vals)) if bnr_vals else 0.0

    X_sc = StandardScaler().fit_transform(mat)
    pc1  = pca.fit_transform(X_sc)[:, 0]
    pc1  = butter_bandpass(pc1, fs)
    if np.abs(pc1.min()) > np.abs(pc1.max()):
        pc1 = -pc1

    return raw_mean, sg_mean, pc1, fs, max_bnr


def spectral_magnitude(x: np.ndarray, fs: float) -> np.ndarray:
    """
    Interpolates normalized FFT magnitude in the breath band onto the _FUSE_FREQS grid.
    Normalized so that the peak value = 1.
    """
    n     = len(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mag   = np.abs(np.fft.rfft(x * np.hanning(n)))

    mask = (freqs >= BREATH_LO) & (freqs <= BREATH_HI)
    if not np.any(mask):
        return np.zeros_like(_FUSE_FREQS)

    f_band = freqs[mask]
    m_band = mag[mask]
    peak   = m_band.max()
    if peak > 0:
        m_band = m_band / peak

    return np.interp(_FUSE_FREQS, f_band, m_band, left=0.0, right=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Module 1: Camera QThread
# ─────────────────────────────────────────────────────────────────────────────
class CameraThread(QThread):
    frame_ready = pyqtSignal(QImage)
    error       = pyqtSignal(str)

    def __init__(self, cam_buf: collections.deque, cam_lock: QMutex,
                 parent=None):
        super().__init__(parent)
        self._cam_buf  = cam_buf
        self._cam_lock = cam_lock
        self._alive    = False
        self._last_y   = None

    def run(self):
        self._alive = True
        if not CV2_OK:
            self.error.emit("OpenCV not found.")
            return

        pose_ctx = None
        if MP_OK and _MP_API == 'tasks':
            if _ensure_model():
                try:
                    opts = _mp_vision.PoseLandmarkerOptions(
                        base_options=_mp_tasks.BaseOptions(
                            model_asset_path=MODEL_PATH),
                        running_mode=_mp_vision.RunningMode.VIDEO,
                        num_poses=1,
                        min_pose_detection_confidence=0.5,
                        min_pose_presence_confidence=0.5,
                        min_tracking_confidence=0.5,
                    )
                    pose_ctx = _mp_vision.PoseLandmarker.create_from_options(opts)
                except Exception as e:
                    self.error.emit(f"Pose could not be initialized: {e}")
        elif MP_OK and _MP_API == 'solutions':
            try:
                pose_ctx = _mp_pose.Pose(
                    static_image_mode=False, model_complexity=0,
                    smooth_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            except Exception as e:
                self.error.emit(f"Pose baslatilmadi: {e}")

        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            self.error.emit(f"Camera {CAMERA_INDEX} could not be opened.")
            if pose_ctx:
                pose_ctx.close()
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS,           30)

        while self._alive:
            ok, frame = cap.read()
            if not ok:
                self.msleep(30)
                continue

            shoulder_y = None
            if pose_ctx is not None:
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if _MP_API == 'tasks':
                        ts_ms  = int(time.monotonic() * 1000)
                        mp_img = _mp_lib.Image(
                            image_format=_mp_lib.ImageFormat.SRGB, data=rgb)
                        res       = pose_ctx.detect_for_video(mp_img, ts_ms)
                        landmarks = (res.pose_landmarks[0]
                                     if res.pose_landmarks else None)
                    else:
                        res       = pose_ctx.process(rgb)
                        landmarks = (res.pose_landmarks.landmark
                                     if res.pose_landmarks else None)

                    if landmarks:
                        shoulder_y = _draw_pose_and_extract(frame, landmarks)
                except Exception:
                    pass

            if shoulder_y is not None:
                self._last_y = shoulder_y
            if self._last_y is not None:
                with QMutexLocker(self._cam_lock):
                    self._cam_buf.append(self._last_y)

            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            self.frame_ready.emit(
                QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy())
            self.msleep(33)

        cap.release()
        if pose_ctx:
            pose_ctx.close()

    def stop(self):
        self._alive = False
        self.wait(2000)


# ─────────────────────────────────────────────────────────────────────────────
# Module 2: Serial Port QThread
# ─────────────────────────────────────────────────────────────────────────────
class SerialThread(QThread):
    buf_updated = pyqtSignal(int)
    status      = pyqtSignal(str)

    def __init__(self, buf: collections.deque, lock: QMutex, parent=None):
        super().__init__(parent)
        self._buf   = buf
        self._lock  = lock
        self._alive = False

    def run(self):
        self._alive = True
        if not SERIAL_OK:
            self.status.emit("ERROR: pyserial not installed!")
            return

        while self._alive:
            try:
                self.status.emit(f"Connecting -> {RX_PORT} @ {RX_BAUD}")
                ser    = serial.Serial(RX_PORT, RX_BAUD, timeout=1.0)
                n_raw  = 0
                self.status.emit(f"Connected: {RX_PORT}")

                while self._alive:
                    try:
                        raw = ser.readline()
                        if not raw:
                            continue
                        decoded = raw.decode('ascii', errors='ignore').strip()
                        if not decoded:
                            continue
                        n_raw += 1
                        amps = parse_csi_amplitudes(decoded)
                        if amps is not None:
                            with QMutexLocker(self._lock):
                                self._buf.append(amps)
                            self.buf_updated.emit(len(self._buf))
                        elif n_raw <= 8 and 'CSI_START' not in decoded:
                            print(f"[Serial] CSI_START missing [{n_raw}]: "
                                  f"{decoded[:90]}")
                    except serial.SerialException as exc:
                        self.status.emit(f"Read error: {exc}")
                        break
                    except Exception:
                        continue
                try:
                    ser.close()
                except Exception:
                    pass
            except serial.SerialException as exc:
                self.status.emit(f"Could not open port ({RX_PORT}): {exc}")
            except Exception as exc:
                self.status.emit(f"Error: {exc}")

            if self._alive:
                self.msleep(3000)

    def stop(self):
        self._alive = False
        self.wait(3000)


# ─────────────────────────────────────────────────────────────────────────────
# Module 3a: Camera Breath Worker
# ─────────────────────────────────────────────────────────────────────────────
class CameraBreathWorker(QObject):
    result = pyqtSignal(object, object, float, float)   # raw_y, bp_y, bpm, fs
    status = pyqtSignal(str)

    def __init__(self, cam_buf: collections.deque, cam_lock: QMutex,
                 parent=None):
        super().__init__(parent)
        self._buf  = cam_buf
        self._lock = cam_lock

    def process(self):
        with QMutexLocker(self._lock):
            n = len(self._buf)
            if n < MIN_CAM_SAMP:
                self.status.emit(
                    f"Camera: {n}/{MIN_CAM_SAMP} samples waiting...")
                return
            samples = np.array(list(self._buf), dtype=np.float64)

        try:
            raw_y, filtered, fs = cam_pipeline(samples)
            bpm = dominant_bpm_fft(filtered, fs)
            self.result.emit(raw_y.astype(np.float32),
                             filtered.astype(np.float32), bpm, fs)
            self.status.emit(
                f"Camera: FS~{fs:.1f}Hz | {len(samples)} frames | BPM={bpm:.1f}")
        except Exception:
            self.status.emit(
                f"Camera processing: {traceback.format_exc(limit=2)[:120]}")


# ─────────────────────────────────────────────────────────────────────────────
# Module 3b: CSI Processing Worker
# ─────────────────────────────────────────────────────────────────────────────
class CSIProcessingWorker(QObject):
    result = pyqtSignal(object, object, object, float, float)  # raw,sg,pc1,bpm,fs
    status = pyqtSignal(str)

    def __init__(self, buf: collections.deque, lock: QMutex, parent=None):
        super().__init__(parent)
        self._buf  = buf
        self._lock = lock
        self._pca  = PCA(n_components=1)

    def process(self):
        with QMutexLocker(self._lock):
            n = len(self._buf)
            if n < MIN_CSI_SAMP:
                self.status.emit(
                    f"CSI: {n}/{MIN_CSI_SAMP} packets waiting "
                    f"({n/FS_NOMINAL:.1f}/{MIN_CSI_SAMP/FS_NOMINAL:.0f}s)")
                return
            frames = list(self._buf)

        if min(len(f) for f in frames) < 4:
            return

        try:
            raw_mean, sg_mean, pc1, fs, max_bnr = csi_pipeline(frames, self._pca)
            exp_var = self._pca.explained_variance_ratio_[0] * 100

            if max_bnr < BNR_GATE_THRESH:
                # BNR gate: skip FFT, emit bpm=-1 signal
                self.result.emit(raw_mean.astype(np.float32),
                                 sg_mean.astype(np.float32),
                                 pc1.astype(np.float32), -1.0, fs)
                self.status.emit(
                    f"CSI: Empty Room / Weak Signal  "
                    f"BNR={max_bnr:.2f} (threshold={BNR_GATE_THRESH})")
                return

            bpm = dominant_bpm_fft(pc1, fs)
            self.result.emit(raw_mean.astype(np.float32),
                             sg_mean.astype(np.float32),
                             pc1.astype(np.float32), bpm, fs)
            self.status.emit(
                f"CSI: FS~{fs:.1f}Hz | {len(frames)}pkt | "
                f"PC1={exp_var:.0f}% | BPM={bpm:.1f} | BNR={max_bnr:.2f}")
        except Exception:
            self.status.emit(
                f"CSI processing: {traceback.format_exc(limit=3)[:160]}")


# ─────────────────────────────────────────────────────────────────────────────
# Module 3c: Fused Worker
#
# After both signals pass through the full pipeline, their FFT spectra are
# taken, normalized, and their geometric mean is computed.
# Result: the dominant frequency that BOTH signals show simultaneously.
#
# Agreement score: derived from the difference between the two signals' peak BPMs.
# ─────────────────────────────────────────────────────────────────────────────
class FusedBreathWorker(QObject):
    #          fused_bpm  cam_bpm  csi_bpm  agreement(0-1)
    result = pyqtSignal(float, float, float, float)
    status = pyqtSignal(str)

    def __init__(self,
                 cam_buf:  collections.deque, cam_lock:  QMutex,
                 csi_buf:  collections.deque, csi_lock:  QMutex,
                 parent=None):
        super().__init__(parent)
        self._cam_buf  = cam_buf
        self._cam_lock = cam_lock
        self._csi_buf  = csi_buf
        self._csi_lock = csi_lock
        self._pca      = PCA(n_components=1)

    def process(self):
        # Get camera data
        with QMutexLocker(self._cam_lock):
            n_cam = len(self._cam_buf)
            if n_cam < MIN_CAM_SAMP:
                return
            cam_samples = np.array(list(self._cam_buf), dtype=np.float64)

        # Get CSI data
        with QMutexLocker(self._csi_lock):
            n_csi = len(self._csi_buf)
            if n_csi < MIN_CSI_SAMP:
                return
            csi_frames = list(self._csi_buf)

        if min(len(f) for f in csi_frames) < 4:
            return

        try:
            # Camera pipeline -> breath waveform + FS
            _, cam_signal, cam_fs = cam_pipeline(cam_samples)

            # CSI pipeline -> PC1 breath waveform + FS + BNR
            _, _, csi_signal, csi_fs, max_bnr = csi_pipeline(csi_frames, self._pca)
        except Exception:
            return

        # BNR gate: skip fusion if CSI signal is not strong enough
        if max_bnr < BNR_GATE_THRESH:
            return

        try:
            # Normalize FFT spectra -> common grid
            cam_spec = spectral_magnitude(cam_signal, cam_fs)
            csi_spec = spectral_magnitude(csi_signal, csi_fs)

            # Geometric mean: MAX if both signals agree on a frequency
            # DROPS if one disagrees -> consensus frequency stands out
            fused_spec = np.sqrt(cam_spec * csi_spec)

            if fused_spec.max() == 0:
                return

            peak_idx   = int(np.argmax(fused_spec))
            fused_bpm  = float(_FUSE_FREQS[peak_idx] * 60.0)

            # Compute individual BPM values (for agreement score)
            cam_bpm = float(_FUSE_FREQS[np.argmax(cam_spec)] * 60.0)
            csi_bpm = float(_FUSE_FREQS[np.argmax(csi_spec)] * 60.0)

            # Agreement score: 0-1 range if two signals' peak BPM diff is at most 6 BPM
            agreement = float(max(0.0, 1.0 - abs(cam_bpm - csi_bpm) / 6.0))

            self.result.emit(fused_bpm, cam_bpm, csi_bpm, agreement)
            self.status.emit(
                f"Fused: Camera={cam_bpm:.1f} CSI={csi_bpm:.1f} "
                f"-> {fused_bpm:.1f} BPM  agreement={agreement*100:.0f}%"
            )
        except Exception:
            self.status.emit(
                f"Fusion processing: {traceback.format_exc(limit=2)[:120]}")


# ─────────────────────────────────────────────────────────────────────────────
# Module 4: UI
# ─────────────────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            f"Dual-Modal CSI Breath Tracking  --  RX: {RX_PORT} @ {RX_BAUD}"
        )
        self.resize(1440, 930)

        self._csi_buf  = collections.deque(maxlen=BUF_MAX)
        self._csi_lock = QMutex()
        self._cam_buf  = collections.deque(maxlen=CAM_BUF_MAX)
        self._cam_lock = QMutex()

        self._cam_thr    = None
        self._ser_thr    = None
        # Camera worker
        self._cam_pthr   = None
        self._cam_pwrk   = None
        self._cam_ptmr   = None
        # CSI worker
        self._csi_pthr   = None
        self._csi_pwrk   = None
        self._csi_ptmr   = None
        # Fused worker
        self._fus_pthr   = None
        self._fus_pwrk   = None
        self._fus_ptmr   = None

        self._build_ui()
        self._start()

    # ─────────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        root     = QWidget()
        root_lay = QVBoxLayout(root)
        root_lay.setContentsMargins(8, 8, 8, 4)
        root_lay.setSpacing(6)
        self.setCentralWidget(root)

        pg.setConfigOptions(antialias=True,
                            background="#0d1117", foreground="#c9d1d9")

        # ── Three BPM labels ──────────────────────────────────────────────────
        bpm_row = QHBoxLayout()
        bpm_row.setSpacing(8)

        self._cam_bpm_lbl = QLabel("Camera  --  - BPM")
        self._cam_bpm_lbl.setFont(QFont("Segoe UI", 30, QFont.Bold))
        self._cam_bpm_lbl.setAlignment(Qt.AlignCenter)
        self._cam_bpm_lbl.setFixedHeight(96)
        self._cam_bpm_lbl.setStyleSheet(self._sty_cam(ok=True))
        bpm_row.addWidget(self._cam_bpm_lbl, stretch=3)

        # Fused BPM -- center, larger
        self._fus_bpm_lbl = QLabel("Fused    --  - BPM")
        self._fus_bpm_lbl.setFont(QFont("Segoe UI", 36, QFont.Bold))
        self._fus_bpm_lbl.setAlignment(Qt.AlignCenter)
        self._fus_bpm_lbl.setFixedHeight(96)
        self._fus_bpm_lbl.setStyleSheet(self._sty_fused(agreement=1.0))
        bpm_row.addWidget(self._fus_bpm_lbl, stretch=4)

        self._csi_bpm_lbl = QLabel("Wi-Fi CSI  --  - BPM")
        self._csi_bpm_lbl.setFont(QFont("Segoe UI", 30, QFont.Bold))
        self._csi_bpm_lbl.setAlignment(Qt.AlignCenter)
        self._csi_bpm_lbl.setFixedHeight(96)
        self._csi_bpm_lbl.setStyleSheet(self._sty_csi(ok=True))
        bpm_row.addWidget(self._csi_bpm_lbl, stretch=3)

        root_lay.addLayout(bpm_row)

        # ── Content ───────────────────────────────────────────────────────────
        content = QHBoxLayout()
        content.setSpacing(8)
        root_lay.addLayout(content, stretch=1)

        # Left: camera
        cam_box = QGroupBox("Camera  --  Chest / Shoulder Tracking")
        cam_lay = QVBoxLayout(cam_box)
        cam_lay.setContentsMargins(4, 4, 4, 4)
        cam_lay.setSpacing(4)

        self._cam_lbl = QLabel("Starting...")
        self._cam_lbl.setAlignment(Qt.AlignCenter)
        self._cam_lbl.setMinimumSize(440, 340)
        self._cam_lbl.setStyleSheet(
            "background:#161b22;color:#8b949e;border:1px solid #30363d;")
        cam_lay.addWidget(self._cam_lbl, stretch=1)

        self._cam_info = QLabel("Camera FS: --  |  Buffer: 0")
        self._cam_info.setStyleSheet("color:#8b949e;font-size:11px;")
        cam_lay.addWidget(self._cam_info)
        content.addWidget(cam_box, stretch=5)

        # Right: 5 signal plots
        sig_box = QGroupBox("Signal Pipeline")
        sig_lay = QVBoxLayout(sig_box)
        sig_lay.setContentsMargins(4, 4, 4, 4)
        sig_lay.setSpacing(3)

        def _plot(title: str, color: str, ylabel: str = "a.u."):
            pw = pg.PlotWidget(title=title)
            pw.setLabel("left",   ylabel)
            pw.setLabel("bottom", "Sample")
            pw.showGrid(x=True, y=True, alpha=0.18)
            pw.setMaximumHeight(136)
            curve = pw.plot(pen=pg.mkPen(color=color, width=1.8))
            sig_lay.addWidget(pw)
            return curve

        self._cv_cam_raw = _plot(
            "Camera 1  --  Raw Shoulder-Y (inverted)",         "#58a6ff")
        self._cv_cam_bp  = _plot(
            "Camera 2  --  Hampel + S-G + Bandpass  (breath)", "#ffa657")
        self._cv_csi_raw = _plot(
            "CSI 1  --  Raw Subcarrier Amplitude Mean",        "#bc8cff")
        self._cv_csi_sg  = _plot(
            "CSI 2  --  After Hampel + Savitzky-Golay",        "#d2a8ff")
        self._cv_csi_pc1 = _plot(
            "CSI 3  --  Bandpass + PCA(PC1)  (breath)",        "#39d353",
            ylabel="PC1")

        footer = QHBoxLayout()
        self._csi_buf_lbl = QLabel(f"CSI buffer: 0/{BUF_MAX}")
        self._csi_buf_lbl.setStyleSheet("color:#8b949e;font-size:11px;")
        footer.addWidget(self._csi_buf_lbl)
        footer.addStretch()
        self._csi_fs_lbl = QLabel("CSI FS: -- Hz")
        self._csi_fs_lbl.setStyleSheet("color:#8b949e;font-size:11px;")
        footer.addWidget(self._csi_fs_lbl)
        sig_lay.addLayout(footer)

        content.addWidget(sig_box, stretch=9)

    # ── Styles ────────────────────────────────────────────────────────────────
    @staticmethod
    def _sty_cam(ok=True):
        fg = "#ffa657" if ok else "#f85149"
        bd = "#d1742f" if ok else "#da3633"
        return (f"color:{fg};background:#161b22;"
                f"border:2px solid {bd};border-radius:12px;padding:8px;")

    @staticmethod
    def _sty_csi(ok=True):
        fg = "#39d353" if ok else "#f85149"
        bd = "#238636" if ok else "#da3633"
        return (f"color:{fg};background:#161b22;"
                f"border:2px solid {bd};border-radius:12px;padding:8px;")

    @staticmethod
    def _sty_fused(agreement: float = 1.0, ok: bool = True):
        if not ok:
            return ("color:#f85149;background:#161b22;"
                    "border:2px solid #da3633;border-radius:12px;padding:8px;")
        if agreement > 0.70:
            fg, bd = "#a371f7", "#8957e5"   # bright violet  -- high agreement
        elif agreement > 0.40:
            fg, bd = "#9b72df", "#7048ba"   # mid violet     -- medium agreement
        else:
            fg, bd = "#cc9af7", "#553098"   # dim violet     -- low agreement
        return (f"color:{fg};background:#161b22;"
                f"border:2px solid {bd};border-radius:12px;padding:8px;")

    # ── Start / Stop ──────────────────────────────────────────────────────────
    def _start(self):
        self._cam_thr = CameraThread(self._cam_buf, self._cam_lock)
        self._cam_thr.frame_ready.connect(self._on_frame)
        self._cam_thr.error.connect(
            lambda m: self.statusBar().showMessage(f"[Camera] {m}"))
        self._cam_thr.start()

        self._ser_thr = SerialThread(self._csi_buf, self._csi_lock)
        self._ser_thr.buf_updated.connect(
            lambda n: self._csi_buf_lbl.setText(f"CSI buffer: {n}/{BUF_MAX}"))
        self._ser_thr.status.connect(self.statusBar().showMessage)
        self._ser_thr.start()

        self._cam_pthr, self._cam_pwrk, self._cam_ptmr = \
            self._make_worker(
                CameraBreathWorker(self._cam_buf, self._cam_lock),
                self._on_cam_result)

        self._csi_pthr, self._csi_pwrk, self._csi_ptmr = \
            self._make_worker(
                CSIProcessingWorker(self._csi_buf, self._csi_lock),
                self._on_csi_result)

        self._fus_pthr, self._fus_pwrk, self._fus_ptmr = \
            self._make_worker(
                FusedBreathWorker(
                    self._cam_buf, self._cam_lock,
                    self._csi_buf, self._csi_lock),
                self._on_fused_result)

    def _make_worker(self, wrk: QObject, result_slot):
        thr = QThread()
        wrk.moveToThread(thr)
        wrk.result.connect(result_slot)
        wrk.status.connect(self.statusBar().showMessage)
        tmr = QTimer()
        tmr.setInterval(PROC_INTERVAL)
        tmr.timeout.connect(wrk.process)
        tmr.moveToThread(thr)
        thr.started.connect(tmr.start)
        thr.start()
        return thr, wrk, tmr

    def _stop(self):
        for tmr, thr in [
            (self._cam_ptmr, self._cam_pthr),
            (self._csi_ptmr, self._csi_pthr),
            (self._fus_ptmr, self._fus_pthr),
        ]:
            if tmr and thr and thr.isRunning():
                QMetaObject.invokeMethod(tmr, "stop", Qt.QueuedConnection)

        for thr in [self._cam_thr, self._ser_thr]:
            if thr:
                thr.stop()

        for thr in [self._cam_pthr, self._csi_pthr, self._fus_pthr]:
            if thr:
                thr.quit()
                thr.wait(3000)

        (self._cam_thr, self._ser_thr,
         self._cam_pthr, self._cam_pwrk, self._cam_ptmr,
         self._csi_pthr, self._csi_pwrk, self._csi_ptmr,
         self._fus_pthr, self._fus_pwrk, self._fus_ptmr) = (None,) * 11

    # ── Slots ─────────────────────────────────────────────────────────────────
    def _on_frame(self, img: QImage):
        px = QPixmap.fromImage(img).scaled(
            self._cam_lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._cam_lbl.setPixmap(px)

    def _on_cam_result(self, raw_y, filtered_y, bpm: float, fs: float):
        bpm_r = int(round(bpm))
        if bpm_r > 0:
            self._cam_bpm_lbl.setText(f"Camera  --  {bpm_r} BPM")
            self._cam_bpm_lbl.setStyleSheet(
                self._sty_cam(ok=(3 <= bpm_r <= 24)))
        else:
            self._cam_bpm_lbl.setText("Camera  --  - BPM")
            self._cam_bpm_lbl.setStyleSheet(self._sty_cam(ok=True))

        self._cv_cam_raw.setData(raw_y)
        self._cv_cam_bp.setData(filtered_y)
        n = len(raw_y) if hasattr(raw_y, '__len__') else 0
        self._cam_info.setText(f"Camera FS ~ {fs:.1f} Hz  |  Buffer: {n}")

    def _on_csi_result(self, raw_mean, sg_mean, pc1, bpm: float, fs: float):
        bpm_r = int(round(bpm))
        if bpm < 0:
            # BNR gate triggered
            self._csi_bpm_lbl.setText("Wi-Fi CSI  --  0 BPM  (Empty Room / Weak Signal)")
            self._csi_bpm_lbl.setStyleSheet(self._sty_csi(ok=False))
        elif bpm_r > 0:
            self._csi_bpm_lbl.setText(f"Wi-Fi CSI  --  {bpm_r} BPM")
            self._csi_bpm_lbl.setStyleSheet(
                self._sty_csi(ok=(3 <= bpm_r <= 24)))
        else:
            self._csi_bpm_lbl.setText("Wi-Fi CSI  --  - BPM")
            self._csi_bpm_lbl.setStyleSheet(self._sty_csi(ok=True))

        self._cv_csi_raw.setData(raw_mean)
        self._cv_csi_sg.setData(sg_mean)
        self._cv_csi_pc1.setData(pc1)
        self._csi_fs_lbl.setText(f"CSI FS ~ {fs:.1f} Hz")

    def _on_fused_result(self, fused_bpm: float, cam_bpm: float,
                         csi_bpm: float, agreement: float):
        bpm_r = int(round(fused_bpm))
        in_range = 3 <= bpm_r <= 24
        if bpm_r > 0 and in_range:
            self._fus_bpm_lbl.setText(f"Fused  --  {bpm_r} BPM")
            self._fus_bpm_lbl.setStyleSheet(
                self._sty_fused(agreement=agreement, ok=True))
        elif bpm_r > 0:
            self._fus_bpm_lbl.setText(f"Fused  --  {bpm_r} BPM")
            self._fus_bpm_lbl.setStyleSheet(self._sty_fused(ok=False))
        else:
            self._fus_bpm_lbl.setText("Fused  --  - BPM")
            self._fus_bpm_lbl.setStyleSheet(self._sty_fused(agreement=1.0))

        self.statusBar().showMessage(
            f"Fused: {bpm_r} BPM  |  "
            f"Camera={cam_bpm:.1f}  CSI={csi_bpm:.1f}  "
            f"Agreement={agreement*100:.0f}%"
        )

    def closeEvent(self, event):
        self._stop()
        super().closeEvent(event)


# ─────────────────────────────────────────────────────────────────────────────
# Dark palette
# ─────────────────────────────────────────────────────────────────────────────
def _dark_palette() -> QPalette:
    pal = QPalette()
    for role, rgb in [
        (QPalette.Window,          (13,  17,  23)),
        (QPalette.WindowText,      (201, 209, 217)),
        (QPalette.Base,            (22,  27,  34)),
        (QPalette.AlternateBase,   (30,  37,  47)),
        (QPalette.Text,            (201, 209, 217)),
        (QPalette.Button,          (30,  37,  47)),
        (QPalette.ButtonText,      (201, 209, 217)),
        (QPalette.Highlight,       (31,  111, 235)),
        (QPalette.HighlightedText, (255, 255, 255)),
        (QPalette.ToolTipBase,     (30,  37,  47)),
        (QPalette.ToolTipText,     (201, 209, 217)),
    ]:
        pal.setColor(role, QColor(*rgb))
    return pal


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(f"[SYSTEM] RX={RX_PORT} @ {RX_BAUD}  |  Camera={CAMERA_INDEX}")
    print(f"[CSI]   FS~{FS_NOMINAL}Hz | BP={BREATH_LO}-{BREATH_HI}Hz | "
          f"Window={WINDOW_S}s")
    print("[CAM]   Shoulder-Y -> Hampel -> S-G -> Bandpass -> FFT -> BPM")
    print("[FUS]   Camera + CSI spectral geometric-mean -> BPM")

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setPalette(_dark_palette())
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
