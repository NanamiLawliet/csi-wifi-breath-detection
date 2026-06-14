#!/usr/bin/env python3
"""
ESP32 CSI Real-Time Live Monitor
─────────────────────────────────
Serial thread → sliding-window deque → pyqtgraph live plot
No heavy algorithm yet — just verifies data flows cleanly.
"""

import signal
import sys
import threading
import time
from collections import deque
from typing import Optional

import numpy as np
import serial
import serial.tools.list_ports
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg


# ─── User Configuration ───────────────────────────────────────────────────────

PORT       = "COM11"    # ← Change to your ESP32 COM port (e.g. "COM5", "/dev/ttyUSB0")
BAUD       = 115200    # ← Match your ESP32 firmware baud rate (common: 115200 / 921600)

FS_EST     = 100       # Estimated packet rate (Hz) — only used for time-axis scaling
WINDOW_SEC = 15        # Sliding window length in seconds
MAX_FRAMES = int(FS_EST * WINDOW_SEC)   # 1500 frames at 100 Hz

PLOT_SUB   = 30        # Which subcarrier index to display live (0-indexed, 0..N_SUB-1)
GUI_FPS    = 20        # Plot refresh rate (frames per second)

# ─────────────────────────────────────────────────────────────────────────────


def parse_csi_amplitude(line: str) -> Optional[np.ndarray]:
    """
    Parse one serial line from the ESP32 CSI firmware.

    Expected: any line containing a bracket-enclosed comma-separated int list,
    e.g.  "CSI_DATA,...,[0,2,-4,8,...]"
    Returns amplitude array (float32) or None on parse failure.

    The first two complex pairs are pilot tones with non-physical values;
    we skip the first 4 int16 values (same as the offline pipeline).
    """
    try:
        start = line.index("[")
        end   = line.index("]", start)
        raw   = np.fromstring(line[start + 1:end], dtype=np.int16, sep=",")
        raw   = raw[4:]                          # skip 2 pilot pairs
        I     = raw[0::2].astype(np.float32)
        Q     = raw[1::2].astype(np.float32)
        return np.sqrt(I ** 2 + Q ** 2)
    except (ValueError, IndexError):
        return None


# ─── Serial Reader Thread ─────────────────────────────────────────────────────

class SerialReader(threading.Thread):
    """
    Daemon thread — reads lines from the serial port and appends the
    full amplitude vector to the shared circular buffer.
    Dies automatically when the main process exits.
    """

    def __init__(self, port: str, baud: int,
                 buf: deque, lock: threading.Lock):
        super().__init__(daemon=True)
        self.port  = port
        self.baud  = baud
        self.buf   = buf
        self.lock  = lock
        self._stop = threading.Event()
        self.error: Optional[str] = None
        self.frame_count = 0

    def run(self):
        try:
            ser = serial.Serial(self.port, self.baud, timeout=1)
            print(f"[Serial] OK  →  {self.port} @ {self.baud} baud")
        except serial.SerialException as exc:
            self.error = str(exc)
            print(f"[Serial] FAIL  →  {exc}")
            return

        DEBUG_LINES = 200  # Print first N raw lines to console, then stop printing
        printed = 0

        with ser:
            while not self._stop.is_set():
                try:
                    raw_bytes = ser.readline()
                except serial.SerialException as exc:
                    self.error = str(exc)
                    break

                if not raw_bytes:
                    continue

                line = raw_bytes.decode("ascii", errors="ignore").strip()

                # ── Debug: show raw bytes (hex) + decoded text ────────────
                if printed < DEBUG_LINES:
                    print(f"[RAW {printed+1:02d}] hex={raw_bytes[:40].hex()}  |  txt={line[:120]!r}")
                    printed += 1
                elif printed == DEBUG_LINES:
                    print("[Serial] Debug done — parsing silently from here.")
                    printed += 1
                # ─────────────────────────────────────────────────────────────

                amps = parse_csi_amplitude(line)
                if amps is None:
                    continue

                with self.lock:
                    # Store full amplitude vector — easy to extend later
                    self.buf.append(amps)

                self.frame_count += 1

    def stop(self):
        self._stop.set()


# ─── Main Window ──────────────────────────────────────────────────────────────

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, buf: deque, lock: threading.Lock,
                 reader: SerialReader):
        super().__init__()
        self.buf    = buf
        self.lock   = lock
        self.reader = reader

        self.setWindowTitle(f"ESP32 CSI Live — Subcarrier #{PLOT_SUB}")
        self.resize(1100, 420)

        # ── pyqtgraph global style ─────────────────────────────────────────
        pg.setConfigOption("background", "#0d0d0d")   # near-black
        pg.setConfigOption("foreground", "#cccccc")   # light grey axes

        pw = pg.PlotWidget()
        self.setCentralWidget(pw)

        pw.setLabel("left",   "Amplitude (a.u.)",      color="#aaaaaa")
        pw.setLabel("bottom", "Time (s, 0 = now)",     color="#aaaaaa")
        pw.setTitle(
            f"Subcarrier #{PLOT_SUB}  —  live amplitude  "
            f"(window = {WINDOW_SEC} s,  ~{GUI_FPS} FPS)",
            color="#dddddd"
        )
        pw.showGrid(x=True, y=True, alpha=0.25)
        pw.setMouseEnabled(x=False, y=True)   # allow y-zoom, freeze x-pan

        self.curve = pw.plot(
            pen=pg.mkPen(color="#00BFFF", width=1.5)   # sky-blue line
        )

        # ── refresh timer ──────────────────────────────────────────────────
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(int(1000 / GUI_FPS))
        self.timer.timeout.connect(self._update)
        self.timer.start()

        self._prev_n = 0
        self._t0     = time.perf_counter()

    def _update(self):
        # Grab a snapshot while holding the lock as briefly as possible
        with self.lock:
            if not self.buf:
                return
            # Extract only the target subcarrier from every stored frame
            try:
                data = np.array([frame[PLOT_SUB] for frame in self.buf],
                                dtype=np.float32)
            except IndexError:
                return   # frames shorter than PLOT_SUB — firmware mismatch

        n = len(data)
        if n < 2:
            return

        # Relative time axis: newest sample sits at t=0, oldest at t=-WINDOW_SEC
        t = (np.arange(n) - n + 1) / FS_EST   # seconds, ≤ 0

        self.curve.setData(t, data)

        # Status bar — update only when frame count changes
        if n != self._prev_n:
            elapsed  = time.perf_counter() - self._t0
            live_fps = self.reader.frame_count / max(elapsed, 1e-3)
            self.statusBar().showMessage(
                f"Buffer: {n}/{MAX_FRAMES} frames  |  "
                f"Subcarrier: {PLOT_SUB}  |  "
                f"Incoming rate: {live_fps:.1f} pkt/s"
            )
            self._prev_n = n

    def closeEvent(self, event):
        self.timer.stop()
        event.accept()


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    # Print available ports so the user can identify the right COM number
    available = serial.tools.list_ports.comports()
    if available:
        print("Available serial ports:")
        for p in available:
            print(f"  {p.device:10s}  {p.description}")
    else:
        print("No serial ports found. Check USB connection.")

    print(f"\nConnecting to {PORT} @ {BAUD} baud …")
    print(f"Sliding window: {WINDOW_SEC} s  ({MAX_FRAMES} frames max)")
    print(f"Displaying subcarrier index: {PLOT_SUB}")
    print("Close the window to quit.\n")

    buf    = deque(maxlen=MAX_FRAMES)
    lock   = threading.Lock()

    reader = SerialReader(PORT, BAUD, buf, lock)
    reader.start()

    app    = QtWidgets.QApplication(sys.argv)
    signal.signal(signal.SIGINT, signal.SIG_DFL)  # Ctrl+C kills the process immediately
    window = MainWindow(buf, lock, reader)
    window.show()

    exit_code = app.exec_()

    reader.stop()
    reader.join(timeout=2)

    if reader.error:
        print(f"\n[Serial] Last error: {reader.error}")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
