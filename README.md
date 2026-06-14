# Wi-Fi CSI Based Respiration Monitoring System

Contactless real-time respiration rate (BPM) measurement using Wi-Fi CSI (Channel State Information) signals between two ESP32 devices and an RGB camera.

---

## Documentation

- [System Documentation](SYSTEM_DOCUMENTATION.md) — Architecture, modules, installation and troubleshooting
- [Technical System Documentation](SYSTEM_TECHNICAL_DOC.md) — Signal processing pipeline, algorithms, analysis results

---

## Project Files

### Python Software

| File | Description |
|------|-------------|
| `csi_dual_monitor.py` | Dual-modality real-time monitor (CSI + Camera) |
| `csi_live_monitor.py` | CSI-only live monitor |
| `csi_classifier.py` | CNN+BiLSTM respiration classifier (training + testing) |
| `csi_1d_bilstm.py` | 1D-CNN + BiLSTM hybrid model |
| `analyze_csi.py` | Basic CSI data analysis |
| `analyze_csi_advanced.py` | Advanced CSI analysis (BNR, PCA, FFT) |
| `analyze_new_csi.py` | Current analysis pipeline |

### Firmware

| File | Description |
|------|-------------|
| `main.c` | ESP32 CSI collection firmware (ESP-IDF) |

### Trained Models

| File | Description |
|------|-------------|
| `best_csi_model.keras` | CNN+BiLSTM Keras model |
| `best_1d_model.keras` | 1D-CNN+BiLSTM Keras model |
| `best_csi_model.pth` | PyTorch model weights |

### Datasets

| File | Description |
|------|-------------|
| `csi_data_5bpm_breath.csv` | 5 breaths/min measurement data |
| `csi_data_10bpm_breath.csv` | 10 breaths/min measurement data |
| `csi_data_25bpm_breath.csv` | 25 breaths/min measurement data |
| `csi_data_long_breath.csv` | Long-duration breath data |
| `csi_data_empty_room.csv` | Empty room reference data |
| `csi_data_empty_my_room.csv` | Empty room reference data (2nd environment) |

---

## Hardware

- **TX ESP32**: Transmits CSI packets (ESP-NOW)
- **RX ESP32**: Receives CSI, forwards to PC via USB Serial
- **PC**: Real-time processing and visualization with Python GUI
- **Camera**: Optional visual reference (MediaPipe Pose)

---

## Installation

```bash
pip install numpy scipy matplotlib pyserial opencv-python mediapipe tensorflow torch scikit-learn
```

For detailed setup see the [System Documentation](SYSTEM_DOCUMENTATION.md#installation-and-setup) section.

---

## Usage

```bash
# Dual-modality monitor (CSI + Camera)
python csi_dual_monitor.py

# CSI-only monitor
python csi_live_monitor.py

# Recorded data analysis
python analyze_new_csi.py
```
