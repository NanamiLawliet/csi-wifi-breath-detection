# Wi-Fi CSI Tabanlı Nefes Takip Sistemi

ESP32 çift cihaz arasındaki Wi-Fi CSI (Channel State Information) sinyali ve RGB kamera kullanarak temas gerektirmeden gerçek zamanlı nefes hızı (BPM) ölçümü.

---

## Belgeler

- [Sistem Dokümantasyonu (TR)](SISTEM_DOKUMANTASYONU.md) — Mimari, modüller, kurulum ve sorun giderme
- [Teknik Sistem Dokümantasyonu (TR)](SYSTEM_TECHNICAL_DOC.md) — Sinyal işleme pipeline, algoritmalar, analiz sonuçları

---

## Proje Dosyaları

### Python Yazılımı

| Dosya | Açıklama |
|-------|----------|
| `csi_dual_monitor.py` | Çift modaliteli gerçek zamanlı monitör (CSI + Kamera) |
| `csi_live_monitor.py` | Sadece CSI tabanlı canlı monitör |
| `csi_classifier.py` | CNN+BiLSTM nefes sınıflandırıcısı (eğitim + test) |
| `csi_1d_bilstm.py` | 1D-CNN + BiLSTM hibrit modeli |
| `analyze_csi.py` | Temel CSI veri analizi |
| `analyze_csi_advanced.py` | Gelişmiş CSI analizi (BNR, PCA, FFT) |
| `analyze_new_csi.py` | Güncel analiz pipeline'ı |

### Firmware

| Dosya | Açıklama |
|-------|----------|
| `main.c` | ESP32 CSI toplama firmware'i (ESP-IDF) |

### Eğitilmiş Modeller

| Dosya | Açıklama |
|-------|----------|
| `best_csi_model.keras` | CNN+BiLSTM Keras modeli |
| `best_1d_model.keras` | 1D-CNN+BiLSTM Keras modeli |
| `best_csi_model.pth` | PyTorch model ağırlıkları |

### Veri Setleri

| Dosya | Açıklama |
|-------|----------|
| `csi_data_5bpm_breath.csv` | 5 nefes/dk ölçüm verisi |
| `csi_data_10bpm_breath.csv` | 10 nefes/dk ölçüm verisi |
| `csi_data_25bpm_breath.csv` | 25 nefes/dk ölçüm verisi |
| `csi_data_long_breath.csv` | Uzun süreli nefes verisi |
| `csi_data_empty_room.csv` | Boş oda referans verisi |
| `csi_data_empty_my_room.csv` | Boş oda referans verisi (2. ortam) |

---

## Donanım

- **TX ESP32**: CSI paketleri gönderir (ESP-NOW)
- **RX ESP32**: CSI alır, USB Serial ile PC'ye iletir
- **PC**: Python GUI ile gerçek zamanlı işleme ve görselleştirme
- **Kamera**: Opsiyonel görsel referans (MediaPipe Pose)

---

## Kurulum

```bash
pip install numpy scipy matplotlib pyserial opencv-python mediapipe tensorflow torch scikit-learn
```

Detaylı kurulum için [Sistem Dokümantasyonu](SISTEM_DOKUMANTASYONU.md#kurulum-ve-çalıştırma) bölümüne bakın.

---

## Kullanım

```bash
# Çift modaliteli monitör (CSI + Kamera)
python csi_dual_monitor.py

# Sadece CSI monitörü
python csi_live_monitor.py

# Kayıtlı veri analizi
python analyze_new_csi.py
```
