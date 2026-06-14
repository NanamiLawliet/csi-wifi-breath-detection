# Çift Modaliteli Wi-Fi CSI Nefes Takip Sistemi — Sistem Dokümantasyonu

## İçindekiler
1. [Genel Bakış](#genel-bakış)
2. [Donanım Mimarisi](#donanım-mimarisi)
3. [Yazılım Mimarisi](#yazılım-mimarisi)
4. [Veri Akışı: Uçtan Uca](#veri-akışı-uçtan-uca)
5. [Modül 1: Kamera İş Parçacığı](#modül-1-kamera-iş-parçacığı)
6. [Modül 2: Seri Port İş Parçacığı](#modül-2-seri-port-iş-parçacığı)
7. [Modül 3: Sinyal İşleme Worker](#modül-3-sinyal-işleme-worker)
8. [Modül 4: Grafik Arayüzü](#modül-4-grafik-arayüzü)
9. [CSI Verisi: Format ve Parse](#csi-verisi-format-ve-parse)
10. [Sinyal İşleme Pipeline'ı](#sinyal-işleme-pipelineı)
11. [Thread Güvenliği](#thread-güvenliği)
12. [Sistem Parametreleri](#sistem-parametreleri)
13. [Kurulum ve Çalıştırma](#kurulum-ve-çalıştırma)
14. [Sorun Giderme](#sorun-giderme)

---

## Genel Bakış

Bu sistem, iki ESP32 mikrodenetleyicisi arasındaki Wi-Fi sinyalinin **Kanal Durum Bilgisi (CSI — Channel State Information)** verisini analiz ederek kişinin nefes hızını gerçek zamanlı olarak ölçer. Kamera yalnızca görsel referans amacıyla kullanılır; nefes tahmini **sadece CSI sinyalinden** yapılır.

```
[TX ESP32] ──ESP-NOW──► [RX ESP32] ──USB Serial──► [PC: Python GUI]
                              ↑
                        İki cihaz arasında
                        oturan kişi nefes alır
                        → CSI dalgalanır
```

---

## Donanım Mimarisi

### Cihazlar

| Cihaz | Rol | Bağlantı | MAC Adresi |
|-------|-----|----------|-----------|
| TX ESP32 | Verici (Master) | Bağımsız (USB yok) | `08:D1:F9:F6:7C:EC` |
| RX ESP32 | Alıcı (Slave) | USB → PC Seri Port | `68:FE:71:0B:A4:00` |

### Fiziksel Yerleşim

```
[TX ESP32]  ←— ~1 metre —→  [SEN]  ←— ~1 metre —→  [RX ESP32]
                              ↑
                        Kişi burada oturur
                        (nefes alıp verir)
```

TX ve RX arasındaki mesafe ~1 metredir. Kişi ortada oturduğunda nefes hareketi göğüs/karın bölgesinin genişleyip daralmasına neden olur; bu da iki cihaz arasındaki radyo yolunu hafifçe değiştirerek CSI'yi modüle eder.

### Firmware Ayarları (main.c)

```c
#define ESPNOW_CHANNEL      1
#define SEND_INTERVAL_MS    50      // 20 Hz nominal → PC'de ~13-14 Hz ölçüldü
#define CSI_MAX_DATA_LEN    256

wifi_csi_config_t csi_cfg = {
    .lltf_en         = true,   // Legacy LTF: 52 subcarrier
    .htltf_en        = true,   // HT LTF:    56 subcarrier
    .stbc_htltf2_en  = true,
    .ltf_merge_en    = true,
    .channel_filter_en = false,
    .manu_scale      = false,
    .shift           = 0,
};
```

- **Kanal genişliği:** 20 MHz (`WIFI_SECOND_CHAN_NONE`)
- **Phy modu:** HT20 (`WIFI_PHY_RATE_MCS0_LGI`)
- **CSI callback:** Yalnızca TX MAC + lokal MAC eşleşen paketleri işler

---

## Yazılım Mimarisi

### Thread Diyagramı

```
Ana Thread (Qt Event Loop)
│
├─► CameraThread  ─────────────────────────────────────────────────
│   QThread.run() döngüsü                                         │
│   • OpenCV webcam okuma (~30 FPS)                               │
│   • MediaPipe Pose iskelet algılama                             │
│   • frame_ready(QImage) sinyali emit                            │
│         └──── queued signal ───────────────► _on_frame() slot  │
│                                              GUI güncellenir    │
│                                                                  │
├─► SerialThread  ─────────────────────────────────────────────────
│   QThread.run() döngüsü                                         │
│   • pyserial.readline() bloklayan çağrı                         │
│   • CSI satırı parse et → amplitude vektörü                    │
│   • QMutexLocker ile deque'ye append                            │
│   • buf_updated(int) sinyali emit                               │
│         └──── queued signal ───────────────► tampon sayacı      │
│                                                                  │
├─► ProcessingThread (QThread + QObject Worker)  ─────────────────
│   QTimer (1000 ms) → ProcessingWorker.process()                 │
│   • deque snapshot (QMutex ile)                                 │
│   • Hampel → S-G → Bandpass → Scaler → PCA → BP → FFT         │
│   • result(ndarray, float, float) sinyali emit                  │
│         └──── queued signal ───────────────► _on_result() slot  │
│                                              BPM + grafik güncellenir
│
└─► GUI (MainWindow)
    • Kontrol çubuğu (port, baud, kamera)
    • Sol: kamera görüntüsü (QLabel)
    • Sağ: BPM etiketi (büyük font) + pyqtgraph dalga
    • Debug şeridi (ham satır / parse durumu)
```

### Signal/Slot Haberleşme Tablosu

| Sinyal | Kaynak Thread | Hedef Thread | Taşınan Veri |
|--------|--------------|--------------|--------------|
| `frame_ready` | CameraThread | Ana Thread | `QImage` (RGB frame) |
| `buf_updated` | SerialThread | Ana Thread | `int` (tampon boyutu) |
| `status` | SerialThread | Ana Thread | `str` (durum mesajı) |
| `debug` | SerialThread | Ana Thread | `str` (ham satır) |
| `result` | ProcessingThread | Ana Thread | `ndarray, float, float` |
| `status` | ProcessingWorker | Ana Thread | `str` (işlem durumu) |

Tüm cross-thread sinyaller Qt tarafından otomatik **queued connection** olarak işlenir → GUI asla donmaz.

---

## Veri Akışı: Uçtan Uca

```
TX ESP32
│  ESP-NOW paketi (50 ms aralıkla)
▼
RX ESP32  ──── wifi_csi_cb() tetiklenir
│  • MAC filtresi: sadece TX'ten gelen paketler
│  • csi_event_t yapısına kopyala
│  • FreeRTOS Queue'ya gönder (ISR'dan)
▼
csi_processing_task()
│  • Queue'dan oku
│  • JSON formatında seri porta yaz:
│    CSI_START{...,"csi_data":[v0..v255]}CSI_END
▼
USB Serial (115200 baud)
▼
Python: SerialThread.run()
│  • readline() → satır oku
│  • parse_csi_amplitudes() çağır
│    ├── CSI_START{...}CSI_END regex ile bul
│    ├── json.loads() ile parse et
│    ├── csi_data[4:] → ilk 4 header değer atlanır
│    ├── imag = vals[0::2],  real = vals[1::2]
│    └── amplitude = sqrt(imag² + real²)  → 126 float
│  • QMutexLocker ile deque'ye ekle (maxlen=300)
▼
ProcessingWorker.process()  (her 1 saniyede bir)
│  • deque snapshot → (N, 126) matris
│  • Null subcarrier sütunları sil (var < 1e-6)
│  • Hampel filtresi (her sütuna)
│  • Savitzky-Golay (her sütuna)
│  • Butterworth bandpass 0.05-0.40 Hz (her sütuna)
│  • StandardScaler (her sütunu normalize et)
│  • PCA → PC1 vektörü (1D nefes dalgası)
│  • PC1'e bandpass (ekstra temizlik)
│  • FFT → dominant frekans → BPM
▼
GUI (Ana Thread)
│  • Büyük BPM etiketi güncellenir
│  • pyqtgraph'ta nefes dalgası çizilir
▼
Kullanıcı
```

---

## Modül 1: Kamera İş Parçacığı

**Sınıf:** `CameraThread(QThread)`

### Görev
Yalnızca görsel referans. Nefes tahminine **katılmaz**.

### Çalışma Akışı

```python
while self._alive:
    ok, frame = cap.read()         # ~30 FPS
    ─► MediaPipe Pose işle
       ─► İskelet çiz (cv2.line / cv2.circle)
    ─► BGR → RGB dönüştür
    ─► QImage oluştur ve emit et
    self.msleep(33)                # ~30 FPS
```

### MediaPipe Sürüm Uyumluluğu

Sistem iki farklı MediaPipe API'sini otomatik dener:

```
1. Tasks API (mediapipe >= 0.10)
   └── PoseLandmarkerOptions + VIDEO modu
   └── Model: pose_landmarker_lite.task (~5 MB, otomatik indirilir)
   └── Landmark çizimi: cv2 ile manuel (_draw_pose_cv2)

2. Solutions API (mediapipe <= 0.9, legacy)
   └── mp.solutions.pose.Pose()
   └── mp.solutions.drawing_utils.draw_landmarks()
```

### İskelet Bağlantıları

MediaPipe 33 vücut noktası tanımlar. Bu noktalar 32 bağlantı çizgisiyle birleştirilir (omuzlar, kollar, bacaklar, gövde, yüz ana hatları).

---

## Modül 2: Seri Port İş Parçacığı

**Sınıf:** `SerialThread(QThread)`

### Güvenlik Özellikleri

| Durum | Davranış |
|-------|---------|
| Port açılamadı | Hata mesajı emit, 3 sn bekle, yeniden dene |
| Okuma hatası | Bağlantı kes, 3 sn bekle, yeniden bağlan |
| Bozuk satır | `continue` ile atla, çökmez |
| JSON parse hatası | `None` döner, satır atlanır |
| Yeterince kısa satır | Atla (len < 8) |

### Otomatik Yeniden Bağlanma

```python
while self._alive:
    try:
        ser = serial.Serial(port, baud, timeout=1.0)
        while self._alive:
            line = ser.readline()
            parse → deque
    except SerialException:
        msleep(3000)    # 3 sn bekle → döngü yeniden bağlanır
```

### Debug Çıktısı

İlk 5 ham satır ve ilk 3 başarılı parse arayüzdeki debug şeridinde gösterilir. Bu sayede baud rate doğru mu, format eşleşiyor mu anında görülebilir.

---

## Modül 3: Sinyal İşleme Worker

**Sınıf:** `ProcessingWorker(QObject)` → `QThread`'e taşındı

### Neden QObject + QThread (QThread subclass değil)?

Qt best-practice: `QObject.moveToThread()` ile worker `QThread`'in event loop'unda çalışır.  
`QTimer` da aynı thread'e taşınır → `QTimer.start()` ve `stop()` doğru thread'den çağrılır.

```python
self._proc_thr = QThread()
self._proc_wrk = ProcessingWorker(...)
self._proc_wrk.moveToThread(self._proc_thr)

self._proc_tmr = QTimer()
self._proc_tmr.moveToThread(self._proc_thr)
self._proc_thr.started.connect(self._proc_tmr.start)   # timer kendi thread'inde başlar
self._proc_thr.start()
```

### Durdurma (cross-thread killTimer önlemi)

```python
# YANLIŞ: self._proc_tmr.stop()  → farklı thread'den çağrı → Qt uyarısı
# DOĞRU:
QMetaObject.invokeMethod(self._proc_tmr, "stop", Qt.QueuedConnection)
```

`QueuedConnection` ile `stop()` çağrısı timer'ın kendi event loop'una kuyruğa alınır.

---

## Modül 4: Grafik Arayüzü

### Bileşenler

```
MainWindow (1440 × 820)
│
├── Kontrol Çubuğu
│   ├── RX Port seçici (QComboBox, editable — doğrudan "COM10" yazılabilir)
│   ├── ↺ Port yenile butonu
│   ├── Baud rate seçici (115200 / 921600 / 460800 / 230400)
│   ├── Kamera indeksi (QSpinBox, 0-5)
│   └── ▶ Başlat / ■ Durdur butonu
│
├── Debug Şeridi (tek satır, küçük font)
│   └── Ham satır / parse durumu / hata mesajları
│
└── QSplitter (yatay, 500:940)
    │
    ├── Sol: Kamera Paneli
    │   └── QLabel ← QPixmap ← QImage ← CameraThread
    │
    └── Sağ: Analiz Paneli
        ├── BPM Etiketi (font 46pt, yeşil/kırmızı)
        ├── pyqtgraph PlotWidget (filtrelenmiş PC1 dalgası)
        └── Alt bilgi (tampon sayacı, FS tahmini)
```

### BPM Renk Kodlaması

| BPM Aralığı | Renk | Anlam |
|------------|------|-------|
| 3 – 24 BPM | Yeşil (`#39d353`) | Normal solunum bandında |
| < 3 veya > 24 BPM | Kırmızı (`#f85149`) | Bant dışı (gürültü olabilir) |
| Hesaplanamadı | Yeşil, `— BPM` | Henüz yeterli veri yok |

---

## CSI Verisi: Format ve Parse

### Firmware Çıktısı (main.c, satır 263-283)

```
CSI_START{"rssi":-12,"rate":11,"channel":1,"bandwidth":20,
          "data_length":256,"esp_timestamp":2322828743,
          "csi_data":[75,-80,4,0,-24,-3,-23,-3,...,-42,5]}CSI_END
```

### csi_data Dizisi Yapısı (20 MHz, LLTF + HT-LTF)

```
İndeks  | İçerik
--------|----------------------------------------------------------
0-3     | Header / Pilot: [75, -80, 4, 0] → HER ZAMAN ATLANIR
4-53    | LLTF Negatif alt-taşıyıcılar (-26..-2): 25 I/Q çifti
54-75   | Null alt-taşıyıcılar (DC, guard band): 11 sıfır çifti
76-131  | LLTF Pozitif alt-taşıyıcılar (+2..+26): 28 I/Q çifti
132-133 | Separator [-1, -1]
134-253 | HT-LTF alt-taşıyıcıları: 60 I/Q çifti
254-255 | Trailing bytes
```

### I/Q'dan Genliğe

```
int8 dizisi:  [imag0, real0, imag1, real1, ..., imag125, real125]
                                                                  ↑
                                             indexler: 0,1,2,3,...,251

imag = vals[0::2]   → [imag0, imag1, ..., imag125]
real = vals[1::2]   → [real0, real1, ..., real125]
amp  = sqrt(imag² + real²)   → 126 float32 değer
```

### Null Subcarrier Filtreleme

PCA öncesinde sıfır-varyans sütunlar kaldırılır:

```python
active = np.where(np.var(mat, axis=0) > 1e-6)[0]
mat    = mat[:, active]    # sadece aktif sütunlar kalır
```

Tipik olarak ~52 aktif subcarrier kalır (LLTF negatif + pozitif alt-taşıyıcılar).

### Neden İlk 4 Değer Atlanır?

ESP32 CSI tamponunun başında her zaman `[75, -80, 4, 0]` gibi anormal değerler bulunur. Bunlar yüksek genlikli header/pilot sembollerine ait ve nefes sinyaliyle ilgisizdir. Offline analizde de (`analyze_csi.py`, satır 57-58) aynı şekilde atlanmaktadır:

```python
vals = vals[4:]   # İlk 4 değer (2 I/Q çifti) atlanır
```

---

## Sinyal İşleme Pipeline'ı

Her 1 saniyede bir tetiklenir. Giriş: son 15 saniyenin tamponu.

### Adım 0: Snapshot ve Ön Kontrol

```python
with QMutexLocker(lock):
    frames = list(deque)          # Thread-safe kopya

mat = np.array(frames)            # (N_paket × 126) matris
fs  = N_paket / WINDOW_S         # Gerçek örnekleme hızı tahmini
```

### Adım 1: Null Sütun Temizliği

DC bileşeni ve guard band'lara karşılık gelen sıfır sütunlar çıkarılır.

### Adım 2a: Hampel Filtresi

**Amaç:** Anlık RF parazitinden kaynaklanan aykırı değerleri (spike) tespit edip medyanla değiştir.

```
Parametreler:
  half_win = 10   → ±10 örnek pencere
  k        = 3.0  → medyan ± 3 × 1.4826 × MAD eşiği

Her subcarrier için bağımsız uygulanır.
```

**Neden gerekli:** ESP32 bazen tek bir pakette ani genlik sıçraması yapar (kanal karmaşıklığı, yeniden iletim). Bu spike'lar Savitzky-Golay'ı bozar; Hampel onları önceden temizler.

### Adım 2b: Savitzky-Golay Filtresi

**Amaç:** Sinyali polinomsal en küçük kareler uydurumuyla yumuşat.

```
Parametreler:
  window_length = min(51, n_samp//4 × 2 + 1)   → tek sayı, ≤ 51
  polyorder     = 3

Her subcarrier için bağımsız uygulanır.
```

**Neden gerekli:** Hampel sonrası kalan yüksek frekanslı gürültüyü kaldırır. Bandpass filtreden **önce** uygulanır çünkü pürüzlü sinyal Butterworth filtresinin transient davranışını bozabilir.

### Adım 2c: Butterworth Bant Geçiren Filtre

**Amaç:** Sadece solunum frekans bandını (0.05 – 0.40 Hz) geçir.

```
Parametreler:
  order  = 4
  low    = 0.05 Hz  → 3 BPM  (yavaş nefes için bu sınır gerekli)
  high   = 0.40 Hz  → 24 BPM
  method = filtfilt (sıfır-faz, iki-geçişli)
```

**Neden sıfır-faz:** `filtfilt` ileri-geri uygular → nefes dalgasında faz kayması olmaz → tepe noktaları gerçek zamana karşılık gelir.

**Her subcarriera ayrı uygulanır.** Bu sayede farklı subcarrierların farklı gürültü profillerine sahip olduğu durumlarda bile doğru bandpass yapılır.

### Adım 3: StandardScaler + PCA

**Amaç:** 52 aktif subcarrierin ortak varyansından en baskın bileşeni çıkar.

```python
X_sc = StandardScaler().fit_transform(mat)
# Her sütun: mean=0, std=1  →  farklı güçteki subcarrierlar eşitleniyor

pca  = PCA(n_components=1)
pc1  = pca.fit_transform(X_sc)[:, 0]
```

**Neden StandardScaler:** Bazı subcarrierların ortalama genliği diğerlerinin 5-10 katı olabilir. Ölçekleme yapılmazsa PCA bu güçlü kanalları dominanta seçer; ama bunlar her zaman nefes bilgisi taşımaz. Scaler tüm kanalları eşit ağırlığa getirir.

**Neden PCA(1):** Nefes hareketi tüm subcarrierleri eş fazlı ve benzer şekilde modüle eder. PC1, bu ortak modülasyonu temsil eder. Gürültü ise kanallar arasında rastgele dağılır → PC2, PC3, ... içinde kalır.

### Adım 3b: PC1'e Tekrar Bandpass

```python
pc1 = butter_bandpass(pc1, fs)
```

PCA sonrası PC1'de hala çok küçük miktarda bant dışı enerji kalabilir (sayısal hata, subcarrier seçim etkisi). Bu adım offline analizle (`analyze_csi.py`, satır 176) birebir aynıdır.

### Adım 3c: İşaret Tutarlılığı

```python
if abs(pc1.min()) > abs(pc1.max()):
    pc1 = -pc1
```

PCA işareti belirsizdir (PC veya -PC eşdeğer). Bu satır pozitif tepelerin nefes alma (genişleme) anına karşılık gelmesini sağlar.

### Adım 4: FFT → Dominant Frekans → BPM

```python
n     = len(pc1)
freqs = np.fft.rfftfreq(n, d=1.0 / fs)
mag   = abs(rfft(pc1 * hanning(n)))      # Hann penceresi ile spectral leakage azaltılır

mask        = (freqs >= 0.05) & (freqs <= 0.40)
dominant_hz = freqs[mask][argmax(mag[mask])]
bpm         = dominant_hz * 60.0
```

**Neden Hann penceresi:** Dikdörtgen pencere (`rfft` varsayılanı) kenarlarda spektral kaçak yapar. Hann penceresi bu etkiyi baskılar → frekans tespiti daha doğru.

**Frekans çözünürlüğü:**
```
Δf = fs / N = 13.5 Hz / (15 s × 13.5 Hz) ≈ 0.067 Hz ≈ 4 BPM
```
15 saniyelik pencere ile BPM hassasiyeti ±4 BPM seviyesindedir. Daha uzun pencere (30 s) daha hassas sonuç verir ama gecikmeyi artırır.

---

## Thread Güvenliği

### Paylaşılan Kaynak: `deque`

```
SerialThread      → append()
ProcessingWorker  → list(deque)  [snapshot]
```

Her iki erişim de `QMutexLocker` ile korunur:

```python
with QMutexLocker(self._lock):
    self._buf.append(amps)          # SerialThread

with QMutexLocker(self._lock):
    frames = list(self._buf)        # ProcessingWorker
    n      = len(self._buf)
```

`QMutexLocker` bir context manager'dır; blok sonunda kilidi otomatik serbest bırakır. Kilit tutulma süresi minimumdur (sadece `append` veya `list()` kadar).

### GUI Thread Güvenliği

Qt'de **yalnızca ana thread GUI widget'larını güncelleyebilir.** Tüm cross-thread sinyal bağlantıları `Qt.AutoConnection` (varsayılan) ile kurulur; bu, iki thread farklıysa otomatik olarak `QueuedConnection`'a dönüşür. Böylece:

- `CameraThread` → `frame_ready` → ana thread'deki `_on_frame` slot
- `SerialThread` → `buf_updated` → ana thread'deki lambda
- `ProcessingWorker` → `result` → ana thread'deki `_on_result` slot

Hiçbiri birbirini bloklamaz.

---

## Sistem Parametreleri

| Parametre | Değer | Kaynak |
|-----------|-------|--------|
| `FS_NOMINAL` | 13 Hz | CSV timestamp analizi (~71 ms arası) |
| `WINDOW_S` | 15 saniye | Nefes sinyal penceresinin standart uzunluğu |
| `BUF_MAX` | 300 slot | 15 s × 20 Hz (güvenli üst sınır) |
| `BREATH_LO` | 0.05 Hz (3 BPM) | Çok yavaş nefes denemelerini kapsar |
| `BREATH_HI` | 0.40 Hz (24 BPM) | Normal aktivite üst sınırı |
| `PROC_INTERVAL` | 1000 ms | Her 1 saniyede bir BPM güncellenir |
| `MIN_SAMPLES` | 65 paket (~5 s) | İlk hesaplama için minimum veri |
| Hampel half_win | 10 | ±10 örnek = ±0.74 saniye |
| S-G window | ≤51 (adaptif) | N/4 ile sınırlandırılır |
| S-G polyorder | 3 | Kübik polinom yumuşatma |
| Butterworth order | 4 | -80 dB/dekad kesim eğimi |
| PCA bileşeni | 1 (PC1) | Nefes modülasyonu dominant bileşen |

---

## Kurulum ve Çalıştırma

### Gerekli Python Paketleri

```bash
pip install pyqt5 pyqtgraph pyserial opencv-python mediapipe numpy scipy scikit-learn
```

### RX ESP32 Bağlantısı

1. RX ESP32'yi USB ile PC'ye bağla
2. Uygun COM portunu Aygıt Yöneticisi'nden öğren (ör. `COM10`)
3. Baud rate: **115200** (ESP32 varsayılan `printf` hızı)
   - Eğer veri gelmezse **921600** dene (bazı firmware'lerde yüksek baud kullanılır)

### Çalıştırma

```bash
cd Desktop/ehb/ehb440/experiment
python csi_dual_monitor.py
```

### Başlatma Adımları

1. Uygulama açılır → konsol mevcut seri portları listeler
2. **RX Port** açılır menüsünden COM portunu seç (veya doğrudan yaz)
3. **Baud** hızını seç (115200 ile başla)
4. **▶ Başlat** butonuna tıkla
5. Debug şeridinde `HAM[1]: CSI_START{...}CSI_END` görünmeli
6. ~5 saniye sonra `Veri bekleniyor… 65/65` tamamlanır ve BPM hesaplanmaya başlar

---

## Sorun Giderme

### Debug Şeridinde "CSI_START YOK"

```
Sebep  : Yanlış baud rate veya yanlış port
Çözüm  : 115200 ile dene → çalışmazsa 921600 ile dene
         Konsolda "Mevcut seri portlar:" listesine bak
```

### Debug Şeridinde "Parse hatası"

```
Sebep  : Kısmi satır okundu (buffer overflow veya USB latency)
Çözüm  : Genellikle geçici, birkaç paket sonra düzelir
         RX ESP32'yi yeniden başlat
```

### "Aktif subcarrier yok"

```
Sebep  : Tüm subcarrierlar sıfır → ESP32 CSI kaydedemiyor
Çözüm  : TX ESP32'nin çalıştığını ve aynı kanalde olduğunu doğrula
         RX ESP32 konsolunda CSI diagnostic taskını kontrol et
```

### "Veri bekleniyor" Mesajı Uzun Süre Devam Ediyor

```
Sebep  : Tampon dolu değil (MIN_SAMPLES = 65 paket ~ 5 saniye)
Çözüm  : Normal bekleme süresi. İlk BPM ~5-7 saniye sonra gelir.
```

### BPM Sonucu Mantıksız (örn. her zaman 3 veya 24 BPM)

```
Sebep  : FFT bandının uç noktasına yapışıyor → sinyal gürültülü
Çözüm  : Kişi ESP32'lerin tam ortasında oturmalı
         Ortamda büyük hareket eden cisimler olmamalı
         15 saniyelik pencere için sakin otur
```

### Kamera Görüntüsü Açılmıyor

```
Sebep  : Yanlış kamera indeksi veya MediaPipe modeli indirilemiyor
Çözüm  : Kamera indeksini 0, 1, 2 ile dene
         İnternet bağlantısını kontrol et (model indirme için)
         Konsoldaki [MediaPipe] hata mesajına bak
         Kamera olmadan CSI analizi çalışmaya devam eder
```

### `QObject::killTimer` Uyarısı

Bu uyarı düzeltilmiştir. `QMetaObject.invokeMethod(timer, "stop", Qt.QueuedConnection)` ile timer kendi thread'inden durdurulur.

---

*Belge otomatik olarak `csi_dual_monitor.py` kaynak kodu ve `main.c` firmware kodundan türetilmiştir.*
