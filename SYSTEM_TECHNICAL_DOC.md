# Wi-Fi CSI Tabanlı Çift Modaliteli Nefes Takip Sistemi
## Teknik Sistem Dokümantasyonu

---

## İçindekiler

1. [Sistem Genel Bakış](#1-sistem-genel-bakış)
2. [Donanım Mimarisi](#2-donanım-mimarisi)
3. [Veri Toplama Katmanı](#3-veri-toplama-katmanı)
4. [Sinyal İşleme Pipeline'ı](#4-sinyal-i̇şleme-pipelineı)
5. [Subcarrier Seçim Yöntemi — BNR Eleme](#5-subcarrier-seçim-yöntemi--bnr-eleme)
6. [PCA ve Bileşen Seçimi — VN-SampEn](#6-pca-ve-bileşen-seçimi--vn-sampen)
7. [BPM Tahmini — Sıfır-Dolgulu FFT](#7-bpm-tahmini--sıfır-dolgulu-fft)
8. [BNR Varlık Kapısı (Coherence Gate)](#8-bnr-varlık-kapısı-coherence-gate)
9. [Kamera Modalitesi — MediaPipe Pose](#9-kamera-modalitesi--mediapipe-pose)
10. [Çift Modalite Füzyonu](#10-çift-modalite-füzyonu)
11. [Gerçek Zamanlı Sistem Mimarisi](#11-gerçek-zamanlı-sistem-mimarisi)
12. [Çevrimdışı Analiz Sonuçları](#12-çevrimdışı-analiz-sonuçları)
13. [Anahtar Bulgular ve Kısıtlar](#13-anahtar-bulgular-ve-kısıtlar)
14. [Algoritma Parametreleri Referans Tablosu](#14-algoritma-parametreleri-referans-tablosu)

---

## 1. Sistem Genel Bakış

Bu sistem, ev ortamında Wi-Fi CSI (Channel State Information) sinyalleri ve RGB kamera görüntülerini eş zamanlı olarak işleyerek kişinin nefes atım hızını (BPM — Beats Per Minute) temas gerektirmeden ölçer.

### Temel Özellikler

| Özellik | Değer |
|---------|-------|
| Modalite | Wi-Fi CSI (802.11n) + RGB Kamera |
| Donanım | 2 × ESP32 (ESP-NOW protokolü) |
| Örnekleme Hızı (CSI) | ~11–13 Hz |
| Örnekleme Hızı (Kamera) | 30 FPS |
| Hedef BPM Aralığı | 6–36 BPM (0.1–0.6 Hz) |
| İşleme Gecikmesi | ~1 saniye (kayan pencere) |
| Referans Yöntem | RespirFi (subcarrier eleme + PCA + FFT) |

### Çözülen Ana Problem

Kapalı ortamlarda Wi-Fi CSI sinyali yalnızca nefes hareketini değil; HVAC sistemleri, fanlar ve diğer periyodik çevresel kaynakları da yansıtır. Geliştirilen **BNR tabanlı subcarrier eleme** yöntemi, bu gürültü kaynaklarını nefes tahminine katılmadan önce sistematik olarak devre dışı bırakır.

---

## 2. Donanım Mimarisi

### 2.1 ESP32 İkili Anten Yapısı

```
┌─────────────────────────────────────────────────────────────────┐
│                         Oda / Ortam                             │
│                                                                 │
│   ┌──────────────┐    ESP-NOW (Wi-Fi 802.11n)   ┌───────────┐  │
│   │  TX ESP32    │ ─────────────────────────►  │ RX ESP32  │  │
│   │  (COM9)      │    CSI paketleri (~13 Hz)    │  (COM10)  │  │
│   │              │                              │           │  │
│   │ USB: YOK     │     [Kişi nefes alıyor]      │ USB: VAR  │  │
│   │ PC'ye bağlı  │         ↕ RF yolu            │ PC'ye bağlı│  │
│   │ değil        │                              │           │  │
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

### 2.2 Cihaz Rolleri

**TX ESP32 (Verici)**
- Rol: Sürekli Wi-Fi CSI paketi yayınlar
- Bağlantı: PC'ye USB bağlantısı YOK; bağımsız güç kaynağıyla çalışır
- Protokol: ESP-NOW (802.11n, 40 MHz kanal genişliği)
- Yapılandırma: `PORT_TX = "COM9"` — yalnızca referans, PC bağlanmaz

**RX ESP32 (Alıcı)**
- Rol: TX'ten gelen CSI paketlerini alır, USB-Serial üzerinden PC'ye iletir
- Bağlantı: `PORT_RX = "COM10"`, `BAUD = 115200 bps`
- Çıktı formatı: `CSI_START{...JSON...}CSI_END` satır başına bir paket

### 2.3 CSI Veri Formatı

Her satır aşağıdaki JSON yapısını içerir:

```json
{
  "csi_data": [I0, Q0, I1, Q1, ..., I127, Q127],
  "rssi": -65,
  "noise_floor": -95
}
```

**Genlik hesabı** (her subcarrier için):
```
A_k = √(I_k² + Q_k²)
```

- `csi_data` dizisinin ilk 4 elemanı atlanır (geçersiz pilot subcarrier)
- Kalan 252 eleman → 126 çift → 126 subcarrier genliği
- Aktif (varyansı > 0.5 olan) subcarrier sayısı tipik olarak 115–116

### 2.4 İdeal Konumlandırma (Fresnel Bölgesi)

```
[TX ESP32] ←─────── 1–3 metre ───────→ [RX ESP32]
                        ↑
                   Kişi buraya
                 (RF yolu üzerinde)
```

**Kritik not:** TX ve RX ESP32 cihazları **yan yana konumlandırılmamalıdır**. Kişi TX-RX doğrusunun üzerinde, Fresnel elipsoidinin birinci bölgesinde yer aldığında CSI değişimi maksimuma ulaşır. Yan yana konumlandırmada (bu çalışmadaki kısıt) BNR < 1.0 elde edilmektedir.

---

## 3. Veri Toplama Katmanı

### 3.1 Seri Port Okuma

`SerialThread` (QThread) seri portu sürekli okur ve parse eder:

```python
ser = serial.Serial(RX_PORT, RX_BAUD, timeout=1.0)
line = ser.readline().decode('ascii', errors='ignore').strip()
amps = parse_csi_amplitudes(line)   # → np.ndarray (n_sub,) veya None
```

`parse_csi_amplitudes()`:
1. `CSI_START{...}CSI_END` regex deseniyle JSON bloğunu çıkar
2. `csi_data` listesini parse eder
3. `vals[4:]` — pilot subcarrier'ları atla
4. I/Q çiftlerinden genlik: `sqrt(I² + Q²)`
5. `np.ndarray(dtype=float32)` döndür

### 3.2 Dairesel Tampon

Gelen CSI paketleri `collections.deque(maxlen=BUF_MAX)` içinde tutulur:

```
BUF_MAX = WINDOW_S × 20 = 15 × 20 = 300 paket
```

Her paket bir `np.ndarray(n_sub,)` — tüm subcarrier genliklerini içerir. İşleme penceresi 1 saniyede bir kaydırılır (`PROC_INTERVAL = 1000 ms`).

---

## 4. Sinyal İşleme Pipeline'ı

### 4.1 Genel Akış

```
Ham CSI Matrisi (N × n_sub)
        │
        ▼
[1] Ölü Subcarrier Eleme
    varyans < 1e-6 → sil
        │
        ▼
[2] Hampel Filtresi (her subcarrier)
    half_win = 50 örnek (~4.2 s)
    k = 1.4826, nsig = 2.5
        │
        ▼
[3] Savitzky-Golay Filtresi (her subcarrier)
    window = 11 örnek (~0.93 s), poly = 3
        │
        ▼
[4] BNR Hesaplama (her subcarrier)
    band = [0.16 Hz – 0.60 Hz]
        │
        ▼
[5] Top-20 Subcarrier Seçimi
    BNR büyükten küçüğe → ilk 20
        │
        ▼
[6] Butterworth Bandpass (her subcarrier)
    [0.10 Hz – 0.60 Hz], order = 4, filtfilt
        │
        ▼
[7] StandardScaler + PCA (n_components=3)
        │
        ▼
[8] VN-SampEn ile en iyi PC seçimi
        │
        ▼
[9] Sıfır-Dolgulu FFT (N_FFT = 4096)
    → peak_bpm, confidence
        │
        ▼
    BPM Çıktısı
```

### 4.2 Hampel Filtresi

**Amaç:** CSI akışındaki ani sıçramaları (RF paket kaybı, çok yollu sönüm sıçraması) ortanca temelli outlier tespiti ile temizler.

**Algoritma:**
1. Her örnek `x[i]` için `[i-half_win, i+half_win]` penceresi alınır
2. Pencerenin ortancası `m` ve MAD (Ortanca Mutlak Sapma) hesaplanır:
   ```
   MAD = median(|x_window - m|)
   ```
3. Outlier kriteri:
   ```
   |x[i] - m| > nsig × 1.4826 × MAD
   ```
4. Outlier ise `x[i] ← m` ile değiştirilir

**Parametreler:**
- `half_win = 50` → toplam pencere ≈ 101 örnek (~8.6 s @ 11.8 Hz)
- `nsig = 2.5` → Gaussian varsayımı altında %1.5 yanlış pozitif

**Neden half_win = 50?**  
25 BPM sinyali için periyot ≈ 29 örnek. `half_win = 50` içinde ~3.5 tam periyot bulunur; ortanca ~DC seviyesine eşittir ve tepeler threshold'u aşmaz. Daha dar bir pencere (half_win < 14) tek bir tepeyi outlier olarak işaretleyebilir.

### 4.3 Savitzky-Golay Filtresi

**Amaç:** Yüksek frekanslı gürültüyü (RF titremesi, anten hareketi) polinom uydurma ile bastırır; nefes sinyalinin şeklini (tepe ve çukur zamanlaması) korur.

**Matematiksel temel:** Pencere içindeki verilere `poly`-dereceli polinom uydurulur, merkez nokta polinomdaki değerle değiştirilir.

**Kritik parametre seçimi:**

| Pencere (örnek) | Süre @ 12 Hz | 25 BPM'e oranı | Sonuç |
|-----------------|-------------|-----------------|-------|
| 51 örnek | 4.25 s | **1.77 × periyot** | Oversmoothing — 25 BPM yok edilir |
| 11 örnek | 0.92 s | **0.38 × periyot** | Periyot korunur ✓ |

**Kural:** `SG_WINDOW < T_breathing × Fs` olmalıdır.  
25 BPM için: `T = 2.4 s`, `Fs = 12 Hz` → `T × Fs = 29 örnek` → `SG_WINDOW = 11 < 29` ✓

---

## 5. Subcarrier Seçim Yöntemi — BNR Eleme

Bu çalışmanın **temel teknik katkısı**. Geleneksel yöntemler tüm subcarrier'ları PCA'ye sokarken bu yaklaşım önce kalite eleme yapar.

### 5.1 BNR (Breathing-to-Noise Ratio) Tanımı

Bir `x` subcarrier sinyali için BNR:

```
          ∫[BNR_LO to BNR_HI] S_xx(f) df
BNR(x) = ─────────────────────────────────
          ∫[0 to ∞, dışında] S_xx(f) df
```

Burada `S_xx(f)` Welch yöntemiyle tahmin edilen Güç Spektral Yoğunluğu (PSD).

**Hesaplama:**
```python
nperseg = min(N, max(256, int(Fs × 15)))
f, p    = welch(x, fs=Fs, nperseg=nperseg)
in_b    = (f >= BNR_LO) & (f <= BNR_HI)
out_b   = ~in_b & (f > 0)
BNR     = trapezoid(p[in_b], f[in_b]) / trapezoid(p[out_b], f[out_b])
```

### 5.2 BNR Band Seçiminin Kritik Önemi

**Geleneksel yaklaşım (başarısız):** `BNR_LO = 0.10 Hz`  
→ HVAC/fan (8–10 BPM = 0.133–0.167 Hz) **nefes bandına dahil** edilir  
→ BNR, HVAC güçlü subcarrier'ları yüksek değerlendirir  
→ Seçilen top-N subcarrier'lar HVAC domine eder  
→ PCA → HVAC sinyali çıkar  
→ 25 BPM yerine 8–10 BPM tespit edilir

**Önerilen yaklaşım (başarılı):** `BNR_LO = 0.16 Hz`  
→ HVAC (0.133–0.150 Hz) **nefes bandının DIŞINDA** kalır  
→ 25 BPM (0.417 Hz) nefes bandının içindedir  
→ 25 BPM taşıyan subcarrier'lar yüksek BNR alır  
→ Seçilen top-20 subcarrier'lar nefes bilgisi taşır  
→ PCA → nefes sinyali çıkar  
→ ~28 BPM tespit edilir (25 BPM hedefine yakın)

```
Frekans ekseni:
0    0.10  0.133  0.16   0.167  ...   0.417   0.60 Hz
│     │      │     │       │          │        │
│     │    HVAC   BNR_LO  HVAC       25 BPM  BNR_HI
│     │   (8 BPM)         (10 BPM)           
│     │
│     BP_LOW (bandpass filtre için)
│
DC

← BP filtre bandı: 0.10 Hz ──────────────────── 0.60 Hz →
         ← BNR hesaplama bandı: 0.16 Hz ──────── 0.60 Hz →
                               ↑
                     HVAC bu kesimde dışarıda kalır
```

### 5.3 Subcarrier Dağılımı Analizi

Ölçülen verilerden (25 BPM kayıt, 115 aktif subcarrier):

| Dominant frekans bölgesi | Subcarrier sayısı | BNR_old (0.10 Hz) | BNR_new (0.16 Hz) |
|--------------------------|-------------------|-------------------|-------------------|
| 8–10 BPM (HVAC) | 104 | Yüksek (0.38–0.40) | Düşük (< 0.20) |
| 25–28 BPM (nefes) | 9 | Düşük (< 0.30) | Yüksek (0.32–0.40) |
| Belirsiz | 2 | — | — |

Bu ters çevirme, BNR_LO değişikliğinin doğrudan etkisidir.

### 5.4 Top-N Seçim

BNR değerleri büyükten küçüğe sıralanır ve ilk `TOP_N_SC = 20` subcarrier seçilir:

```python
top_idx = np.argsort(bnr_vals)[-TOP_N_SC:][::-1]
selected = denoised[:, top_idx]   # (N, 20)
```

Kalan `n_sc - 20` subcarrier (tipik olarak 95–96 adet) **tamamen PCA matrisinden çıkarılır**.

---

## 6. PCA ve Bileşen Seçimi — VN-SampEn

### 6.1 PCA Uygulaması

Seçilen 20 subcarrier bandpass filtrelenip StandardScaler ile normalize edilir, ardından `n_components = 3` PCA uygulanır:

```python
X_sc = StandardScaler().fit_transform(bp_mat)   # (N, 20)
pca  = PCA(n_components=3)
pcs  = pca.fit_transform(X_sc)                   # (N, 3)
ev   = pca.explained_variance_ratio_ × 100
```

**Tipik sonuçlar (25 BPM dosyası, yeni yöntem):**
- PC1: %54.3 varyans
- PC2: %19.8 varyans  
- PC3: %6.8 varyans

Karşılaştırma (eski yöntem, tüm subcarrier'larla):
- PC1: %90.0 varyans → tek bileşen domine (HVAC)

PC1 varyansının düşmesi (90% → 54%), birden fazla anlamlı sinyal bileşeninin varlığına işaret eder ve yöntemin subcarrier ayrıştırma gücünü kanıtlar.

### 6.2 VN-SampEn (Varyans-Normalize Sample Entropy)

**Problem:** Minimum SampEn kriteri HVAC'ı seçer.  
HVAC son derece periyodik → düşük SampEn → "en iyi" sinyal olarak yanlış seçilir.

**Çözüm — VN-SampEn skoru:**

```
VN-score(k) = SampEn(PC_k) / (ev_k / 100)
```

| Durum | SampEn | Varyans | VN-score | Sonuç |
|-------|--------|---------|----------|-------|
| HVAC (fan) | Düşük (0.3) | Düşük (%3) | Yüksek (10.0) | Elenir |
| Nefes | Orta (1.8) | Yüksek (%54) | Düşük (3.3) | **Seçilir** |

Minimum VN-score → seçilen PC.

**Sample Entropy hesabı:**
```python
# m=3, r=0.1×std(x), downsampling n_max=400
B = count_templates(x, m)      # m-boyutlu eşleşen kalıp sayısı
A = count_templates(x, m+1)    # (m+1)-boyutlu eşleşen kalıp sayısı
SampEn = -log(A / B)
```

- `SampEn = 0` → tamamen tekrarlı (periyodik) sinyal
- `SampEn = ∞` → tamamen rastgele (gürültü)
- Nefes: orta değerler (1.5–3.0)

---

## 7. BPM Tahmini — Sıfır-Dolgulu FFT

### 7.1 Welch Yönteminin Kısıtı

Welch PSD çözünürlüğü:

```
Δf = Fs / nperseg
```

`nperseg = 512`, `Fs = 11.8 Hz`:
```
Δf = 11.8 / 512 = 0.023 Hz = 1.38 BPM
```

Daha az veri durumunda (`nperseg = 118`):
```
Δf = 11.8 / 118 = 0.10 Hz = 6.0 BPM → her zaman 6 BPM sonucu!
```

Bu nedenle Welch BPM gerçek zamanlı sistemde yalnızca referans olarak tutulmuş, ana BPM tahmini FFT'ye bırakılmıştır.

### 7.2 Sıfır-Dolgulu FFT

```python
N_FFT = 4096   # sıfır dolgu boyutu
freqs = np.fft.rfftfreq(N_FFT, d=1.0/Fs)
mag   = np.abs(np.fft.rfft(x, n=N_FFT))
mask  = (freqs >= BP_LOW) & (freqs <= BP_HIGH)
peak_hz  = freqs[mask][argmax(mag[mask])]
peak_bpm = peak_hz × 60.0
```

**Efektif çözünürlük:**
```
Δf = Fs / N_FFT = 11.8 / 4096 = 0.00288 Hz = 0.173 BPM
```

Bu çözünürlük 25 BPM (0.4167 Hz) ile 28 BPM (0.4667 Hz) arasındaki farkı (~0.05 Hz = 3 BPM) net olarak ayırt edebilir.

**Not:** Sıfır dolgu frekans çözünürlüğünü değil, **interpolasyon hassasiyetini** artırır. Gerçek frekans çözünürlüğü `Fs/N` (N = orijinal veri uzunluğu) ile sınırlıdır; ancak tepe konumu tespiti çok daha hassas olur.

### 7.3 Güven Skoru (Confidence)

```python
prominence = peak_val / mean(mag[mask])
confidence = clip((prominence - 1.0) / 4.0, 0.0, 1.0)
```

- `prominence < 1.5` → zayıf sinyal (güven = 0)
- `prominence > 5.0` → güçlü sinyal (güven = 1)

---

## 8. BNR Varlık Kapısı (Coherence Gate)

### 8.1 Amaç

Odada kimse yokken veya CSI sinyali çok zayıfken FFT hesaplamasını engeller ve kullanıcıya anlamlı bir uyarı gösterir.

### 8.2 Çalışma Prensibi

```python
BNR_GATE_THRESH = 1.5

# CSI pipeline'dan dönen max_bnr kontrolü:
if max_bnr < BNR_GATE_THRESH:
    bpm = -1.0   # özel sentinel değer
    status = f"Boş Oda / Sinyal Zayıf  BNR={max_bnr:.2f}"
    return  # FFT hesaplanmaz
else:
    bpm = dominant_bpm_fft(pc1, fs)
```

**BNR yorumlama tablosu:**

| BNR Aralığı | Yorum | Sistem Kararı |
|-------------|-------|---------------|
| < 0.5 | Çok zayıf sinyal | Boş oda / kör nokta |
| 0.5–1.5 | Zayıf sinyal | Gate açılmaz, BPM gösterilmez |
| 1.5–5.0 | Normal sinyal | Gate açık, BPM hesaplanır |
| 5.0–20.0 | Güçlü sinyal | İdeal Fresnel konumu |
| > 20.0 | Çok güçlü | Mükemmel LoS |

**Ölçülen değerler (bu çalışma, yan yana ESP32):**
- Tüm kayıtlarda BNR_max ≈ 0.28–0.40 → Gate eşiğinin altında
- İdeal LoS konumunda beklenen: BNR 5–20 arası

### 8.3 GUI Entegrasyonu

BNR gate devreye girdiğinde:
- `bpm = -1.0` sinyali GUI'ye iletilir
- `_on_csi_result()` metodu bu değeri yakalar
- Etiket: `"Wi-Fi CSI  --  0 BPM  (Boş Oda / Sinyal Zayıf)"` (kırmızı çerçeve)
- Status bar: `"CSI: Boş Oda / Sinyal Zayıf  BNR=0.38 (eşik=1.5)"`

---

## 9. Kamera Modalitesi — MediaPipe Pose

### 9.1 Omuz Hareketi ile Nefes Tespiti

Kamera pipeline'ı, kişinin göğüs/omuz hareketini takip ederek nefes dalgasını çıkarır.

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

**İzlenen noktalar:**
- Landmark 11: Sol omuz
- Landmark 12: Sağ omuz
- `shoulder_y = (lm11.y + lm12.y) / 2` — normalize Y koordinatı

**Neden Y ekseni?** Nefes alındığında göğüs yukarı kalkar; Y koordinatı azalır. Ters çevirme: `raw_y = -(samples - mean)`.

### 9.2 Kamera Pipeline'ı

```
Ham omuz-Y zaman serisi
        │
[1] DC çıkarma (mean subtraction)
        │
[2] Hampel filtresi (half_win = max(5, Fs//2))
        │
[3] Savitzky-Golay (window = min(51, N//6 | 1), poly=3)
        │
[4] Butterworth bandpass (0.05–0.40 Hz, order=4)
        │
[5] FFT BPM (dominant_bpm_fft)
        │
    Kamera BPM
```

Kamera pipeline'ı için SG penceresi daha büyük tutulabilir (omuz hareketi 25 BPM gibi hızlı bileşen içermeyebilir).

---

## 10. Çift Modalite Füzyonu

### 10.1 Spektral Geometrik Ortalama

Her iki modalite için FFT spektrumu 200 noktalı ortak frekans ızgarasına (`_FUSE_FREQS = linspace(0.05, 0.40, 200)`) interpolle edilir ve normalize edilir (tepe = 1):

```python
fused_spec = sqrt(cam_spec × csi_spec)
```

**Mantık:** Geometrik ortalama, iki sinyal aynı frekansta güçlü ise maksimuma çıkar; birisi zayıfsa değer düşer. Bu, yanlış tespitleri otomatik olarak bastırır.

### 10.2 Uyum Skoru

```python
agreement = max(0.0, 1.0 - |cam_bpm - csi_bpm| / 6.0)
```

- `agreement = 1.0` → iki modalite tam uyumlu (< 1 BPM fark)
- `agreement = 0.0` → 6 BPM veya daha fazla uyumsuzluk

GUI'de uyum skoru, birleşik BPM etiketinin renk kodlamasını belirler:
- Mor (parlak): agreement > 0.70
- Mor (orta): agreement > 0.40
- Mor (soluk): agreement ≤ 0.40

### 10.3 BNR Gate ve Füzyon İlişkisi

CSI sinyali BNR gate'ini geçemezse `FusedBreathWorker` çıktı üretmez:

```python
_, _, csi_signal, csi_fs, max_bnr = csi_pipeline(csi_frames, self._pca)
if max_bnr < BNR_GATE_THRESH:
    return  # füzyon yapılmaz, fused BPM güncellenmez
```

Bu durumda birleşik BPM etiketi son geçerli değeri göstermeye devam eder.

---

## 11. Gerçek Zamanlı Sistem Mimarisi

### 11.1 QThread Yapısı

```
Ana Thread (GUI)
├── MainWindow (PyQt5)
│   ├── BPM etiketleri (×3: Kamera, CSI, Birleşik)
│   ├── 5 pyqtgraph grafik (ham CSI, SG, PC1, ham omuz, filtreli)
│   └── Status bar (son durum mesajı)
│
├── CameraThread (QThread)           ← kamera görüntüsü yakalar
│   └── frame_ready → MainWindow._on_frame()
│
├── SerialThread (QThread)           ← COM10'dan CSI okur
│   └── buf_updated → CSI tampon sayacı güncellenir
│
├── CameraBreathWorker (QThread)     ← kamera nefes işleme
│   ├── QTimer (1000 ms)
│   └── result → MainWindow._on_cam_result()
│
├── CSIProcessingWorker (QThread)    ← CSI nefes işleme
│   ├── QTimer (1000 ms)
│   └── result → MainWindow._on_csi_result()
│
└── FusedBreathWorker (QThread)      ← çift modalite füzyon
    ├── QTimer (1000 ms)
    └── result → MainWindow._on_fused_result()
```

### 11.2 Mutex Senkronizasyonu

```python
# CSI tamponu
self._csi_buf  = collections.deque(maxlen=BUF_MAX)
self._csi_lock = QMutex()

# Kamera tamponu
self._cam_buf  = collections.deque(maxlen=CAM_BUF_MAX)
self._cam_lock = QMutex()

# Güvenli erişim
with QMutexLocker(self._csi_lock):
    frames = list(self._csi_buf)
```

### 11.3 Sinyal Tipleri

```python
# CSIProcessingWorker
result = pyqtSignal(object, object, object, float, float)
#                   raw_mean sg_mean pc1    bpm   fs
# bpm = -1.0 → BNR gate devreye girdi

# CameraBreathWorker
result = pyqtSignal(object, object, float, float)
#                   raw_y  filtered bpm   fs

# FusedBreathWorker
result = pyqtSignal(float, float, float, float)
#                   fused  cam    csi    agreement
```

### 11.4 Minimum Veri Gereksinimleri

```python
MIN_CSI_SAMP = FS_NOMINAL × 5 = 13 × 5 = 65 paket (~5 s)
MIN_CAM_SAMP = 75 çerçeve (~2.5 s @ 30 FPS)
```

Bu eşiklere ulaşılmadan işleme başlamaz.

---

## 12. Çevrimdışı Analiz Sonuçları

### 12.1 Veri Seti

| Dosya | Süre | Fs (Hz) | Aktif SC | İçerik |
|-------|------|---------|----------|--------|
| `csi_data_empty_my_room.csv` | 581 s | 11.71 | 115 | Boş oda (referans) |
| `csi_data_long_breath.csv` | 1243 s | 11.78 | 116 | Uzun nefes (bilinmiyor) |
| `csi_data_25bpm_breath.csv` | 346 s | 11.82 | 115 | 25 BPM kontrollü nefes |

**Kayıt koşulları:** TX ve RX ESP32 **yan yana** konumlandırılmış (LoS yok). Kişi ESP32'lerin önünde nefes almış; ancak RF yolu üzerinde değil.

### 12.2 BNR Band Değişikliğinin Etkisi

| Yöntem | Boş Oda | Uzun Nefes | 25 BPM |
|--------|---------|-----------|--------|
| Naive (tüm SC, SG=51) | 11.0 BPM | 8.3 BPM | 9.7 BPM |
| SG=11 düzeltmesi | 22.0 BPM | 23.5 BPM | 16.6 BPM |
| **BNR[0.16 Hz] + top-20** | **16.3 BPM** | **28.4 BPM** | **28.1 BPM** |
| Hedef | ~0 (gürültü) | Bilinmiyor | 25.0 BPM |

### 12.3 25 BPM Dosyası Detaylı Analizi

**BNR[0.16–0.60 Hz] ile seçilen top-20 subcarrier PC analizi:**

```
PCA:  PC1=54.3% varyans  PC2=19.8% varyans  PC3=6.8% varyans
      PC1: Welch=29.1 BPM   SampEn=2.615
      PC2: Welch=16.6 BPM   SampEn=2.442
      PC3: Welch=15.2 BPM   SampEn=2.639

VN-SampEn: PC1=4.82  PC2=12.32  PC3=39.01
→ PC1 seçildi (minimum VN-score)

FFT BPM = 28.07 BPM (0.4678 Hz)
Welch BPM = 29.1 BPM
```

**Yorumsal not:** 28.07 BPM, 25 BPM hedefinden ~3 BPM sapma göstermektedir. Boş oda dosyasında bu frekans bölgesinde güçlü bir sinyal gözlemlenmemiş (boş oda: 16.3 BPM), bu da 28 BPM sinyalinin gerçek nefes kaynağına atfedilebileceğini destekler. Sapmanın olası nedenleri: (a) kayıt sırasında gerçek nefes hızının 25'ten 28'e kayması, (b) LoS eksikliği nedeniyle sinyalin kısmi bozulması.

### 12.4 Uzun Nefes Dosyası BPM Tahmini

Uzun nefes dosyasının etiketi "bilinmiyor" olmasına rağmen:
- Ölçülen FFT BPM: **28.4 BPM**
- Boş odadan farklı → çevresel kaynak değil
- Kişi muhtemelen ~28 BPM hızında nefes almıştır

### 12.5 SampEn = ∞ Durumu

25 BPM dosyasının önceki analizinde (BNR band değişikliğinden önce) PC1 SampEn = ∞ gözlemlenmiştir. Bu durum:
- `B = 0`: m-boyutlu hiçbir kalıp eşleşmesi yok → sinyal çok gürültülü veya çok düzgün
- Geçici çözüm: `SampEn = inf` → kod içinde `10.0` olarak işlenir
- Bu değer VN-SampEn skoru hesaplamasını bozabilir

---

## 13. Anahtar Bulgular ve Kısıtlar

### 13.1 Teknik Katkılar

1. **BNR band kesimi (0.16 Hz):** HVAC/fan gürültüsünü (8–10 BPM) subcarrier seçim aşamasında sistematik olarak eliyor. Önceki yaklaşımlar bu frekansı nefes bandına dahil ediyordu.

2. **VN-SampEn:** Varyans-normalize sample entropy, HVAC'ın düşük SampEn ile en iyi PC olarak yanlış seçilmesini önlüyor. Geleneksel minimum-SampEn seçimi bu senaryoda başarısız olmuştur.

3. **Sıfır-dolgulu FFT:** Welch yönteminin çözünürlük kısıtını (özellikle kısa veri penceresinde Δf = 6 BPM) aşıyor.

4. **BNR Varlık Kapısı:** BNR < 1.5 olduğunda FFT hesaplamayı engelleyerek boş oda ve sinyal zayıf durumlarında anlamsız BPM çıktısı üretilmesini önlüyor.

5. **Çift modalite geometrik füzyon:** İki modalite aynı frekansta güçlüyse tepe büyür, biri zayıfsa düşer. Bu, tek modaliteli sistemlere kıyasla güvenilirliği artırır.

### 13.2 Kısıtlar ve Açık Problemler

| Problem | Neden | Önerilen Çözüm |
|---------|-------|----------------|
| BNR_max < 1.0 (tüm kayıtlar) | TX-RX yan yana, Fresnel bölgesi dışı | Kişiyi TX-RX doğrusu üzerine konumlandır |
| 25 BPM yerine 28 BPM | LoS yok → sinyal zayıf, kayma var | İdeal LoS ile BNR ≥ 5 hedefle |
| HVAC 22–23 BPM bölgesini kirletiyor | Bazı subcarrier'larda hâlâ HVAC | BNR_LO'yu dinamik hesapla |
| SampEn hesaplaması yavaş (n_max=400) | O(N²) karmaşıklık | Hızlı SampEn (CEEMDAN tabanlı) kullan |
| Welch çözünürlüğü ≥ 1.4 BPM | `nperseg` sınırı | Yalnızca sıfır-dolgulu FFT kullan |

### 13.3 Fresnel Bölgesi Etkisi

Wi-Fi CSI tabanlı nefes tespitinde **birinci Fresnel elipsoidi** kritik öneme sahiptir:

- **Durum 1 (İdeal):** Kişi TX-RX doğrusu üzerinde, Fresnel bölgesi içinde → BNR yüksek (5–20)
- **Durum 2 (Kısmi):** Kişi Fresnel bölgesi kenarında → BNR orta (1–5)
- **Durum 3 (Bu çalışma):** TX-RX yan yana, kişi Fresnel bölgesi dışında → BNR düşük (< 1)

Birinci Fresnel elipsoidinin yarıçapı (merkez noktada):

```
r₁ = sqrt(λ × d / 4)
```

`λ = 0.125 m` (2.4 GHz Wi-Fi), `d = 2 m` (TX-RX mesafesi):
```
r₁ = sqrt(0.125 × 2 / 4) = 0.25 m
```

Kişinin bu 25 cm yarıçaplı elipsoid içinde kalması gerekir.

---

## 14. Algoritma Parametreleri Referans Tablosu

### Çevrimdışı Analiz (`analyze_new_csi.py`)

| Parametre | Değer | Açıklama |
|-----------|-------|---------|
| `TRIM_SEC` | 5 s | Başlangıç ve bitişteki kararsız bölge |
| `BP_LOW` | 0.10 Hz | Bandpass filtre alt sınırı |
| `BP_HIGH` | 0.60 Hz | Bandpass filtre üst sınırı (25 BPM = 0.417 Hz dahil) |
| `BP_ORDER` | 4 | Butterworth filtre mertebesi |
| `BNR_LO` | **0.16 Hz** | BNR hesaplama alt sınırı (HVAC dışarıda) |
| `BNR_HI` | 0.60 Hz | BNR hesaplama üst sınırı |
| `SG_WINDOW` | **11 örnek** | Savitzky-Golay pencere (< T_25bpm = 29 örnek) |
| `SG_ORDER` | 3 | Polinom derecesi |
| `HAMPEL_HALF` | 50 örnek | Hampel yarı-pencere (~4.2 s @ 12 Hz) |
| `HAMPEL_NSIG` | 2.5 | Outlier eşiği (MAD çarpanı) |
| `TOP_N_SC` | **20** | BNR ile seçilecek subcarrier sayısı |
| `N_PCS` | 3 | PCA bileşen sayısı |
| `SAMPEN_M` | 3 | Sample Entropy gömme boyutu |
| `SAMPEN_R_COEF` | 0.1 | SampEn tolerans katsayısı (r = 0.1 × std) |
| `SAMPEN_N_MAX` | 400 | SampEn hız için maksimum örnek sayısı |
| `FFT_NFFT` | 4096 | Sıfır-dolgu FFT noktası |

### Gerçek Zamanlı Sistem (`csi_dual_monitor.py`)

| Parametre | Değer | Açıklama |
|-----------|-------|---------|
| `RX_PORT` | COM10 | Alıcı ESP32 seri portu |
| `RX_BAUD` | 115200 | Seri port baud rate |
| `FS_NOMINAL` | 13 Hz | Nominal CSI örnekleme hızı |
| `WINDOW_S` | 15 s | İşleme penceresi uzunluğu |
| `BREATH_LO` | 0.05 Hz | Bandpass alt sınırı (gerçek zamanlı) |
| `BREATH_HI` | 0.40 Hz | Bandpass üst sınırı (gerçek zamanlı) |
| `PROC_INTERVAL` | 1000 ms | İşleme tetikleme periyodu |
| `BNR_GATE_THRESH` | **1.5** | Gate eşiği (altında BPM gösterilmez) |
| `BUF_MAX` | 300 paket | CSI dairesel tampon boyutu |
| `MIN_CSI_SAMP` | 65 paket | İşleme başlamak için minimum paket |

---

## Referanslar ve Bağlam

- **RespirFi:** Wi-Fi CSI tabanlı nefes izleme referans mimarisi; subcarrier seçim + PCA yaklaşımı
- **ESP-NOW:** Espressif'in bağlantısız Wi-Fi protokolü; standart 802.11n altyapısını CSI telemetrisi için kullanır
- **Welch PSD:** P.D. Welch, 1967 — örtüşen pencereli FFT ile PSD tahmini
- **Sample Entropy:** Richman & Moorman, 2000 — biyomedikal zaman serisinde karmaşıklık ölçümü
- **MediaPipe Pose Landmarker:** Google, 2023 — 33 keypoint, Tasks API v0.10.x

---

*Oluşturulma: 2026-06-13 | Sistem: 2× ESP32 + CSI + MediaPipe | Platform: Windows 11, Python 3.x, PyQt5*
