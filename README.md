# 🏗️ Binalarda Enerji Verimliliği Analizi: Termal Görüntü ve OSOS Veri Füzyonu

> **TÜBİTAK 2209-A** kapsamında geliştirilmiş, Dr. Seda Balta Kaç danışmanlığında,  
> Kocaeli Üniversitesi Bilgi Sistemleri Mühendisliği bitirme projesi.

Binalardaki ısı kayıplarının (termal köprülerin) drone görüntülerinden otonom olarak tespiti ve bu kayıpların **aylık finansal karşılığının (TL/Ay)** hesaplanmasını sağlayan bir Karar Destek Sistemi.

---

## 🔍 Projenin Özgün Değeri

Literatürdeki mevcut çalışmalar genellikle yalnızca görüntü işleme ile ısı kaybı tespitine odaklanmaktadır. Bu çalışma iki katmanlı bir **Veri Füzyonu (Data Fusion)** yaklaşımı sunar:

| Katman | Açıklama |
|--------|----------|
| 🛰️ **Mekansal Analiz** | Custom Attention U-Net mimarisi ile drone görüntülerinden termal köprülerin hassas tespiti |
| 💰 **Finansal Analiz** | Tespit edilen anomali alanının akıllı sayaç (OSOS) verileriyle birleştirilerek aylık finansal kaybın tahmini |

---

## 🧠 Teknik Mimari

- **Model:** Custom Attention U-Net (Attention Gates + Dropout)
- **Loss Function:** Focal Tversky Loss (sınıf dengesizliğini minimize etmek için)
- **Deployment:** Streamlit etkileşimli kullanıcı arayüzü

---

## 📊 Eğitim Süreci ve Sonuçlar

![Training Curves](images/training-curves.png)

*Eğitim sürecinde Loss optimizasyonu, IoU skoru ve Global Piksel Doğruluğu metrikleri. Model, Dr. Seda Balta Kaç'ın hedeflediği %90 doğruluk eşiğini geçmiştir.*

---

## 🖼️ Örnek Tahmin Sonucu

![Prediction Result](images/prediction-result.png)

*Soldan sağa: Termal girdi görüntüsü — Uzman tarafından işaretlenmiş Ground Truth — AI tahmini (IoU: 0.480) — Termal overlay görselleştirme.*

---

## 📈 Benchmark Karşılaştırması

| Model Mimarisi | Accuracy (%) | Mean IoU | Dice Score |
|----------------|:------------:|:--------:|:----------:|
| FCN Baseline | 84.20 | 0.4820 | 0.5210 |
| Standard U-Net | 89.50 | 0.7140 | 0.7520 |
| **Custom Attention U-Net** | **94.80** | **0.8350** | **0.8640** |

---

## 🚀 Kurulum ve Kullanım

**1. Depoyu klonlayın:**
```bash
git clone https://github.com/sudenazz1/Termal-Veri-Fuzyonu.git
cd Termal-Veri-Fuzyonu
```

**2. Gerekli kütüphaneleri yükleyin:**
```bash
pip install -r requirements.txt
```

**3. Model dosyasını indirin ve `models/` dizinine yerleştirin:**

📥 [Model dosyasını indirmek için tıklayın](https://drive.google.com/drive/folders/1dlizpYsKwWcxpiN6Kv0UhqIdc1UZIjrK)

**4. Arayüzü başlatın:**
```bash
streamlit run app.py
```

---

## 📁 Proje Yapısı

Termal-Veri-Fuzyonu/

├── models/

│   └── unet_final.keras

├── images/

│   ├── training-curves.png

│   └── prediction-result.png

├── app.py

├── requirements.txt

└── README.md

---

## 👥 Proje Ekibi

- **Danışman:** Dr. Seda Balta Kaç
- **Destekleyen Kurum:** TÜBİTAK 2209-A Programı
- **Üniversite:** Kocaeli Üniversitesi, Bilgi Sistemleri Mühendisliği
