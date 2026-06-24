Binalarda Enerji Verimliliği Analizi: Termal Görüntü ve OSOS Veri Füzyonu
Bu proje, Kocaeli Üniversitesi Bilgi Sistemleri Mühendisliği bitirme projesi kapsamında, binalardaki ısı kayıplarının (termal köprülerin) otonom olarak tespiti ve bu kayıpların finansal karşılığının hesaplanması amacıyla geliştirilmiş bir Karar Destek Sistemi'dir.

Projenin Özgün Değeri
Literatürdeki mevcut çalışmalar genellikle sadece görüntü işleme ile ısı kaybının tespitine odaklanmaktadır. Bu çalışma ise iki katmanlı bir Veri Füzyonu (Data Fusion) yaklaşımı sunar:

Mekansal Analiz: "Custom Attention U-Net" mimarisi ile drone görüntülerinden termal köprülerin hassas tespiti.

Finansal Analiz: Tespit edilen anomali alanının, akıllı sayaç (OSOS) verileriyle birleştirilerek aylık finansal kaybın (TL/Ay) tahmin edilmesi.

Teknik Mimari
Model: Custom Attention U-Net (Attention Gates ve Dropout ile özelleştirilmiş).

Loss Function: Focal Tversky Loss.

Performans: 91.83% Genel Doğruluk (Accuracy) ve 0.2914 IoU.

Deployment: Streamlit üzerinden etkileşimli kullanıcı arayüzü.

Eğitim Analizi ve Model Performansı
Aşağıdaki grafikler, modelin eğitim sürecindeki kayıp optimizasyonunu, IoU gelişimini ve doğruluk oranlarını göstermektedir:

Aşağıdaki görsel, modelin termal girdi üzerindeki segmentasyon başarısını ve gerçek verilerle (Ground Truth) kıyaslamasını içermektedir:

Kurulum ve Kullanım
Projeyi yerel ortamda çalıştırmak için:

Depoyu klonlayın:

Bash
git clone https://github.com/sudenazz1/Termal-Veri-Fuzyonu.git
cd Termal-Veri-Fuzyonu
Gerekli kütüphaneleri yükleyin:

Bash
pip install -r requirements.txt
Model dosyasını models/ dizinine yerleştirin:
Model dosyasını indirmek için tıklayın

Arayüzü çalıştırın:

Bash
streamlit run app.py
Metrikler (Benchmark)
Model Mimarisi	Accuracy (%)	Mean IoU	Dice Score
FCN Baseline	84.20	0.4820	0.5210
Standard U-Net	89.50	0.7140	0.7520
Custom Attention U-Net	91.83	0.2914	0.4484
Not: Bu proje TÜBİTAK 2209-A kapsamında, Dr. Seda Balta Kaç danışmanlığında geliştirilmiştir.
