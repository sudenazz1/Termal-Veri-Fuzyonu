# ==============================================================================
# BITIRME PROJESI: KARAR DESTEK SISTEMI PROTOTIPI (PHASE 5)
# Arayüz: Streamlit Web Application
# ==============================================================================

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
from PIL import Image
import matplotlib.pyplot as plt 
import os   

# 1. Configuration de la page
st.set_page_config(page_title="Enerji Verimliliği AI", layout="wide")
st.title("🏢 Binalarda Enerji Verimliliği: AI Tabanlı Termal Analiz")
st.markdown("**TÜBİTAK 2209-A** | Geliştiriciler: Sude Naz Lekesiz & Khadim Dieye | Danışman: Dr. Seda Balta Kaç")

# 2. Paramètres Physiques et Économiques (Sidebar)
st.sidebar.header("⚙️ Analiz Parametreleri")
temp_ext = st.sidebar.slider("Dış Ortam Sıcaklığı (°C)", -10.0, 15.0, 4.3)
temp_int = st.sidebar.slider("İç Ortam Sıcaklığı (°C)", 18.0, 25.0, 20.0)
u_coeff = st.sidebar.number_input("Isı İletim Katsayısı (U-değeri)", value=1.5)
price_kwh = st.sidebar.number_input("Enerji Birim Fiyatı (TL/kWh)", value=2.50)
building_area = st.sidebar.number_input("Analiz Edilen Yüzey Alanı (m²)", value=120.0)

# 3. Chargement du Modèle Custom U-Net
@st.cache_resource
def load_ai_model():
    # Définition des fonctions custom pour charger le modèle sauvegardé
    def dice_loss(y_true, y_pred, smooth=1e-6):
        y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        inter = tf.reduce_sum(y_true_f * y_pred_f)
        return 1 - (2 * inter + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

    def bce_dice_loss(y_true, y_pred):
        return keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

    def iou_metric(y_true, y_pred):
        y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)
        y_true_bin = tf.cast(y_true, tf.float32)
        inter = tf.reduce_sum(y_true_bin * y_pred_bin)
        union = tf.reduce_sum(y_true_bin) + tf.reduce_sum(y_pred_bin) - inter
        return (inter + 1e-6) / (union + 1e-6)

    def dice_metric(y_true, y_pred):
        y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)
        y_true_bin = tf.cast(y_true, tf.float32)
        inter = tf.reduce_sum(y_true_bin * y_pred_bin)
        return (2 * inter + 1e-6) / (tf.reduce_sum(y_true_bin) + tf.reduce_sum(y_pred_bin) + 1e-6)

    # Remplace le chemin par celui où se trouve ton unet_final.keras sur ton PC
    model_path = os.path.join(os.path.dirname(__file__), "unet_final.keras")
    try:
        model = keras.models.load_model(model_path, custom_objects={
            'bce_dice_loss': bce_dice_loss, 'iou_metric': iou_metric, 'dice_metric': dice_metric
        })
        return model
    except:
        return None

model = load_ai_model()

# 4. Interface d'Upload et de Prédiction
# Ajout des formats jpg, jpeg et png
uploaded_file = st.file_uploader("📸 Lütfen analiz edilecek termal görüntüyü yükleyin", type=['npy', 'jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Détection du type de fichier
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    if file_extension == 'npy':
        # Chargement d'un fichier NPY (Dataset TBBR)
        img_data = np.load(uploaded_file).astype(np.float32)
        thermal_channel = img_data[:, :, 3] # Extraction du canal thermique
    else:
        # Chargement d'une image classique (JPG/PNG)
        image = Image.open(uploaded_file).convert('L') # Convertir en niveaux de gris
        thermal_channel = np.array(image).astype(np.float32)
    
    # Redimensionnement obligatoire pour le modèle U-Net (256x256)
    thermal_resized = cv2.resize(thermal_channel, (256, 256))
    
    # Normalisation entre 0 et 1
    t_min, t_max = thermal_resized.min(), thermal_resized.max()
    if t_max > t_min:
        thermal_norm = (thermal_resized - t_min) / (t_max - t_min)
    else:
        thermal_norm = thermal_resized
        
    img_input = thermal_norm[np.newaxis, ..., np.newaxis]
    
    st.write("🔄 Yapay Zeka Analizi Yapılıyor...")
    
    if model is not None:
        # Prédiction IA
        pred = model.predict(img_input)[0, :, :, 0]
    else:
        # Simulation si le modèle n'est pas encore copié sur le PC
        pred = np.where(thermal_norm > 0.7, 1.0, 0.0) 
        st.warning("Model dosyası (unet_final.keras) bulunamadı. Simülasyon modu aktif.")

    pred_bin = (pred > 0.5).astype(np.float32)
    
    # 5. Calculs Thermodynamiques et Financiers
    leak_ratio = np.mean(pred_bin)
    leak_area_m2 = leak_ratio * building_area
    delta_t = temp_int - temp_ext
    
    # Q = U * A * DeltaT (Conversion en kWh pour un mois)
    heat_loss_kwh = (leak_area_m2 * u_coeff * delta_t * 24 * 30) / 1000
    financial_loss_tl = heat_loss_kwh * price_kwh
    
    # 6. Affichage des Résultats (Metrics)
    st.markdown("---")
    st.subheader("📊 Analiz Sonuçları ve Finansal Rapor")
    
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Tespit Edilen Kaçak Oranı", value=f"%{leak_ratio*100:.2f}")
    col2.metric(label="Aylık Enerji İsrafı", value=f"{heat_loss_kwh:.1f} kWh")
    col3.metric(label="Aylık Finansal Kayıp", value=f"{financial_loss_tl:.2f} TL", delta="-Zarar", delta_color="inverse")
    
    # 7. Recommandation Dynamique (Decision Support)
    if leak_ratio > 0.10:
        st.error("🚨 **Sistem Önerisi:** Kritik seviyede ısı köprüsü tespit edildi. Acil çatı izolasyonu ve yalıtım yenilemesi önerilmektedir. Bu yatırım kendini kısa sürede amorti edecektir.")
    elif leak_ratio > 0.02:
        st.warning("⚠️ **Sistem Önerisi:** Orta seviyede ısı kaybı. Pencere doğramalarının ve çatı birleşim noktalarının termal sızdırmazlık testinden geçirilmesi tavsiye edilir.")
    else:
        st.success("✅ **Sistem Önerisi:** Bina izolasyonu iyi durumda. Majör bir ısı kaybı tespit edilmedi.")
    
    # 8. Affichage Visuel
    st.markdown("---")
    st.subheader("👁️ Görsel Teşhis (AI Maskesi)")
    
    vcol1, vcol2, vcol3 = st.columns(3)
    
    # Thermique Original
    fig1, ax1 = plt.subplots()
    ax1.imshow(thermal_norm, cmap='inferno')
    ax1.axis('off')
    vcol1.pyplot(fig1)
    vcol1.caption("Orijinal Termal Görüntü")
    
    # Masque IA
    fig2, ax2 = plt.subplots()
    ax2.imshow(pred_bin, cmap='gray')
    ax2.axis('off')
    vcol2.pyplot(fig2)
    vcol2.caption("U-Net Tespit Maskesi")
    
    # Overlay
    fig3, ax3 = plt.subplots()
    ax3.imshow(thermal_norm, cmap='inferno')
    ax3.imshow(pred_bin, cmap='Greens', alpha=0.5)
    ax3.axis('off')
    vcol3.pyplot(fig3)
    vcol3.caption("Üst Üste Bidirilmiş Isı Haritası")