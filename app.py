import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import time

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Smart Trash Sorter", page_icon="♻️", layout="centered")

# --- LOAD MODEL AI (HASIL TRAINING RINOV) ---
# Kita gunakan cache biar model gak loading terus setiap klik tombol
@st.cache_resource
def load_model_ai():
    # Pastikan file model_sampah_kelompok1.h5 ada di folder yang sama dengan app.py
    return tf.keras.models.load_model('model_sampah_kelompok1.h5')

model = load_model_ai()
# Urutan label harus sesuai dengan urutan folder saat training (biasanya alfabetis)
labels = ['Kertas', 'Organik', 'Plastik']

# --- MENU SAMPING (SIDEBAR) ---
st.sidebar.title("👨‍💻 Tim Developer")
st.sidebar.write("1. **Rinov** (AI Engineer & PM)")
st.sidebar.write("2. **Fitrah** (App Developer)")
st.sidebar.write("3. **Owal** (Designer)")
st.sidebar.write("4. **Riazul** (Doc & Creative)")
st.sidebar.markdown("---")
st.sidebar.info("Proyek Akhir AI \n\nTeknik Informatika - Universitas Jabal Ghafur")

# --- HALAMAN UTAMA ---
st.title("♻️ Sistem Pintar Pemilah Sampah")
st.write("Unggah foto sampah (Organik, Plastik, atau Kertas), dan AI kami akan memberi tahu ke mana kamu harus membuangnya!")

# Kotak Upload Gambar
uploaded_file = st.file_uploader("Pilih foto sampah...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Tampilkan gambar yang di-upload
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang akan dianalisis", use_column_width=True)
    
    # 2. Tombol untuk mulai deteksi
    if st.button("🔍 Deteksi Jenis Sampah"):
        with st.spinner("AI sedang menganalisis tekstur dan bentuk sampah..."):
            # --- PROSES PREDIKSI MODEL RINOV ---
            # 1. Resize gambar sesuai input model (MobileNetV2: 224x224)
            img_resized = image.resize((224, 224))
            # 2. Ubah ke array dan normalisasi (0-1)
            img_array = np.array(img_resized) / 255.0
            # 3. Tambahkan dimensi batch (1, 224, 224, 3)
            img_array = np.expand_dims(img_array, axis=0)
            
            # 4. Prediksi menggunakan model
            predictions = model.predict(img_array)
            result_index = np.argmax(predictions)
            result_label = labels[result_index]
            confidence = np.max(predictions) * 100
            
            time.sleep(1) # Efek dramatis sedikit

            # --- TAMPILKAN HASIL REAL ---
            st.success(f"✅ HASIL DETEKSI: **{result_label.upper()}**")
            st.write(f"Tingkat Keyakinan AI: **{confidence:.2f}%**")
            
            # Berikan instruksi berdasarkan hasil
            if result_label == 'Organik':
                st.info("💡 **Tindakan:** Buang ke tempat sampah berwarna **HIJAU**. Sampah ini bisa dijadikan kompos!")
            elif result_label == 'Plastik':
                st.warning("💡 **Tindakan:** Buang ke tempat sampah berwarna **KUNING**. Sampah ini bisa didaur ulang menjadi barang baru!")
            elif result_label == 'Kertas':
                st.info("💡 **Tindakan:** Buang ke tempat sampah berwarna **BIRU**. Pastikan kertas tidak dalam kondisi basah atau berminyak.")