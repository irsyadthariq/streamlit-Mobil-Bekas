import streamlit as st
import pandas as pd
import joblib

# Memuat model
model = joblib.load('best_model.pkl')

# Memuat encoder
label_encoder_model = joblib.load('label_encoder_model.pkl')
label_encoder_wilayah = joblib.load('label_encoder_wilayah.pkl')
label_encoder_merk = joblib.load('label_encoder_merk.pkl')

# Memuat nilai unik
unique_values = joblib.load('unique_values.pkl')
unique_models = unique_values['unique_models']
unique_wilayah = unique_values['unique_wilayah']
unique_merk = unique_values['unique_merk']

# Fit the LabelEncoder instance to the data
label_encoder_model.fit(unique_models)
label_encoder_wilayah.fit(unique_wilayah)
label_encoder_merk.fit(unique_merk)


# Fungsi untuk prediksi
def predict(data):
    prediction = model.predict(data)
    return prediction


# Judul aplikasi
st.title('Prediksi Harga Mobil')

# Input data
st.write('Masukkan data untuk prediksi:')
jarak_tempuh = st.number_input('Jarak Tempuh(Km)\nJika 3000Km, cukup tulis= 3', value=0.0)
kapasitas_mesin = st.number_input('Kapasitas Mesin(CC)\nJika 1500CC, cukup tulis=1.50', value=0.0)
tahun_produksi = st.number_input('Tahun Produksi', value=0.0)
transmisi = st.selectbox('Transmisi', options=['Automatic', 'Manual'])
bahan_bakar = st.selectbox('Bahan Bakar', options=['Bensin', 'Diesel'])
model_mobil = st.selectbox('Model Mobil', options=unique_models)
wilayah = st.selectbox('Wilayah', options=unique_wilayah)
merk_mobil = st.selectbox('Merk Mobil', options=unique_merk)    

# Konversi input transmisi dan bahan bakar
transmisi = 0 if transmisi == 'Automatic' else 1
bahan_bakar = 0 if bahan_bakar == 'Bensin' else 1

# Encode fitur kategorikal
model_mobil_encoded = label_encoder_model.transform([model_mobil])[0]
wilayah_encoded = label_encoder_wilayah.transform([wilayah])[0]
merk_mobil_encoded = label_encoder_merk.transform([merk_mobil])[0]

# Membuat DataFrame dari input
data = pd.DataFrame({
    'model_mobil': [model_mobil_encoded],
    'wilayah': [wilayah_encoded],
    'merk_mobil': [merk_mobil_encoded],
    'transmisi': [transmisi],
    'bahan_bakar': [bahan_bakar],
    'jarak_tempuh': [jarak_tempuh],
    'kapasitas_mesin': [kapasitas_mesin],
    'tahun_produksi': [tahun_produksi]
})

# Tombol untuk prediksi
if st.button('Prediksi'):
    result = predict(data)
    st.write(f'Harga Prediksi: Rp {result[0]:,.2f}')
