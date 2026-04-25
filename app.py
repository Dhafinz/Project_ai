import streamlit as st
import joblib
import numpy as np

# Load model dan scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Konfigurasi Streamlit
st.set_page_config(page_title="Prediksi Harga Rumah", layout="centered")

# Konstanta konversi
M2_TO_FT2 = 10.7639
M2_TO_ACRE = 0.000247105
USD_TO_IDR = 17000

# Header
st.title("Prediksi Harga Rumah")
st.caption("Masukkan spesifikasi rumah untuk mendapatkan estimasi harga dalam USD dan IDR.")
st.divider()

# Form utama
with st.form("house_form"):
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Luas Properti")
        building_area_m2 = st.number_input(
            "Luas bangunan (m2)",
            min_value=10.0,
            max_value=1000.0,
            value=140.0,
            step=5.0,
            help="Contoh rumah sederhana: 36 m2, rumah menengah: 90-180 m2."
        )
        land_area_m2 = st.number_input(
            "Luas tanah (m2)",
            min_value=20.0,
            max_value=40000.0,
            value=200.0,
            step=10.0,
            help="Biasanya lebih besar atau sama dengan luas bangunan."
        )

        st.subheader("Spesifikasi Rumah")
        num_bedrooms = st.number_input("Jumlah kamar tidur", min_value=1, max_value=10, value=3)
        num_bathrooms = st.number_input("Jumlah kamar mandi", min_value=1, max_value=10, value=2)
        garage_size = st.number_input("Kapasitas garasi (mobil)", min_value=0, max_value=5, value=1)

    with col_right:
        st.subheader("Kondisi dan Lingkungan")
        current_year = 2026
        year_built = st.number_input(
            "Tahun dibangun",
            min_value=1900,
            max_value=current_year,
            value=2018
        )
        house_age = current_year - int(year_built)
        st.caption(f"Perkiraan usia bangunan: {house_age} tahun")
        neighborhood_quality = st.slider(
            "Kualitas lingkungan",
            min_value=1,
            max_value=10,
            value=6,
            help="1 = sangat kurang, 10 = sangat baik."
        )

        st.subheader("Preview Konversi Satuan")
        building_area_ft2 = building_area_m2 * M2_TO_FT2
        land_area_acre = land_area_m2 * M2_TO_ACRE
        st.write(f"Luas bangunan (feet): **{building_area_ft2:,.2f} ft2**")
        st.write(f"Luas tanah (acre): **{land_area_acre:,.4f} acre**")

    submit = st.form_submit_button("Prediksi Harga Rumah")

# Proses prediksi
if submit:
    input_features = np.array(
        [[
            building_area_ft2,
            num_bedrooms,
            num_bathrooms,
            year_built,
            land_area_acre,
            garage_size,
            neighborhood_quality,
        ]]
    )
    input_scaled = scaler.transform(input_features)
    predicted_price_usd = model.predict(input_scaled)[0]
    predicted_price_idr = predicted_price_usd * USD_TO_IDR

    st.divider()
    st.subheader("Hasil Prediksi")
    result_col1, result_col2 = st.columns(2)
    result_col1.metric("Estimasi Harga (USD)", f"${predicted_price_usd:,.2f}")
    result_col2.metric("Estimasi Harga (IDR)", f"Rp {predicted_price_idr:,.2f}")
    st.caption("Catatan: Kurs diasumsikan 1 USD = Rp 17.000.")

# Footer
st.divider()
st.write("(c) 2026 Model Regresi Prediksi Harga Rumah. Dibuat oleh Naufal dan Dhafin.")
