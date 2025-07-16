
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Prediksi Revenue Game Steam 2024")

st.title("Dashboard Prediksi Revenue Game Steam 2024")

st.write("UAS Big Data & Predictive Analytics")
st.markdown("""
#### Kelompok Big Data
- Rizal Anggoro - 23.11.5498
- Ahmad Nur Rofik - 23.11.5475
- Andre Aditia - 23.11.5756  
- Lingga Firmansyah - 23.11.5447
""")



try:
    df = pd.read_csv("Steam_2024_bestRevenue.csv")
    st.dataframe(df)

    fitur = ['copiesSold', 'price', 'avgPlaytime', 'reviewScore']
    X = df[fitur]
    y = df['revenue']

    model = LinearRegression()
    model.fit(X, y)

    st.subheader("Input Data Baru untuk Prediksi")

    copies_sold = st.number_input("Jumlah Unit Game Terjual", min_value=0)
    price = st.number_input("Harga Game (USD)", min_value=0.0, step=0.01)
    playtime = st.number_input("Rata-rata Waktu Bermain (jam)", min_value=0.0, step=0.1)
    score = st.slider("Skor Ulasan Pengguna", 0, 100, 75)

    if st.button("Prediksi Revenue"):
        new_data = pd.DataFrame({
            'copiesSold': [copies_sold],
            'price': [price],
            'avgPlaytime': [playtime],
            'reviewScore': [score]
        })
        pred = model.predict(new_data)[0]
        st.success(f"Prediksi Pendapatan Game: ${pred:,.2f}")

except FileNotFoundError:
    st.error("Dataset 'Steam_2024_bestRevenue.csv' tidak ditemukan. Pastikan file ada di folder yang sama.")
