import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta

# Fungsi untuk mempersiapkan data dengan tren historis
def prepare_data_with_trends(data):
    df = data.copy()
    df['TANGGAL'] = pd.to_datetime(df['TANGGAL'], errors='coerce')
    df = df.dropna(subset=['TANGGAL'])  
    df['BULAN'] = df['TANGGAL'].dt.strftime('%B')
    df = df.drop(columns=['NO', 'NO FAKTUR', 'TANGGAL', 'KODE STORE', 'NAMA STORE', 'ALAMAT STORE', 
    'KOTA', 'TIPE', 'KODE SALES', 'SALESMAN', 'JUMLAH', 'TONASE', 'TOTAL'], errors='ignore')

    current_year = datetime.now().year
    df['BULAN'] = pd.to_datetime(df['BULAN'] + ' ' + str(current_year), format='%B %Y', errors='coerce')
    df = df.dropna(subset=['BULAN']) 
    df['product_code'] = df['KODE BARANG'].astype(str)

# Menghitung statistik bulanan
    monthly_stats = df.groupby(['KODE BARANG', 'NAMA BARANG', 'BULAN']).agg({
        'QTY': 'sum',
        'HARGA': 'mean',
        'SATUAN': 'first'
    }).reset_index()

 # Menambahkan fitur tambahan
    monthly_stats = monthly_stats.sort_values(['KODE BARANG', 'BULAN'])
    monthly_stats['rolling_avg_qty'] = monthly_stats.groupby('KODE BARANG')['QTY'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    monthly_stats['qty_growth'] = monthly_stats.groupby('KODE BARANG')['QTY'].pct_change()

  # Encoding fitur kategori
    le = LabelEncoder()
    monthly_stats['product_encoded'] = le.fit_transform(monthly_stats['KODE BARANG'])
    monthly_stats['unit_encoded'] = le.fit_transform(monthly_stats['SATUAN'])

    # Menyiapkan fitur dan target untuk pelatihan model
    features = pd.DataFrame({
        'product_encoded': monthly_stats['product_encoded'],
        'unit_encoded': monthly_stats['unit_encoded'],
        'price': monthly_stats['HARGA'],
        'rolling_avg_qty': monthly_stats['rolling_avg_qty'].fillna(0),
        'qty_growth': monthly_stats['qty_growth'].fillna(0),
        'month': monthly_stats['BULAN'].dt.month
    })

    target = monthly_stats.groupby('KODE BARANG')['QTY'].shift(-1)

    return features, target, monthly_stats, le

# Fungsi untuk melatih model XGBoost
def train_model_with_trends(X, y):
    mask = ~y.isna()
    X = X[mask]
    y = y[mask]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)

    return model

# Fungsi untuk memprediksi penjualan bulan berikutnya
def predict_next_month_sales(data, model, monthly_stats, label_encoder, n_top=5):
    last_month = monthly_stats['BULAN'].max()
    next_month = last_month + pd.DateOffset(months=1)

    latest_data = monthly_stats[monthly_stats['BULAN'] == last_month].copy()
    pred_features = pd.DataFrame({
        'product_encoded': latest_data['product_encoded'],
        'unit_encoded': latest_data['unit_encoded'],
        'price': latest_data['HARGA'],
        'rolling_avg_qty': latest_data['rolling_avg_qty'],
        'qty_growth': latest_data['qty_growth'],
        'month': next_month.month
    })

    predictions = model.predict(pred_features)

    results = pd.DataFrame({
        'KODE BARANG': latest_data['KODE BARANG'],
        'NAMA BARANG': latest_data['NAMA BARANG'],
        'Last_Month_Sales': latest_data['QTY'],
        'Predicted_Next_Month_Sales': np.round(predictions, 2),
        'Expected_Growth': ((predictions - latest_data['QTY']) / latest_data['QTY'] * 100).round(2)
    })

    top_products = results.sort_values('Predicted_Next_Month_Sales', ascending=False).head(n_top)

    return top_products, next_month

# Fungsi untuk menghitung performa penjualan
def calculate_sales_performance(monthly_stats):
    sales_performance = monthly_stats.groupby(['KODE BARANG', 'NAMA BARANG']).agg({
        'QTY': 'sum'
    }).reset_index()
    
    sales_performance = sales_performance.sort_values(by='QTY', ascending=False)
    sales_performance['Rank'] = range(1, len(sales_performance) + 1)
    
    return sales_performance

# Konfigurasi tampilan aplikasi Streamlit
st.set_page_config(page_title="Garam Sales Prediction", layout="wide")

# Tampilan utama aplikasi Streamlit
st.title("Garam Sales Prediction & Performance Ranking")
st.markdown("Upload your Excel file to analyze sales trends, predict next month's sales, and view sales performance ranking.")

uploaded_file = st.file_uploader("Upload Excel File", type="xlsx")

if uploaded_file is not None:
    try:
        data = pd.read_excel(uploaded_file, engine='openpyxl')
        data = data.iloc[1:].reset_index(drop=True)

        st.write("Preview of the uploaded data:")
        st.dataframe(data.head(), use_container_width=True)

        features, target, monthly_stats, label_encoder = prepare_data_with_trends(data)
        model = train_model_with_trends(features, target)

        top_products, next_month = predict_next_month_sales(data, model, monthly_stats, label_encoder)
        sales_performance = calculate_sales_performance(monthly_stats)

        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"### Top 5 Products Predicted for {next_month.strftime('%B %Y')}:")
            
            plt.figure(figsize=(10, 5))
            plt.barh(top_products['NAMA BARANG'], top_products['Predicted_Next_Month_Sales'], color='skyblue')
            plt.xlabel('Predicted Sales')
            plt.ylabel('Nama Barang')
            plt.title('Top 5 Predicted Sales for Next Month')
            plt.gca().invert_yaxis()
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            st.pyplot(plt)
            
            st.dataframe(top_products, use_container_width=True)
        
        with col2:
            st.write("### Sales Performance Ranking :")

            plt.figure(figsize=(10, 5))
            plt.barh(sales_performance['NAMA BARANG'].head(10), sales_performance['QTY'].head(10), color='green')
            plt.xlabel('Total Sales Quantity')
            plt.ylabel('Nama Barang')
            plt.title('Top 10 Sales Performance Ranking')
            plt.gca().invert_yaxis()
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            st.pyplot(plt)

            st.dataframe(sales_performance.head(10), use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
