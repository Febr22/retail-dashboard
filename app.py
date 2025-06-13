import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


st.set_page_config(layout="wide", page_title="Dashboard Analisis Transaksi Ritel")
# Fungsi untuk memuat data dan model
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Retail_Transactions_Dataset.csv')
        # Convert 'Date' to datetime if it's not already
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except FileNotFoundError:
        st.error("Retail_Transactions_Dataset.csv tidak ditemukan. Pastikan file ada di direktori yang sama.")
        return pd.DataFrame()

@st.cache_data
def load_cluster_data():
    try:
        df_cluster = pd.read_csv('df_2023_limited_with_clusters.csv')
        return df_cluster
    except FileNotFoundError:
        st.error("df_2023_limited_with_clusters.csv tidak ditemukan. Pastikan file ada di direktori yang sama.")
        return pd.DataFrame()

@st.cache_resource # Use st.cache_resource for models/scalers
def load_prediction_model():
    try:
        model = joblib.load('model_pembayaran.pkl')
        scaler = joblib.load('scaler_model.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("File model_pembayaran.pkl atau scaler_model.pkl tidak ditemukan.")
        return None, None

@st.cache_resource
def load_segmentation_model():
    try:
        kmeans_model = joblib.load('model_segmentasi.pkl')
        scaler_kmeans = joblib.load('scaler_kmeans.pkl')
        return kmeans_model, scaler_kmeans
    except FileNotFoundError:
        st.error("File model_segmentasi.pkl atau scaler_kmeans.pkl tidak ditemukan.")
        return None, None

# Load data and models
df_main = load_data()
df_cluster = load_cluster_data()
model_pembayaran, scaler_model = load_prediction_model()
kmeans_model, scaler_kmeans = load_segmentation_model()

# Set Streamlit page config


st.title("Dashboard Analisis Transaksi Ritel")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Data Overview", "Prediksi Metode Pembayaran", "Segmentasi Pelanggan"])

# --- Tab 1: Data Overview ---
with tab1:
    st.header("Data Overview")

    if not df_main.empty:
        st.subheader("Lima Baris Pertama Dataset")
        st.dataframe(df_main.head())

        st.subheader("Ringkasan Statistik Deskriptif")
        st.dataframe(df_main.describe())

        st.subheader("Distribusi Persebaran Data Numerik")
        numerical_cols = ['Total_Items', 'Total_Cost']
        for col in numerical_cols:
            st.write(f"Distribusi {col}")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(df_main[col], kde=True, ax=ax)
            st.pyplot(fig)
            plt.close(fig)

        st.subheader("Distribusi Data Kategorikal")
        # Kolom kategorikal asli sebelum OHE untuk visualisasi di Data Overview
        categorical_cols = ['City', 'Season', 'Store_Type', 'Payment_Method', 'Customer_Category']
        for col in categorical_cols:
            if col in df_main.columns:
                st.write(f"Distribusi {col}")
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.countplot(data=df_main, y=col, order=df_main[col].value_counts().index, ax=ax)
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.warning(f"Kolom '{col}' tidak ditemukan di dataset utama.")
    else:
        st.warning("Tidak dapat menampilkan Data Overview karena dataset utama tidak ditemukan.")

# --- Tab 2: Prediksi Metode Pembayaran ---
with tab2:
    st.header("Prediksi Metode Pembayaran (Kartu Kredit)")
    st.write("Masukkan nilai fitur-fitur untuk memprediksi apakah pembayaran akan menggunakan Kartu Kredit.")

    if model_pembayaran and scaler_model:
        # Define input features based on the provided code's X.columns.tolist()
        # Ensure these match the exact column names after preprocessing (OHE, feature engineering)
        
        # Numerical features
        total_items = st.number_input("Total Items", min_value=0, value=5)
        total_cost = st.number_input("Total Cost", min_value=0.0, value=50.0, format="%.2f")
        discount_applied = st.checkbox("Discount Applied", value=False)
        day_of_week = st.slider("Day of Week (0=Senin, 6=Minggu)", 0, 6, 0)
        month = st.slider("Month (1=Jan, 12=Dec)", 1, 12, 1)
        is_weekend = st.checkbox("Is Weekend", value=False)
        average_item_cost = st.number_input("Average Item Cost", min_value=0.0, value=10.0, format="%.2f")

        # Categorical features for OHE - using the exact column names provided by the user
        city_options = ['Atlanta', 'Boston', 'Chicago', 'Dallas', 'Houston', 'Los Angeles', 'Miami', 'New York', 'San Francisco', 'Seattle']
        selected_city = st.selectbox("City", city_options)

        season_options = ['Fall', 'Spring', 'Summer', 'Winter']
        selected_season = st.selectbox("Season", season_options)

        store_type_options = ['Convenience Store', 'Department Store', 'Pharmacy', 'Specialty Store', 'Supermarket', 'Warehouse Club']
        selected_store_type = st.selectbox("Store Type", store_type_options)

        customer_category_options = ['Homemaker', 'Middle-Aged', 'Professional', 'Retiree', 'Senior Citizen', 'Student', 'Teenager', 'Young Adult']
        selected_customer_category = st.selectbox("Customer Category", customer_category_options)

        if st.button("Prediksi Metode Pembayaran"):
            # Create a DataFrame for the input with all expected OHE columns
            # Initialize all boolean columns to False
            input_data = pd.DataFrame(np.zeros((1, 35)), columns=[
                'Total_Items', 'Total_Cost', 'Discount_Applied', 'day_of_week', 'month',
                'is_weekend', 'average_item_cost', 'Season_Fall', 'Season_Spring',
                'Season_Summer', 'Season_Winter', 'Customer_Category_Homemaker',
                'Customer_Category_Middle-Aged', 'Customer_Category_Professional',
                'Customer_Category_Retiree', 'Customer_Category_Senior Citizen',
                'Customer_Category_Student', 'Customer_Category_Teenager',
                'Customer_Category_Young Adult', 'City_Atlanta', 'City_Boston',
                'City_Chicago', 'City_Dallas', 'City_Houston', 'City_Los Angeles',
                'City_Miami', 'City_New York', 'City_San Francisco', 'City_Seattle',
                'Store_Type_Convenience Store', 'Store_Type_Department Store',
                'Store_Type_Pharmacy', 'Store_Type_Specialty Store',
                'Store_Type_Supermarket', 'Store_Type_Warehouse Club'
            ], dtype=float) # Use float for numerical consistency before scaling

            # Fill numerical features
            input_data['Total_Items'] = total_items
            input_data['Total_Cost'] = total_cost
            input_data['Discount_Applied'] = bool(discount_applied) # Ensure boolean type
            input_data['day_of_week'] = day_of_week
            input_data['month'] = month
            input_data['is_weekend'] = bool(is_weekend) # Ensure boolean type
            input_data['average_item_cost'] = average_item_cost

            # Fill OHE features by setting the selected one to 1.0 (True)
            input_data[f'Season_{selected_season}'] = 1.0
            input_data[f'Customer_Category_{selected_customer_category}'] = 1.0
            input_data[f'City_{selected_city}'] = 1.0
            input_data[f'Store_Type_{selected_store_type}'] = 1.0

            # Scale the input data
            input_data_scaled = scaler_model.transform(input_data)

            # Make prediction
            prediction_proba = model_pembayaran.predict_proba(input_data_scaled)[:, 1]
            # Using a threshold of 0.5 for binary classification
            prediction_class = (prediction_proba >= 0.5).astype(int) 

            st.subheader("Hasil Prediksi")
            if prediction_class[0] == 1:
                st.success(f"Metode Pembayaran Diprediksi: **Kartu Kredit**")
            else:
                st.info(f"Metode Pembayaran Diprediksi: **Non-Kartu Kredit**")
            st.write(f"Probabilitas Kartu Kredit: **{prediction_proba[0]:.2f}**")
    else:
        st.warning("Model Prediksi Metode Pembayaran tidak dapat dimuat. Pastikan file model ada.")


# --- Tab 3: Cluster Segmentasi Pelanggan ---
with tab3:
    st.header("Segmentasi Pelanggan")

    if not df_cluster.empty and kmeans_model and scaler_kmeans:
        st.subheader("Analisis Hasil Clustering K-Means")
        st.write("Mean 'Total Items' dan 'Total Cost' per Cluster:")
        cluster_means = df_cluster.groupby('Cluster')[['Total_Items', 'Total_Cost']].mean()
        st.dataframe(cluster_means)

        st.subheader("Distribusi Musim di Setiap Cluster")
        df_cluster_season = df_cluster.groupby('Cluster')['Season'].value_counts(normalize=True).unstack(fill_value=0)
        fig_season, ax_season = plt.subplots(figsize=(12, 7))
        df_cluster_season.plot(kind='bar', stacked=True, colormap='viridis', ax=ax_season)
        ax_season.set_title('Distribution of Season within Each Cluster')
        ax_season.set_xlabel('Cluster')
        ax_season.set_ylabel('Proportion')
        ax_season.tick_params(axis='x', rotation=0)
        ax_season.legend(title='Season', bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig_season)
        plt.close(fig_season)

        # Prepare features for PCA/TSNE based on what was used in K-Means training
        features_for_kmeans_scaling_cols = ['Total_Items', 'Total_Cost', 'Season_Fall', 'Season_Spring', 'Season_Summer', 'Season_Winter']
        
        # Ensure OHE season columns exist in df_cluster for consistent scaling
        for season_val in ['Fall', 'Spring', 'Summer', 'Winter']:
            col_name = f'Season_{season_val}'
            if col_name not in df_cluster.columns:
                df_cluster[col_name] = (df_cluster['Season'] == season_val).astype(int) # Add as int for consistency

        # Select and scale the features from df_cluster
        features_for_kmeans_scaling_data = df_cluster[features_for_kmeans_scaling_cols]
        scaled_features_for_viz = scaler_kmeans.transform(features_for_kmeans_scaling_data)

        st.subheader("Visualisasi Kluster dengan PCA")
        pca = PCA(n_components=2, random_state=42)
        pca_components = pca.fit_transform(scaled_features_for_viz)
        df_pca = pd.DataFrame(data=pca_components, columns=['PC1', 'PC2'], index=df_cluster.index)
        df_pca['Cluster'] = df_cluster['Cluster']
        df_pca['Season'] = df_cluster['Season']

        fig_pca, ax_pca = plt.subplots(figsize=(10, 7))
        sns.scatterplot(x='PC1', y='PC2', hue='Cluster', palette='viridis', data=df_pca, s=100, alpha=0.7, ax=ax_pca)
        ax_pca.set_title('K-Means Clusters Visualized with PCA')
        ax_pca.set_xlabel('Principal Component 1')
        ax_pca.set_ylabel('Principal Component 2')
        ax_pca.legend(title='Cluster')
        ax_pca.grid(True)
        st.pyplot(fig_pca)
        plt.close(fig_pca)

        st.subheader("Visualisasi Kluster dengan t-SNE")
        if scaled_features_for_viz.shape[0] > 5000:
            st.warning("t-SNE dapat memakan waktu lama untuk dataset besar. Menampilkan visualisasi t-SNE mungkin lambat.")
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300) 
        tsne_components = tsne.fit_transform(scaled_features_for_viz)
        df_tsne = pd.DataFrame(data=tsne_components, columns=['TSNE1', 'TSNE2'], index=df_cluster.index)
        df_tsne['Cluster'] = df_cluster['Cluster']
        df_tsne['Season'] = df_cluster['Season']

        fig_tsne_cluster, ax_tsne_cluster = plt.subplots(figsize=(10, 7))
        sns.scatterplot(x='TSNE1', y='TSNE2', hue='Cluster', palette='viridis', data=df_tsne, s=100, alpha=0.7, ax=ax_tsne_cluster)
        ax_tsne_cluster.set_title('K-Means Clusters Visualized with t-SNE')
        ax_tsne_cluster.set_xlabel('t-SNE Component 1')
        ax_tsne_cluster.set_ylabel('t-SNE Component 2')
        ax_tsne_cluster.legend(title='Cluster')
        ax_tsne_cluster.grid(True)
        st.pyplot(fig_tsne_cluster)
        plt.close(fig_tsne_cluster)

        st.subheader("Visualisasi Musim Asli dengan t-SNE (Untuk Perbandingan)")
        fig_tsne_season, ax_tsne_season = plt.subplots(figsize=(10, 7))
        sns.scatterplot(x='TSNE1', y='TSNE2', hue='Season', palette='tab10', data=df_tsne, s=100, alpha=0.7, ax=ax_tsne_season)
        ax_tsne_season.set_title('Original Season Visualized with t-SNE')
        ax_tsne_season.set_xlabel('t-SNE Component 1')
        ax_tsne_season.set_ylabel('t-SNE Component 2')
        ax_tsne_season.legend(title='Season')
        ax_tsne_season.grid(True)
        st.pyplot(fig_tsne_season)
        plt.close(fig_tsne_season)

    else:
        st.warning("Data kluster atau model K-Means/scaler tidak dapat dimuat. Pastikan file ada.")
