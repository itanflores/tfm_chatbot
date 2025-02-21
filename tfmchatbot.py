import streamlit as st
import pandas as pd
from google.cloud import storage
import boto3
from io import StringIO

# 📌 Configuración de Buckets
BUCKET_GCP = "monitoreo_gcp_bucket"
BUCKET_S3 = "tfm-monitoring-data"

# Inicializar clientes de almacenamiento en la nube
storage_client = storage.Client()
gcp_bucket = storage_client.bucket(BUCKET_GCP)

s3_client = boto3.client("s3")

# 📌 Sección Principal con Pestañas
st.title("📊 Comparación de Modelos de Clasificación")

tab1, tab2, tab3, tab_chatbot, tab_upload = st.tabs([
    "🌳 Árbol de Decisión",
    "📈 Regresión Logística",
    "🌲 Random Forest",
    "🤖 ChatBot de Soporte",
    "📤 Cargar y Enviar Datasets"
])

# 📌 Sección de Comparación de Modelos (Ya existente en tu código)
with tab1:
    st.subheader("🌳 Árbol de Decisión")
    st.write("Aquí se mostrarán los resultados del modelo de Árbol de Decisión.")

with tab2:
    st.subheader("📈 Regresión Logística")
    st.write("Aquí se mostrarán los resultados del modelo de Regresión Logística.")

with tab3:
    st.subheader("🌲 Random Forest")
    st.write("Aquí se mostrarán los resultados del modelo Random Forest.")

with tab_chatbot:
    st.subheader("🤖 ChatBot de Soporte TI")
    st.write("Puedes hacer preguntas sobre los modelos y datos.")

# 📌 NUEVA SECCIÓN: Cargar y Enviar Datasets
with tab_upload:
    st.header("📤 Cargar y Enviar Datasets")

    # 📌 Pestañas para elegir entre GCP y S3
    tab_gcp, tab_s3 = st.tabs(["📥 Subir a GCP", "☁️ Subir a S3"])

    with tab_gcp:
        st.subheader("📥 Subir un archivo CSV a Google Cloud Storage")

        uploaded_file = st.file_uploader("Selecciona un archivo CSV", type=["csv"])

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("📊 **Vista previa del dataset:**")
            st.dataframe(df.head())

            if st.button("📤 Enviar a GCP"):
                try:
                    blob = gcp_bucket.blob(uploaded_file.name)
                    blob.upload_from_string(uploaded_file.getvalue().decode("utf-8"), content_type="text/csv")
                    st.success(f"✅ Archivo '{uploaded_file.name}' subido a GCP ({BUCKET_GCP}) correctamente.")
                except Exception as e:
                    st.error(f"❌ Error al subir archivo a GCP: {e}")

    with tab_s3:
        st.subheader("☁️ Subir un archivo CSV a Amazon S3")

        uploaded_file_s3 = st.file_uploader("Selecciona un archivo CSV para S3", type=["csv"], key="s3")

        if uploaded_file_s3 is not None:
            df_s3 = pd.read_csv(uploaded_file_s3)
            st.write("📊 **Vista previa del dataset:**")
            st.dataframe(df_s3.head())

            if st.button("📤 Enviar a S3"):
                try:
                    s3_client.put_object(Bucket=BUCKET_S3, Key=uploaded_file_s3.name, Body=uploaded_file_s3.getvalue(), ContentType="text/csv")
                    st.success(f"✅ Archivo '{uploaded_file_s3.name}' subido a S3 ({BUCKET_S3}) correctamente.")
                except Exception as e:
                    st.error(f"❌ Error al subir archivo a S3: {e}")
