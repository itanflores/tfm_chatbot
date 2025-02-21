import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from google.cloud import storage
import boto3
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer

# 📌 Configuración del Cliente de Google Cloud Storage
BUCKET_NAME_GCP = "monitoreo_gcp_bucket"
BUCKET_NAME_S3 = "tfm-monitoring-data"
ARCHIVO_DATOS = "dataset_monitoreo_servers.csv"

# Diccionario con los nombres de los datasets procesados para cada modelo
ARCHIVOS_PROCESADOS = {
    "Árbol de Decisión": "dataset_procesado_arbol_decision.csv",
    "Regresión Logística": "dataset_procesado_regresion_logistica.csv",
    "Random Forest": "dataset_procesado_random_forest.csv"
}

# Inicializar clientes de almacenamiento
storage_client = storage.Client()
bucket_gcp = storage_client.bucket(BUCKET_NAME_GCP)

s3_client = boto3.client("s3")

# 📌 Función para cargar datos desde GCP
@st.cache_data
def cargar_datos():
    try:
        blob = bucket_gcp.blob(ARCHIVO_DATOS)
        contenido = blob.download_as_text()
        df = pd.read_csv(StringIO(contenido))
        return df
    except Exception as e:
        st.error(f"❌ Error al cargar el archivo desde GCP: {e}")
        return None

# 📌 Función para procesar los datos para cada modelo
def procesar_datos(df, modelo):
    df_procesado = df.copy()
    
    df_procesado["Fecha"] = pd.to_datetime(df_procesado["Fecha"], errors="coerce")
    df_procesado.drop_duplicates(inplace=True)
    df_procesado.dropna(inplace=True)
    
    estado_mapping = {"Inactivo": 0, "Normal": 1, "Advertencia": 2, "Crítico": 3}
    df_procesado["Estado del Sistema Codificado"] = df_procesado["Estado del Sistema"].map(estado_mapping)
    
    df_procesado = pd.get_dummies(df_procesado, columns=["Tipo de Servidor"], drop_first=True)
    
    scaler = MinMaxScaler()
    metricas = ["Uso CPU (%)", "Temperatura (°C)", "Carga de Red (MB/s)", "Latencia Red (ms)"]
    
    if modelo == "Regresión Logística":
        df_procesado[metricas] = (df_procesado[metricas] - df_procesado[metricas].mean()) / df_procesado[metricas].std()
    else:
        df_procesado[metricas] = scaler.fit_transform(df_procesado[metricas])
    
    return df_procesado

# 📌 Procesamiento automático después de subir a GCP
if "datos_procesados" not in st.session_state:
    st.session_state["datos_procesados"] = {}

df = cargar_datos()
if df:
    for modelo in ARCHIVOS_PROCESADOS.keys():
        df_procesado = procesar_datos(df, modelo)
        archivo_salida = ARCHIVOS_PROCESADOS[modelo]
        blob = bucket_gcp.blob(archivo_salida)
        blob.upload_from_string(df_procesado.to_csv(index=False), content_type="text/csv")
        st.session_state["datos_procesados"][modelo] = df_procesado

# 📌 Chatbot solo se activa si los datasets están procesados
datasets_cargados = all(modelo in st.session_state["datos_procesados"] for modelo in ARCHIVOS_PROCESADOS.keys())

# 📌 SECCIÓN: COMPARACIÓN DE MODELOS
st.header("📊 Comparación de Modelos de Clasificación")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🌳 Árbol de Decisión", "📈 Regresión Logística", "🌲 Random Forest", "🤖 ChatBot de Soporte", "📂 Cargar y Enviar Datasets"])

# 📌 Sección para cada modelo
for tab, modelo in zip([tab1, tab2, tab3], ARCHIVOS_PROCESADOS.keys()):
    with tab:
        st.subheader(modelo)
        st.write(f"Aquí se mostrarán los resultados del modelo de {modelo}.")

# 📌 Sección del ChatBot
with tab4:
    st.subheader("🤖 ChatBot de Soporte TI")
    
    if datasets_cargados:
        chatbot = ChatBot("Soporte TI")
        trainer = ListTrainer(chatbot)

        preguntas_respuestas = [
            ("¿Cuántos servidores están en estado crítico?", "Déjame revisar los datos..."),
            ("¿Cuántos registros tiene el dataset?", f"El dataset tiene {len(df)} registros."),
            ("¿Cuál es la temperatura promedio de los servidores?", f"La temperatura promedio es {df['Temperatura (°C)'].mean():.2f}°C.")
        ]

        for pregunta, respuesta in preguntas_respuestas:
            trainer.train([pregunta, respuesta])

        pregunta_usuario = st.text_input("Escribe tu pregunta:")

        if st.button("Enviar"):
            respuesta = chatbot.get_response(pregunta_usuario)
            st.text_area("🤖 Respuesta:", value=str(respuesta), height=100)
    else:
        st.warning("⚠️ El ChatBot se activará cuando los datos estén procesados.")

# 📌 Sección para carga y envío de datasets
with tab5:
    st.subheader("📂 Cargar y Enviar Datasets")
    
    sub_tab1, sub_tab2 = st.tabs(["☁️ Subir a GCP", "🌤 Subir a S3"])
    
    with sub_tab1:
        st.subheader("☁️ Subir un archivo CSV a GCP")
        archivo_gcp = st.file_uploader("Selecciona un archivo CSV para GCP", type=["csv"])
        
        if archivo_gcp and st.button("📤 Enviar a GCP"):
            try:
                blob = bucket_gcp.blob(archivo_gcp.name)
                blob.upload_from_string(archivo_gcp.getvalue(), content_type="text/csv")
                st.success(f"✅ Archivo '{archivo_gcp.name}' subido a GCP ({BUCKET_NAME_GCP}) correctamente.")
            except Exception as e:
                st.error(f"❌ Error al subir el archivo a GCP: {e}")

    with sub_tab2:
        st.subheader("🌤 Subir un archivo CSV a Amazon S3")
        archivo_s3 = st.file_uploader("Selecciona un archivo CSV para S3", type=["csv"])

        if archivo_s3 and st.button("📤 Enviar a S3"):
            try:
                s3_client.upload_fileobj(archivo_s3, BUCKET_NAME_S3, archivo_s3.name)
                st.success(f"✅ Archivo '{archivo_s3.name}' subido a S3 ({BUCKET_NAME_S3}) correctamente.")
            except Exception as e:
                st.error(f"❌ Error al subir el archivo a S3: {e}")
