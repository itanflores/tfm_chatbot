import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from google.cloud import storage
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import boto3
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

# 1. Configuraciones iniciales optimizadas
BUCKET_NAME = "monitoreo_gcp_bucket"
S3_BUCKET_NAME = "tfm-monitoring-data"
ARCHIVO_DATOS = "dataset_monitoreo_servers.csv"

ARCHIVOS_PROCESADOS = {
    "Árbol de Decisión": "dataset_procesado_arbol_decision.csv",
    "Regresión Logística": "dataset_procesado_regresion_logistica.csv",
    "Random Forest": "dataset_procesado_random_forest.csv"
}

# 2. Clientes de cloud inicializados una sola vez
@st.cache_resource
def init_cloud_clients():
    return {
        "gcp": storage.Client().bucket(BUCKET_NAME),
        "s3": boto3.client("s3")
    }

clients = init_cloud_clients()

# 3. Chatbot optimizado con caché
@st.cache_resource
def init_chatbot():
    bot = ChatBot("Soporte TI")
    trainer = ChatterBotCorpusTrainer(bot)
    trainer.train("chatterbot.corpus.spanish")
    return bot

chatbot = init_chatbot()

# 4. Carga de datos más eficiente
@st.cache_data
def cargar_datos():
    try:
        blob = clients["gcp"].blob(ARCHIVO_DATOS)
        return pd.read_csv(StringIO(blob.download_as_text()))
    except Exception as e:
        st.error(f"Error al cargar datos: {str(e)}")
        st.stop()

df = cargar_datos()

# 5. Procesamiento unificado de datos
def procesar_datos(df):
    df = df.copy()
    
    # Limpieza básica
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    df = df.dropna().drop_duplicates()
    
    # Codificación más eficiente
    estado_mapping = {"Inactivo": 0, "Normal": 1, "Advertencia": 2, "Crítico": 3}
    df["Estado Codificado"] = df["Estado del Sistema"].map(estado_mapping)
    
    # Normalización optimizada
    scaler = MinMaxScaler()
    metricas = ["Uso CPU (%)", "Temperatura (°C)", "Carga de Red (MB/s)", "Latencia Red (ms)"]
    df[metricas] = scaler.fit_transform(df[metricas])
    
    # One-Hot Encoding más eficiente
    return pd.get_dummies(df, columns=["Tipo de Servidor"], prefix="Servidor")

# 6. Gestión centralizada de datos procesados
if "datos_procesados" not in st.session_state:
    st.session_state.datos_procesados = {}

# 7. Funciones de exportación mejoradas
def exportar_datos(modelo):
    try:
        df_procesado = st.session_state.datos_procesados.get(modelo)
        if df_procesado is None:
            raise ValueError("Datos no encontrados")
            
        blob = clients["gcp"].blob(ARCHIVOS_PROCESADOS[modelo])
        blob.upload_from_string(df_procesado.to_csv(index=False), "text/csv")
        st.success(f"Datos de {modelo} guardados en GCP!")
    except Exception as e:
        st.error(f"Error en GCP: {str(e)}")

def subir_a_s3(modelo):
    try:
        df_procesado = st.session_state.datos_procesados.get(modelo)
        if df_procesado is None:
            raise ValueError("Datos no encontrados")
            
        clients["s3"].put_object(
            Bucket=S3_BUCKET_NAME,
            Key=ARCHIVOS_PROCESADOS[modelo],
            Body=df_procesado.to_csv(index=False)
        )
        st.success(f"Datos de {modelo} enviados a S3!")
    except Exception as e:
        st.error(f"Error en S3: {str(e)}")

# 8. Interfaz de usuario reorganizada
st.header("📊 Comparación de Modelos de Clasificación")
tabs = st.tabs([*ARCHIVOS_PROCESADOS.keys(), "🤖 ChatBot de Soporte"])

for i, modelo in enumerate(ARCHIVOS_PROCESADOS):
    with tabs[i]:
        st.subheader(modelo)
        
        # Sección de procesamiento
        if st.button(f"⚙️ Procesar datos para {modelo}"):
            st.session_state.datos_procesados[modelo] = procesar_datos(df)
            st.success("Datos procesados correctamente!")
        
        # Sección de exportación
        if modelo in st.session_state.datos_procesados:
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"📤 GCP - {modelo}"):
                    exportar_datos(modelo)
            with col2:
                if st.button(f"🚀 S3 - {modelo}"):
                    subir_a_s3(modelo)

# 9. Chatbot mejorado
with tabs[-1]:
    st.subheader("🤖 ChatBot de Soporte TI")
    
    # Historial de conversación
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Mostrar historial
    for msg in st.session_state.chat_history:
        st.markdown(f"**{msg['role']}:** {msg['content']}")
    
    # Entrada de usuario
    user_input = st.text_input("Escribe tu pregunta:", key="user_input")
    if st.button("Enviar") and user_input:
        response = chatbot.get_response(user_input)
        st.session_state.chat_history.append({"role": "Usuario", "content": user_input})
        st.session_state.chat_history.append({"role": "Asistente", "content": str(response)})
        st.experimental_rerun()
