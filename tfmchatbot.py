import streamlit as st
import pandas as pd
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from google.cloud import storage
from io import StringIO

# 📌 Configuración de Google Cloud Storage
BUCKET_NAME = "monitoreo_gcp_bucket"
ARCHIVO_DATOS = "dataset_monitoreo_servers.csv"

# 📌 Función para cargar datos desde GCP
@st.cache_data
def cargar_datos():
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(ARCHIVO_DATOS)
        contenido = blob.download_as_text()
        df = pd.read_csv(StringIO(contenido))
        return df
    except Exception as e:
        st.error(f"❌ Error al descargar el dataset: {e}")
        return None

# 📌 Inicializar el ChatBot
chatbot = ChatBot("Soporte TI")
trainer = ListTrainer(chatbot)

# 📌 Entrenar el chatbot con preguntas básicas
trainer.train([
    "¿Qué modelos de clasificación están disponibles?",
    "Los modelos disponibles son Árbol de Decisión, Regresión Logística y Random Forest.",
    "¿Cuál es la precisión del modelo Random Forest?",
    "La precisión del modelo Random Forest es del 98%.",
    "¿Qué métricas se monitorean en el sistema?",
    "Las métricas monitoreadas incluyen Uso de CPU, Temperatura, Carga de Red y Latencia."
])

# 📌 Función para generar respuestas dinámicas
def responder_pregunta(pregunta, df):
    pregunta = pregunta.lower()

    # 🔹 Consultar estado del sistema
    if "estado del sistema" in pregunta:
        estados = df["Estado del Sistema"].value_counts().to_dict()
        respuesta = "📊 Distribución del estado del sistema:\n" + "\n".join([f"{k}: {v} servidores" for k, v in estados.items()])
        return respuesta

    # 🔹 Consultar servidores críticos
    elif "servidores en estado crítico" in pregunta:
        criticos = df[df["Estado del Sistema"] == "Crítico"].shape[0]
        return f"🚨 Actualmente hay {criticos} servidores en estado crítico."

    # 🔹 Consultar cantidad de registros en el dataset
    elif "cuántos registros tiene el dataset" in pregunta:
        return f"📂 El dataset contiene {df.shape[0]} registros."

    # 🔹 Pregunta general al chatbot
    else:
        return chatbot.get_response(pregunta)

# 📌 Interfaz en Streamlit
st.title("🤖 ChatBot de Soporte TI")

df = cargar_datos()
if df is not None:
    pregunta = st.text_input("Escribe tu pregunta:")
    
    if st.button("Enviar"):
        respuesta = responder_pregunta(pregunta, df)
        st.text_area("🤖 Respuesta:", value=respuesta, height=150)
