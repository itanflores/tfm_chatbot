import streamlit as st
import pandas as pd
from google.cloud import storage
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer

# 📌 Configuración del Cliente de Google Cloud Storage
BUCKET_NAME = "monitoreo_gcp_bucket"
ARCHIVOS_PROCESADOS = {
    "Árbol de Decisión": "dataset_procesado_arbol_decision.csv",
    "Regresión Logística": "dataset_procesado_regresion_logistica.csv",
    "Random Forest": "dataset_procesado_random_forest.csv"
}

# Inicializar cliente de Google Cloud Storage
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)

# 📌 Función para cargar los datasets desde GCP
@st.cache_data
def cargar_datos():
    datos = {}
    for modelo, archivo in ARCHIVOS_PROCESADOS.items():
        try:
            blob = bucket.blob(archivo)
            contenido = blob.download_as_text()
            df = pd.read_csv(pd.compat.StringIO(contenido))
            datos[modelo] = df
        except Exception as e:
            st.error(f"❌ Error al cargar {archivo}: {e}")
    return datos

# Cargar los datos en memoria
datasets = cargar_datos()

# 📌 Inicializar ChatBot con entrenamiento básico
chatbot = ChatBot("Soporte TI")
trainer = ListTrainer(chatbot)

# Entrenar el chatbot con preguntas comunes
trainer.train([
    "¿Cuántos servidores están en estado crítico?",
    "Voy a revisar el dataset para encontrar esa información.",
    "¿Cuántos registros tiene el dataset?",
    "Déjame contar los registros en el dataset.",
    "¿Cuál es la temperatura promedio de los servidores?",
    "Voy a calcular la temperatura promedio con los datos disponibles."
])

# 📌 Función para procesar preguntas del usuario
def responder_pregunta(pregunta):
    pregunta = pregunta.lower()
    
    if "estado crítico" in pregunta:
        # Contar servidores en estado crítico en cada dataset
        respuesta = ""
        for modelo, df in datasets.items():
            if "Estado del Sistema" in df.columns:
                criticos = df[df["Estado del Sistema"] == "Crítico"].shape[0]
                respuesta += f"🔹 {modelo}: {criticos} servidores en estado crítico.\n"
        return respuesta if respuesta else "No se encontró información sobre servidores críticos."
    
    elif "registros" in pregunta:
        # Contar registros en cada dataset
        respuesta = ""
        for modelo, df in datasets.items():
            respuesta += f"🔹 {modelo}: {df.shape[0]} registros.\n"
        return respuesta
    
    elif "temperatura promedio" in pregunta:
        # Calcular temperatura promedio en cada dataset
        respuesta = ""
        for modelo, df in datasets.items():
            if "Temperatura (°C)" in df.columns:
                temp_promedio = df["Temperatura (°C)"].mean()
                respuesta += f"🔹 {modelo}: {temp_promedio:.2f}°C de temperatura promedio.\n"
        return respuesta if respuesta else "No se encontró información de temperatura en los datasets."
    
    # Si la pregunta no coincide con ninguna consulta, usar el chatbot pre-entrenado
    return str(chatbot.get_response(pregunta))

# 📌 Interfaz en Streamlit
st.title("🤖 ChatBot de Soporte TI")
st.write("Puedes preguntarme sobre los datasets de monitoreo de servidores y modelos de clasificación.")

# Campo de entrada del usuario
pregunta_usuario = st.text_input("Escribe tu pregunta:")

# Botón para enviar pregunta
if st.button("Enviar"):
    if pregunta_usuario:
        respuesta = responder_pregunta(pregunta_usuario)
        st.text_area("🤖 Respuesta:", value=respuesta, height=100, max_chars=None)
    else:
        st.warning("⚠️ Por favor, escribe una pregunta.")
