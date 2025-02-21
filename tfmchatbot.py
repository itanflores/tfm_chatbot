import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from google.cloud import storage
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import boto3

# 📌 Configuración de los Buckets
BUCKET_GCP = "monitoreo_gcp_bucket"
BUCKET_S3 = "tfm-monitoring-data"
ARCHIVO_DATOS = "dataset_monitoreo_servers_EJEMPLO.csv"

# Diccionario con los nombres de los datasets procesados para cada modelo
ARCHIVOS_PROCESADOS = {
    "Árbol de Decisión": "dataset_procesado_arbol_decision.csv",
    "Regresión Logística": "dataset_procesado_regresion_logistica.csv",
    "Random Forest": "dataset_procesado_random_forest.csv"
}

# Inicializar cliente de Google Cloud Storage
storage_client = storage.Client()
bucket_gcp = storage_client.bucket(BUCKET_GCP)

# Inicializar cliente de AWS S3
s3_client = boto3.client("s3")

# 📌 Función para cargar los datos desde GCP Storage
@st.cache_data
def cargar_datos():
    try:
        blob = bucket_gcp.blob(ARCHIVO_DATOS)
        contenido = blob.download_as_text()
        df = pd.read_csv(StringIO(contenido))

        # Verificar las columnas cargadas
        st.write("✅ Columnas cargadas:", df.columns)
        return df
    except Exception as e:
        st.error(f"❌ Error al descargar el archivo desde GCP: {e}")
        return None

# 📌 Procesamiento de Datos
def procesar_datos(df, modelo):
    df_procesado = df.copy()

    # Convertir fecha y limpiar datos
    df_procesado["Fecha"] = pd.to_datetime(df_procesado["Fecha"], errors="coerce")
    df_procesado.drop_duplicates(inplace=True)
    df_procesado.dropna(inplace=True)

    # Verificar existencia de 'Estado del Sistema'
    if "Estado del Sistema" in df_procesado.columns:
        estado_mapping = {"Inactivo": 0, "Normal": 1, "Advertencia": 2, "Crítico": 3}
        df_procesado["Estado del Sistema Codificado"] = df_procesado["Estado del Sistema"].map(estado_mapping)
        # Evitar FutureWarning usando asignación en lugar de inplace=True
        df_procesado["Estado del Sistema Codificado"] = df_procesado["Estado del Sistema Codificado"].fillna(-1)
    else:
        st.error("⚠️ La columna 'Estado del Sistema' no está en el dataset.")
        return None  # Detener si falta esta columna

    # One-Hot Encoding para 'Tipo de Servidor'
    df_procesado = pd.get_dummies(df_procesado, columns=["Tipo de Servidor"], prefix="Servidor", drop_first=True)

    # Normalizar métricas
    scaler = MinMaxScaler()
    metricas_continuas = ["Uso CPU (%)", "Temperatura (°C)", "Carga de Red (MB/s)", "Latencia Red (ms)"]
    df_procesado[metricas_continuas] = scaler.fit_transform(df_procesado[metricas_continuas])

    return df_procesado

# 📌 Entrenar modelos y exportar datos
def entrenar_modelos():
    df = cargar_datos()
    if df is None:
        st.error("❌ No se pudo cargar el dataset.")
        return

    for modelo in ARCHIVOS_PROCESADOS.keys():
        df_procesado = procesar_datos(df, modelo)
        if df_procesado is not None:
            # Excluir la columna original 'Estado del Sistema' que contiene texto
            X = df_procesado.drop(["Estado del Sistema", "Estado del Sistema Codificado", 
                                   "Fecha", "Hostname"], axis=1, errors="ignore")
            y = df_procesado["Estado del Sistema Codificado"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )

            # Seleccionar modelo
            if modelo == "Árbol de Decisión":
                clf = DecisionTreeClassifier(random_state=42)
            elif modelo == "Regresión Logística":
                clf = LogisticRegression(max_iter=1000, random_state=42)
            else:
                clf = RandomForestClassifier(random_state=42, n_jobs=-1)

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            precision = accuracy_score(y_test, y_pred)

            # Guardar dataset procesado en GCP
            archivo_salida = ARCHIVOS_PROCESADOS[modelo]
            blob_procesado = bucket_gcp.blob(archivo_salida)
            blob_procesado.upload_from_string(df_procesado.to_csv(index=False), content_type="text/csv")

            st.success(f"✅ {modelo} entrenado con precisión: {precision:.2%}")
            st.success(f"📤 Datos exportados a GCP: {BUCKET_GCP}/{archivo_salida}")

# -----------------------------------------------------------------------------
#                           STREAMLIT UI
# -----------------------------------------------------------------------------

# Título principal de la app
st.title("Aplicación: ChatBot y Cargar/Enviar Datasets")

# Crea solo dos pestañas
tab_chatbot, tab_datasets = st.tabs([
    "🤖 ChatBot de Soporte", 
    "📂 Cargar y Enviar Datasets"
])

# ===================== PESTAÑA: ChatBot de Soporte ============================
with tab_chatbot:
    st.subheader("🤖 ChatBot de Soporte TI")
    st.write("Puedes hacer preguntas sobre los modelos y datos.")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    pregunta = st.text_input("Escribe tu pregunta:")
    if st.button("Enviar"):
        # Respuesta de ejemplo (puede sustituirse por lógica más compleja)
        respuesta = "Todavía no tengo respuesta para esto. 🚀"
        st.session_state["chat_history"].append(f"👤 {pregunta}")
        st.session_state["chat_history"].append(f"🤖 {respuesta}")

    for msg in st.session_state["chat_history"]:
        st.write(msg)

# ===================== PESTAÑA: Cargar y Enviar Datasets ======================
with tab_datasets:
    st.subheader("📂 Cargar y Enviar Datasets")

    # Subida a GCP
    st.markdown("### 🌍 Subir un archivo CSV a GCP")
    archivo_gcp = st.file_uploader("Selecciona un archivo CSV para GCP", type=["csv"])
    if archivo_gcp and st.button("📤 Enviar a GCP"):
        blob = bucket_gcp.blob(archivo_gcp.name)
        blob.upload_from_file(archivo_gcp)
        st.success(f"✅ Archivo '{archivo_gcp.name}' subido a GCP ({BUCKET_GCP}) correctamente.")

    # Subida a S3
    st.markdown("### ☁️ Subir un archivo CSV a Amazon S3")
    archivo_s3 = st.file_uploader("Selecciona un archivo CSV para S3", type=["csv"])
    if archivo_s3 and st.button("📤 Enviar a S3"):
        s3_client.upload_fileobj(archivo_s3, BUCKET_S3, archivo_s3.name)
        st.success(f"✅ Archivo '{archivo_s3.name}' subido a S3 ({BUCKET_S3}) correctamente.")

    # Botón para procesar modelos (opcional)
    if st.button("⚙️ Procesar Modelos"):
        entrenar_modelos()
