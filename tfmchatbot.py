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

# -----------------------------------------------------------------------------
#                     CONFIGURACIÓN DE BUCKETS Y VARIABLES
# -----------------------------------------------------------------------------

BUCKET_GCP = "monitoreo_gcp_bucket"
BUCKET_S3 = "tfm-monitoring-data"
ARCHIVO_DATOS = "dataset_monitoreo_servers_EJEMPLO.csv"

ARCHIVOS_PROCESADOS = {
    "Árbol de Decisión": "dataset_procesado_arbol_decision.csv",
    "Regresión Logística": "dataset_procesado_regresion_logistica.csv",
    "Random Forest": "dataset_procesado_random_forest.csv"
}

# -----------------------------------------------------------------------------
#                     INICIALIZACIÓN DE CLIENTES GCP Y S3
# -----------------------------------------------------------------------------

storage_client = storage.Client()
bucket_gcp = storage_client.bucket(BUCKET_GCP)

s3_client = boto3.client("s3")

# -----------------------------------------------------------------------------
#                       FUNCIONES DE PROCESAMIENTO Y MODELOS
# -----------------------------------------------------------------------------

@st.cache_data
def cargar_datos():
    """Descarga y carga el CSV desde GCP en un DataFrame de pandas."""
    try:
        blob = bucket_gcp.blob(ARCHIVO_DATOS)
        contenido = blob.download_as_text()
        df = pd.read_csv(StringIO(contenido))
        st.write("✅ Columnas cargadas:", df.columns)
        return df
    except Exception as e:
        st.error(f"❌ Error al descargar el archivo desde GCP: {e}")
        return None


def procesar_datos(df, modelo):
    """Realiza limpieza, codificación y normalización de datos."""
    df_procesado = df.copy()

    # Convertir fecha y limpiar datos
    df_procesado["Fecha"] = pd.to_datetime(df_procesado["Fecha"], errors="coerce")
    df_procesado.drop_duplicates(inplace=True)
    df_procesado.dropna(inplace=True)

    # Codificar 'Estado del Sistema'
    if "Estado del Sistema" in df_procesado.columns:
        estado_mapping = {"Inactivo": 0, "Normal": 1, "Advertencia": 2, "Crítico": 3}
        df_procesado["Estado del Sistema Codificado"] = df_procesado["Estado del Sistema"].map(estado_mapping)
        # Evitar FutureWarning
        df_procesado["Estado del Sistema Codificado"] = df_procesado["Estado del Sistema Codificado"].fillna(-1)
    else:
        st.error("⚠️ La columna 'Estado del Sistema' no está en el dataset.")
        return None

    # One-Hot Encoding para 'Tipo de Servidor'
    df_procesado = pd.get_dummies(df_procesado, columns=["Tipo de Servidor"], prefix="Servidor", drop_first=True)

    # Normalizar métricas
    scaler = MinMaxScaler()
    metricas_continuas = ["Uso CPU (%)", "Temperatura (°C)", "Carga de Red (MB/s)", "Latencia Red (ms)"]
    df_procesado[metricas_continuas] = scaler.fit_transform(df_procesado[metricas_continuas])

    return df_procesado


def entrenar_modelos():
    """Carga datos, entrena cada modelo y guarda los DataFrames procesados."""
    df = cargar_datos()
    if df is None:
        st.error("❌ No se pudo cargar el dataset.")
        return

    # Preparar un contenedor en session_state para guardar los DF procesados
    st.session_state["processed_dfs"] = {}

    for modelo in ARCHIVOS_PROCESADOS.keys():
        df_procesado = procesar_datos(df, modelo)
        if df_procesado is not None:
            # Separar X e y
            X = df_procesado.drop(
                ["Estado del Sistema", "Estado del Sistema Codificado", "Fecha", "Hostname"],
                axis=1,
                errors="ignore"
            )
            y = df_procesado["Estado del Sistema Codificado"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )

            # Seleccionar y entrenar modelo
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

            # Guardar el DataFrame procesado en session_state
            st.session_state["processed_dfs"][modelo] = df_procesado

            st.success(f"✅ {modelo} entrenado con precisión: {precision:.2%}")
            st.success(f"📤 Datos exportados a GCP: {BUCKET_GCP}/{archivo_salida}")


# -----------------------------------------------------------------------------
#                      FUNCIONES ADICIONALES PARA DESCARGA
# -----------------------------------------------------------------------------

def upload_to_s3(file_content, file_name):
    """Función que sube el contenido a S3."""
    s3_client.put_object(
        Bucket=BUCKET_S3,
        Key=file_name,
        Body=file_content
    )
    st.success(f"Archivo '{file_name}' enviado a S3 correctamente.")


# -----------------------------------------------------------------------------
#                           INTERFAZ STREAMLIT
# -----------------------------------------------------------------------------

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
    archivo_gcp_subir = st.file_uploader("Selecciona un archivo CSV para GCP", type=["csv"])
    if archivo_gcp_subir and st.button("📤 Enviar a GCP"):
        blob = bucket_gcp.blob(archivo_gcp_subir.name)
        blob.upload_from_file(archivo_gcp_subir)
        st.success(f"✅ Archivo '{archivo_gcp_subir.name}' subido a GCP ({BUCKET_GCP}) correctamente.")

    # Subida a S3
    st.markdown("### ☁️ Subir un archivo CSV a Amazon S3")
    archivo_s3_subir = st.file_uploader("Selecciona un archivo CSV para S3", type=["csv"])
    if archivo_s3_subir and st.button("📤 Enviar a S3"):
        s3_client.upload_fileobj(archivo_s3_subir, BUCKET_S3, archivo_s3_subir.name)
        st.success(f"✅ Archivo '{archivo_s3_subir.name}' subido a S3 ({BUCKET_S3}) correctamente.")

    # Botón para procesar modelos
    if st.button("⚙️ Procesar Modelos"):
        entrenar_modelos()

    # Sección para descargar archivos procesados y subirlos a S3
    if "processed_dfs" in st.session_state and st.session_state["processed_dfs"]:
        st.write("### Descarga y envío de los archivos procesados")
        for modelo, df_proc in st.session_state["processed_dfs"].items():
            archivo_salida = ARCHIVOS_PROCESADOS[modelo]
            csv_data = df_proc.to_csv(index=False).encode('utf-8')

            st.download_button(
                label=f"Descargar y Enviar a S3 - {modelo}",
                data=csv_data,
                file_name=archivo_salida,
                mime="text/csv",
                on_click=upload_to_s3,
                args=(csv_data, archivo_salida)
            )
