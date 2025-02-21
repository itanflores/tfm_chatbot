# =============================================================================
# Name: streamlit_app_v1.2.py
# Description:
#   Aplicación Streamlit con tres pestañas:
#     1) ChatBot de Soporte TI: Responde preguntas sobre el dataset utilizando
#        únicamente el modelo que obtuvo la mayor precisión.
#     2) Cargar y Enviar Datasets: Permite subir archivos CSV a GCP y S3, entrenar
#        los modelos y descargar los resultados procesados.
#     3) Tablero Interactivo: Muestra visualizaciones y análisis basados en el dataset
#        ganador (el conjunto de datos procesado asociado al modelo con mayor precisión).
#
#   Esta versión unifica la carga de datos, el preprocesamiento y el entrenamiento de
#   modelos para garantizar consistencia, y genera un único dataset ganador que se utiliza
#   en el chatbot y en el tablero.
# Version: 1.2.0
# =============================================================================

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
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# Configuración de la página (única llamada)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Aplicación Integrada: ChatBot, Datasets y Tablero", 
                   page_icon="📊", layout="wide")

# -----------------------------------------------------------------------------
# Configuración de Buckets y Variables
# -----------------------------------------------------------------------------
BUCKET_GCP = "monitoreo_gcp_bucket"
BUCKET_S3 = "tfm-monitoring-data"
# Actualizamos el nombre del archivo según la nueva especificación
ARCHIVO_DATOS = "dataset_monitoreo_servers.csv"

# Nombres de salida para cada modelo
ARCHIVOS_PROCESADOS = {
    "Árbol de Decisión": "dataset_procesado_arbol_decision.csv",
    "Regresión Logística": "dataset_procesado_regresion_logistica.csv",
    "Random Forest": "dataset_procesado_random_forest.csv"
}

# -----------------------------------------------------------------------------
# Inicialización de clientes GCP y S3
# -----------------------------------------------------------------------------
storage_client = storage.Client()
bucket_gcp = storage_client.bucket(BUCKET_GCP)
s3_client = boto3.client("s3")

# -----------------------------------------------------------------------------
# Función para cargar el dataset desde GCP
# -----------------------------------------------------------------------------
@st.cache_data
def cargar_datos():
    """
    Descarga y carga el CSV desde GCP en un DataFrame de pandas.
    """
    try:
        blob = bucket_gcp.blob(ARCHIVO_DATOS)
        contenido = blob.download_as_text()
        df = pd.read_csv(StringIO(contenido))
        st.write("✅ Columnas cargadas:", df.columns)
        return df
    except Exception as e:
        st.error(f"❌ Error al descargar el archivo desde GCP: {e}")
        return None

# -----------------------------------------------------------------------------
# Función para procesar el dataset (unificada para todos los módulos)
# -----------------------------------------------------------------------------
def procesar_datos(df, modelo=None):
    """
    Realiza el preprocesamiento del dataset:
      - Conversión de 'Fecha' a datetime.
      - Eliminación de duplicados y registros nulos.
      - Codificación ordinal de 'Estado del Sistema'.
      - One-hot encoding para 'Tipo de Servidor'.
      - Normalización de métricas continuas mediante MinMaxScaler.
      
    Se busca homogeneizar el método con el script de backup previo.
    """
    df_procesado = df.copy()
    df_procesado["Fecha"] = pd.to_datetime(df_procesado["Fecha"], errors="coerce")
    df_procesado.drop_duplicates(inplace=True)
    df_procesado.dropna(inplace=True)
    
    if "Estado del Sistema" in df_procesado.columns:
        estado_mapping = {"Inactivo": 0, "Normal": 1, "Advertencia": 2, "Crítico": 3}
        df_procesado["Estado del Sistema Codificado"] = df_procesado["Estado del Sistema"].map(estado_mapping)
        df_procesado["Estado del Sistema Codificado"] = df_procesado["Estado del Sistema Codificado"].fillna(-1)
    else:
        st.error("⚠️ La columna 'Estado del Sistema' no está en el dataset.")
        return None

    df_procesado = pd.get_dummies(df_procesado, columns=["Tipo de Servidor"], prefix="Servidor", drop_first=True)
    scaler = MinMaxScaler()
    metricas_continuas = ["Uso CPU (%)", "Temperatura (°C)", "Carga de Red (MB/s)", "Latencia Red (ms)"]
    df_procesado[metricas_continuas] = scaler.fit_transform(df_procesado[metricas_continuas])
    
    return df_procesado

# -----------------------------------------------------------------------------
# Función para entrenar y comparar modelos, y definir el dataset ganador
# -----------------------------------------------------------------------------
def entrenar_modelos():
    """
    Carga el dataset, entrena tres modelos (Árbol de Decisión, Regresión Logística, Random Forest)
    y guarda:
      - Los DataFrames procesados en st.session_state["processed_dfs"].
      - Las precisiones en st.session_state["model_scores"].
      - El modelo con mayor precisión en st.session_state["best_model"].
      - El dataset ganador en st.session_state["best_dataset"].
    """
    df = cargar_datos()
    if df is None:
        st.error("❌ No se pudo cargar el dataset.")
        return

    st.session_state["processed_dfs"] = {}
    st.session_state["model_scores"] = {}

    for modelo in ARCHIVOS_PROCESADOS.keys():
        df_proc = procesar_datos(df, modelo)
        if df_proc is not None:
            # Preparar variables de entrada y objetivo
            X = df_proc.drop(["Estado del Sistema", "Estado del Sistema Codificado", "Fecha", "Hostname"],
                             axis=1, errors="ignore")
            y = df_proc["Estado del Sistema Codificado"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
            
            if modelo == "Árbol de Decisión":
                clf = DecisionTreeClassifier(random_state=42)
            elif modelo == "Regresión Logística":
                clf = LogisticRegression(max_iter=3000, random_state=42)
            else:
                clf = RandomForestClassifier(random_state=42, n_jobs=-1)
            
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            precision = accuracy_score(y_test, y_pred)
            
            # Exportar dataset procesado a GCP
            archivo_salida = ARCHIVOS_PROCESADOS[modelo]
            blob = bucket_gcp.blob(archivo_salida)
            blob.upload_from_string(df_proc.to_csv(index=False), content_type="text/csv")
            
            st.session_state["processed_dfs"][modelo] = df_proc
            st.session_state["model_scores"][modelo] = precision
            
            st.success(f"✅ {modelo} entrenado con precisión: {precision:.2%}")
            st.success(f"📤 Datos exportados a GCP: {BUCKET_GCP}/{archivo_salida}")
    
    if st.session_state["model_scores"]:
        best_model = max(st.session_state["model_scores"], key=st.session_state["model_scores"].get)
        st.session_state["best_model"] = best_model
        st.session_state["best_dataset"] = st.session_state["processed_dfs"][best_model]
        st.success(f"El modelo con mejor precisión es: {best_model} ({st.session_state['model_scores'][best_model]:.2%}).")

# -----------------------------------------------------------------------------
# Función para subir archivos a S3 (usada en el botón de descarga)
# -----------------------------------------------------------------------------
def upload_to_s3(file_content, file_name):
    s3_client.put_object(Bucket=BUCKET_S3, Key=file_name, Body=file_content)
    st.success(f"Archivo '{file_name}' enviado a S3 correctamente.")

# -----------------------------------------------------------------------------
# Función para responder preguntas del ChatBot usando el dataset ganador
# -----------------------------------------------------------------------------
def responder_pregunta(pregunta: str) -> str:
    pregunta_lower = pregunta.lower()
    
    if "best_dataset" not in st.session_state:
        return "Aún no se ha identificado un modelo ganador. Procesa los modelos primero."
    
    df_ref = st.session_state["best_dataset"]
    best_model = st.session_state.get("best_model", "Desconocido")
    base_message = f"De acuerdo al modelo **{best_model}**, "
    
    if ("crítico" in pregunta_lower or "critico" in pregunta_lower):
        if "Estado del Sistema Codificado" in df_ref.columns:
            num_criticos = (df_ref["Estado del Sistema Codificado"] == 3).sum()
            return base_message + f"hay {num_criticos} servidores en estado crítico."
        else:
            return base_message + "no se encontró la columna 'Estado del Sistema Codificado'."
    elif ("registros" in pregunta_lower or "filas" in pregunta_lower or "dataset" in pregunta_lower):
        num_registros = df_ref.shape[0]
        return base_message + f"el dataset tiene {num_registros} registros."
    elif ("temperatura" in pregunta_lower and "promedio" in pregunta_lower):
        if "Temperatura (°C)" in df_ref.columns:
            temp_promedio = df_ref["Temperatura (°C)"].mean()
            return base_message + f"la temperatura promedio es {temp_promedio:.2f} °C."
        else:
            return base_message + "no se encontró la columna 'Temperatura (°C)'."
    else:
        return (
            "Lo siento, no reconozco esa pregunta. Prueba con:\n"
            "- ¿Cuántos servidores están en estado crítico?\n"
            "- ¿Cuántos registros tiene el dataset?\n"
            "- ¿Cuál es la temperatura promedio de los servidores?"
        )

# -----------------------------------------------------------------------------
# INTERFAZ STREAMLIT: Definir tres pestañas
# -----------------------------------------------------------------------------
st.title("Aplicación Integrada: ChatBot, Datasets y Tablero Interactivo")

tab_chatbot, tab_datasets, tab_dashboard = st.tabs([
    "🤖 ChatBot de Soporte",
    "📂 Cargar y Enviar Datasets",
    "📊 Tablero Interactivo"
])

# ---------------------- Pestaña: ChatBot de Soporte --------------------------
with tab_chatbot:
    st.subheader("🤖 ChatBot de Soporte TI")
    st.write("Este ChatBot responderá basándose en el modelo con mayor precisión.")
    st.markdown(
        """
        **Ejemplos de preguntas:**
        - ¿Cuántos servidores están en estado crítico?
        - ¿Cuántos registros tiene el dataset?
        - ¿Cuál es la temperatura promedio de los servidores?
        """
    )
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    pregunta = st.text_input("Escribe tu pregunta:")
    if st.button("Enviar"):
        respuesta = responder_pregunta(pregunta)
        st.session_state["chat_history"].append(f"**Usuario:** {pregunta}")
        st.session_state["chat_history"].append(f"**ChatBot:** {respuesta}")
    for msg in st.session_state["chat_history"]:
        st.write(msg)

# ----------------- Pestaña: Cargar y Enviar Datasets -------------------------
with tab_datasets:
    st.subheader("📂 Cargar y Enviar Datasets")
    st.markdown("### 🌍 Subir un archivo CSV a GCP")
    archivo_gcp_subir = st.file_uploader("Selecciona un archivo CSV para GCP", type=["csv"])
    if archivo_gcp_subir and st.button("📤 Enviar a GCP"):
        blob = bucket_gcp.blob(archivo_gcp_subir.name)
        blob.upload_from_file(archivo_gcp_subir)
        st.success(f"✅ Archivo '{archivo_gcp_subir.name}' subido a GCP ({BUCKET_GCP}) correctamente.")
    st.markdown("### ☁️ Subir un archivo CSV a Amazon S3")
    archivo_s3_subir = st.file_uploader("Selecciona un archivo CSV para S3", type=["csv"])
    if archivo_s3_subir and st.button("📤 Enviar a S3"):
        s3_client.upload_fileobj(archivo_s3_subir, BUCKET_S3, archivo_s3_subir.name)
        st.success(f"✅ Archivo '{archivo_s3_subir.name}' subido a S3 ({BUCKET_S3}) correctamente.")
    if st.button("⚙️ Procesar Modelos"):
        entrenar_modelos()
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

# ------------------- Pestaña: Tablero Interactivo ---------------------------
with tab_dashboard:
    st.subheader("📊 Tablero Interactivo")
    if "best_dataset" not in st.session_state:
        st.warning("Por favor, procesa los modelos para generar el dataset ganador.")
    else:
        df_dash = st.session_state["best_dataset"]
        # Ejemplo: KPI's de Distribución de Estados
        if "Estado del Sistema" in df_dash.columns:
            total_counts = df_dash["Estado del Sistema"].value_counts().reset_index()
            total_counts.columns = ["Estado", "Cantidad"]
            col1, col2, col3, col4 = st.columns(4)
            count_critico = total_counts.loc[total_counts["Estado"]=="Crítico", "Cantidad"].values[0] if "Crítico" in total_counts["Estado"].values else 0
            count_advertencia = total_counts.loc[total_counts["Estado"]=="Advertencia", "Cantidad"].values[0] if "Advertencia" in total_counts["Estado"].values else 0
            count_normal = total_counts.loc[total_counts["Estado"]=="Normal", "Cantidad"].values[0] if "Normal" in total_counts["Estado"].values else 0
            count_inactivo = total_counts.loc[total_counts["Estado"]=="Inactivo", "Cantidad"].values[0] if "Inactivo" in total_counts["Estado"].values else 0
            col1.metric("Crítico", f"{count_critico}")
            col2.metric("Advertencia", f"{count_advertencia}")
            col3.metric("Normal", f"{count_normal}")
            col4.metric("Inactivo", f"{count_inactivo}")
            
            st.markdown("#### Distribución de Estados")
            fig_pie = px.pie(total_counts, values="Cantidad", names="Estado", title="Distribución de Estados del Sistema")
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.error("La columna 'Estado del Sistema' no se encuentra en el dataset ganador.")
