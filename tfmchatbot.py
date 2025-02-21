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

# 📌 Configuración de Google Cloud Storage
BUCKET_NAME = "monitoreo_gcp_bucket"
ARCHIVO_DATOS = "dataset_monitoreo_servers.csv"
ARCHIVOS_PROCESADOS = {
    "Árbol de Decisión": "dataset_procesado_arbol_decision.csv",
    "Regresión Logística": "dataset_procesado_regresion_logistica.csv",
    "Random Forest": "dataset_procesado_random_forest.csv"
}

# 📌 Configuración de AWS S3
AWS_BUCKET_NAME = "tfm-monitoring-data"
s3_client = boto3.client("s3")

# 📌 Inicializar cliente de Google Cloud Storage
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)

# 📌 Estado de procesamiento de datos
if "datos_procesados" not in st.session_state:
    st.session_state["datos_procesados"] = {}
if "modelos_listos" not in st.session_state:
    st.session_state["modelos_listos"] = False

# 📌 Función para cargar datos desde GCP
def cargar_datos():
    try:
        blob = bucket.blob(ARCHIVO_DATOS)
        contenido = blob.download_as_text()
        df = pd.read_csv(StringIO(contenido))
        return df
    except Exception as e:
        st.error(f"🚨 Error al descargar el dataset: {e}")
        return None

# 📌 Función para procesar los datos
def procesar_datos(df, modelo):
    df_procesado = df.copy()

    # Convertir fecha y limpiar datos
    df_procesado["Fecha"] = pd.to_datetime(df_procesado["Fecha"], errors="coerce")
    df_procesado.drop_duplicates(inplace=True)
    df_procesado.dropna(inplace=True)

    # Verificar si 'Estado del Sistema' está en el dataset
    if "Estado del Sistema" in df_procesado.columns:
        estado_mapping = {"Inactivo": 0, "Normal": 1, "Advertencia": 2, "Crítico": 3}
        df_procesado["Estado del Sistema Codificado"] = df_procesado["Estado del Sistema"].map(estado_mapping)
        
        # Si hay valores no mapeados, llenar con -1
        df_procesado["Estado del Sistema Codificado"].fillna(-1, inplace=True)
    else:
        st.error("⚠️ La columna 'Estado del Sistema' no está en el dataset.")
        return None  # Detener el procesamiento si la columna no existe

    # Aplicar One-Hot Encoding a 'Tipo de Servidor'
    df_procesado = pd.get_dummies(df_procesado, columns=["Tipo de Servidor"], prefix="Servidor", drop_first=True)

    # Normalizar métricas
    scaler = MinMaxScaler()
    metricas_continuas = ["Uso CPU (%)", "Temperatura (°C)", "Carga de Red (MB/s)", "Latencia Red (ms)"]
    df_procesado[metricas_continuas] = scaler.fit_transform(df_procesado[metricas_continuas])

    return df_procesado


# 📌 Procesamiento de modelos
def entrenar_modelos():
    df = cargar_datos()
    if df is None:
        return
    
    modelos = {
        "Árbol de Decisión": DecisionTreeClassifier(max_depth=5),
        "Regresión Logística": LogisticRegression(max_iter=50, n_jobs=-1),
        "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
    }
    
    X = df.drop(["Estado del Sistema", "Estado del Sistema Codificado", "Fecha", "Hostname"], axis=1, errors="ignore")
    y = df["Estado del Sistema Codificado"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    for modelo, clf in modelos.items():
        st.write(f"🔄 Entrenando {modelo}...")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        precision = accuracy_score(y_test, y_pred)
        
        st.session_state["datos_procesados"][modelo] = procesar_datos(df, modelo)
        
        archivo_salida = ARCHIVOS_PROCESADOS[modelo]
        blob_procesado = bucket.blob(archivo_salida)
        blob_procesado.upload_from_string(st.session_state["datos_procesados"][modelo].to_csv(index=False), content_type="text/csv")
        
        st.success(f"✅ {modelo} entrenado con precisión {precision:.2%}. Datos guardados en {archivo_salida}")
    
    st.session_state["modelos_listos"] = True
    st.success("🚀 Todos los modelos han sido entrenados y los datasets han sido generados.")

# 📌 Función para subir archivos a AWS S3
def subir_a_s3(archivo):
    try:
        s3_client.upload_fileobj(archivo, AWS_BUCKET_NAME, archivo.name)
        st.success(f"✅ Archivo '{archivo.name}' subido a S3 ({AWS_BUCKET_NAME}) correctamente.")
    except Exception as e:
        st.error(f"❌ Error al subir archivo a S3: {e}")

# 📌 UI Streamlit
st.title("📊 Comparación de Modelos de Clasificación")

tabs = st.tabs(["🌳 Árbol de Decisión", "📈 Regresión Logística", "🌲 Random Forest", "🤖 ChatBot de Soporte", "📂 Cargar y Enviar Datasets"])

# 📌 Sección Modelos
for tab, modelo in zip(tabs[:3], ARCHIVOS_PROCESADOS.keys()):
    with tab:
        st.subheader(modelo)
        if st.session_state["modelos_listos"]:
            st.success(f"✅ {modelo} listo para análisis.")
        else:
            st.warning("⚠️ Los modelos aún no han sido procesados.")

# 📌 Sección Chatbot (Solo si los modelos están listos)
with tabs[3]:
    st.subheader("🤖 ChatBot de Soporte TI")
    if not st.session_state["modelos_listos"]:
        st.warning("⚠️ El ChatBot se activará después de procesar los modelos.")
    else:
        pregunta = st.text_input("Escribe tu pregunta:")
        if st.button("Enviar"):
            if "temperatura" in pregunta.lower():
                temperatura_promedio = st.session_state["datos_procesados"]["Random Forest"]["Temperatura (°C)"].mean()
                st.write(f"🌡 La temperatura promedio de los servidores es {temperatura_promedio:.2f}°C.")
            else:
                st.write("🤖 Lo siento, aún estoy aprendiendo. Intenta otra pregunta.")

# 📌 Sección Cargar y Enviar Datasets
with tabs[4]:
    st.subheader("📂 Cargar y Enviar Datasets")
    
    opciones = st.radio("Selecciona destino:", ["Subir a GCP", "Subir a S3"])
    
    archivo_subido = st.file_uploader("📤 Sube un archivo CSV", type="csv")
    
    if archivo_subido:
        df_vista = pd.read_csv(archivo_subido)
        st.dataframe(df_vista.head())
        
        if opciones == "Subir a GCP":
            if st.button("Enviar a GCP"):
                blob = bucket.blob(archivo_subido.name)
                blob.upload_from_string(archivo_subido.getvalue(), content_type="text/csv")
                st.success(f"✅ Archivo '{archivo_subido.name}' subido a GCP ({BUCKET_NAME}) correctamente.")
        
        elif opciones == "Subir a S3":
            if st.button("Enviar a S3"):
                subir_a_s3(archivo_subido)

# 📌 Botón para procesar modelos
if st.button("⚙️ Procesar Modelos"):
    entrenar_modelos()
