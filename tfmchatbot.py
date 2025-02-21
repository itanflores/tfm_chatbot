# =============================================================================
# Name: streamlit_app_v1.3.py
# Description:
#   Aplicación Streamlit con tres pestañas principales:
#     1) ChatBot de Soporte TI: Responde preguntas sobre el dataset y el modelo,
#        incluyendo la importancia de variables y proyecciones futuras.
#     2) Cargar y Enviar Datasets: Permite subir archivos CSV a GCP y S3, entrenar
#        los modelos y descargar los resultados procesados.
#     3) Tablero Interactivo: Contiene dos subpestañas:
#          A) Monitoreo del Estado Real (dataset original)
#          B) Resultados de la Clasificación (dataset ganador, con visualizaciones de
#             importancia de variables y pronósticos de temperatura).
#
#   Se incluye la función 'upload_to_s3' correctamente definida y utilizada, para
#   evitar el error NameError.
#
# Version: 1.3.0
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
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# -----------------------------------------------------------------------------
# Configuración de la página (única llamada)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Aplicación Integrada: ChatBot, Datasets y Tablero",
                   page_icon="📊",
                   layout="wide")

# -----------------------------------------------------------------------------
# Configuración de Buckets y Variables
# -----------------------------------------------------------------------------
BUCKET_GCP = "monitoreo_gcp_bucket"
BUCKET_S3 = "tfm-monitoring-data"
ARCHIVO_DATOS = "dataset_monitoreo_servers.csv"

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
# Función para subir archivos a S3 (usada en el botón de descarga)
# -----------------------------------------------------------------------------
def upload_to_s3(file_content, file_name):
    """
    Sube el contenido del archivo a S3 y muestra un mensaje de éxito.
    Asegúrate de que BUCKET_S3 esté configurado correctamente.
    """
    s3_client.put_object(
        Bucket=BUCKET_S3,
        Key=file_name,
        Body=file_content
    )
    st.success(f"Archivo '{file_name}' enviado a S3 correctamente.")

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
        st.write("✅ Columnas cargadas (dataset original):", df.columns)
        return df
    except Exception as e:
        st.error(f"❌ Error al descargar el archivo desde GCP: {e}")
        return None

# -----------------------------------------------------------------------------
# Función para procesar el dataset (unificada para todos los módulos)
# -----------------------------------------------------------------------------
def procesar_datos(df):
    """
    Realiza el preprocesamiento del dataset:
      - Conversión de 'Fecha' a datetime.
      - Eliminación de duplicados y registros nulos.
      - Codificación ordinal de 'Estado del Sistema'.
      - One-hot encoding para 'Tipo de Servidor'.
      - Normalización de métricas continuas mediante MinMaxScaler.
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
# Función para entrenar modelos, comparar y definir el dataset ganador
# -----------------------------------------------------------------------------
def entrenar_modelos():
    """
    Entrena tres modelos (Árbol de Decisión, Regresión Logística, Random Forest) y:
      - Guarda los DataFrames procesados en st.session_state["processed_dfs"].
      - Guarda las precisiones en st.session_state["model_scores"].
      - Guarda los modelos entrenados en st.session_state["trained_models"].
      - Determina el modelo con mayor precisión y almacena:
            st.session_state["best_model"] y st.session_state["best_dataset"].
      - Almacena los nombres de las variables en st.session_state["feature_names"].
    """
    df = cargar_datos()
    if df is None:
        st.error("❌ No se pudo cargar el dataset original.")
        return

    st.session_state["original_dataset"] = df
    st.session_state["processed_dfs"] = {}
    st.session_state["model_scores"] = {}
    st.session_state["trained_models"] = {}

    for modelo in ARCHIVOS_PROCESADOS.keys():
        df_proc = procesar_datos(df)
        if df_proc is not None:
            X = df_proc.drop(["Estado del Sistema", "Estado del Sistema Codificado", "Fecha", "Hostname"],
                             axis=1, errors="ignore")
            y = df_proc["Estado del Sistema Codificado"]
            # Almacenar los nombres de las variables (se asume que son iguales para todos)
            st.session_state["feature_names"] = list(X.columns)
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
            st.session_state["trained_models"][modelo] = clf

            st.success(f"✅ {modelo} entrenado con precisión: {precision:.2%}")
            st.success(f"📤 Datos exportados a GCP: {BUCKET_GCP}/{archivo_salida}")

    if st.session_state["model_scores"]:
        best_model = max(st.session_state["model_scores"], key=st.session_state["model_scores"].get)
        st.session_state["best_model"] = best_model
        st.session_state["best_dataset"] = st.session_state["processed_dfs"][best_model]
        st.success(f"El modelo con mejor precisión es: {best_model} "
                   f"({st.session_state['model_scores'][best_model]:.2%}).")

# -----------------------------------------------------------------------------
# Función para predecir temperatura futura usando regresión lineal
# -----------------------------------------------------------------------------
def predecir_temperatura_futura(df, horizon=7):
    """
    Utiliza regresión lineal para predecir la temperatura (°C) en el futuro.
    Se convierte la fecha a ordinal para entrenar el modelo, y luego se aplica
    la conversión inversa a la escala original de temperatura.
    """
    df_forecast = df.copy().sort_values("Fecha")
    df_forecast["Fecha_ordinal"] = df_forecast["Fecha"].map(lambda x: x.toordinal())

    X = df_forecast["Fecha_ordinal"].values.reshape(-1, 1)
    y = df_forecast["Temperatura (°C)"].values  # y está en rango [0, 1] tras el escalado

    # Entrenar un modelo de regresión (p.ej. LinearRegression)
    lr = LinearRegression()
    lr.fit(X, y)

    # Generar fechas futuras
    last_date = df_forecast["Fecha"].max()
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, horizon+1)]
    future_ordinals = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)

    # Predecir valores escalados
    y_pred_scaled = lr.predict(future_ordinals)

    # ---- PASO CLAVE: Convertir de [0,1] a la escala real ----
    temp_min = st.session_state["temp_original_min"]
    temp_max = st.session_state["temp_original_max"]
    y_pred_real = y_pred_scaled * (temp_max - temp_min) + temp_min

    # Crear un DataFrame con los resultados
    forecast_df = pd.DataFrame({
        "Fecha": future_dates,
        "Temperatura Predicha (°C)": y_pred_real  # usar valores reales
    })

    return forecast_df


# -----------------------------------------------------------------------------
# Función extendida para responder preguntas del ChatBot
# -----------------------------------------------------------------------------
def responder_pregunta(pregunta: str) -> str:
    """
    Responde preguntas del usuario basándose en el dataset ganador y la información
    del modelo ganador, incluyendo:
      - Preguntas sobre el número de servidores, registros y temperatura promedio.
      - Preguntas sobre la importancia de variables.
      - Preguntas sobre proyecciones futuras (temperatura).
    """
    pregunta_lower = pregunta.lower()
    if "best_dataset" not in st.session_state:
        return "Aún no se ha definido un dataset ganador. Procesa los modelos primero."
    
    df_ref = st.session_state["best_dataset"]
    best_model = st.session_state.get("best_model", "Desconocido")
    base_message = f"De acuerdo al modelo **{best_model}**, "
    
    # Respuesta básica (número de servidores, registros, temperatura promedio)
    if "crítico" in pregunta_lower or "critico" in pregunta_lower:
        if "Estado del Sistema Codificado" in df_ref.columns:
            num_criticos = (df_ref["Estado del Sistema Codificado"] == 3).sum()
            return base_message + f"hay {num_criticos} servidores en estado crítico."
        else:
            return base_message + "no se encontró la columna 'Estado del Sistema Codificado'."
    elif "registros" in pregunta_lower or "filas" in pregunta_lower or "dataset" in pregunta_lower:
        num_registros = df_ref.shape[0]
        return base_message + f"el dataset tiene {num_registros} registros."
    elif "temperatura" in pregunta_lower and "promedio" in pregunta_lower:
        if "Temperatura (°C)" in df_ref.columns:
            temp_promedio = df_ref["Temperatura (°C)"].mean()
            return base_message + f"la temperatura promedio es {temp_promedio:.2f} °C."
        else:
            return base_message + "no se encontró la columna 'Temperatura (°C)'."
    # Nueva funcionalidad: Importancia de Variables
    elif "variables" in pregunta_lower and ("explican" in pregunta_lower or "importancia" in pregunta_lower):
        if "trained_models" in st.session_state and best_model in st.session_state["trained_models"]:
            model = st.session_state["trained_models"][best_model]
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                features = st.session_state.get("feature_names", [])
                if features:
                    df_imp = pd.DataFrame({"Variable": features, "Importancia": importances})
                    df_imp = df_imp.sort_values("Importancia", ascending=False)
                    top_features = df_imp.head(3)
                    response = "Las variables que más explican el estado son: " + ", ".join(
                        f"{row['Variable']} ({row['Importancia']*100:.1f}%)" for _, row in top_features.iterrows()
                    )
                    return response
                else:
                    return base_message + "No se encontraron nombres de variables."
            else:
                return base_message + "El modelo ganador no proporciona importancias de variables."
    # Nueva funcionalidad: Proyección de Temperatura Futura
    elif "proyección" in pregunta_lower or "futuro" in pregunta_lower or "pronóstico" in pregunta_lower:
        if "Temperatura (°C)" in df_ref.columns and "Fecha" in df_ref.columns:
            forecast_df = predecir_temperatura_futura(df_ref, horizon=7)
            avg_forecast = forecast_df["Temperatura Predicha (°C)"].mean()
            return base_message + f"se proyecta que la temperatura promedio en los próximos 7 días será de aproximadamente {avg_forecast:.2f} °C."
        else:
            return base_message + "No se dispone de las columnas necesarias para la proyección."
    else:
        return (
            "Lo siento, no reconozco esa pregunta. Prueba con:\n"
            "- ¿Cuántos servidores están en estado crítico?\n"
            "- ¿Cuántos registros tiene el dataset?\n"
            "- ¿Cuál es la temperatura promedio de los servidores?\n"
            "- ¿Qué variables explican mejor el estado?\n"
            "- ¿Qué proyección hay para la temperatura futura?"
        )

# -----------------------------------------------------------------------------
# Interfaz Streamlit: Definición de pestañas principales y subpestañas
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
    st.write("Este ChatBot responde en base al modelo con mayor precisión y puede informar sobre la importancia de variables y proyecciones futuras.")
    st.markdown(
        """
        **Ejemplos de preguntas:**
        - ¿Cuántos servidores están en estado crítico?
        - ¿Cuántos registros tiene el dataset?
        - ¿Cuál es la temperatura promedio de los servidores?
        - ¿Qué variables explican mejor el estado?
        - ¿Qué proyección hay para la temperatura futura?
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
                on_click=upload_to_s3,  # Se usa la función definida al inicio
                args=(csv_data, archivo_salida)
            )

# ------------------- Pestaña: Tablero Interactivo ---------------------------
with tab_dashboard:
    st.subheader("📊 Tablero Interactivo")
    # Se crean dos subpestañas para separar visualizaciones basadas en:
    # A) el dataset original (Monitoreo del Estado Real) y
    # B) el dataset ganador (Resultados de la Clasificación)
    sub_tab1, sub_tab2 = st.tabs(["Monitoreo del Estado Real", "Resultados de la Clasificación"])

    # -------- Subpestaña A: Monitoreo del Estado Real --------
    with sub_tab1:
        st.markdown("### Monitoreo del Estado Real")
        if "original_dataset" not in st.session_state:
            st.warning("Por favor, procesa los modelos para cargar el dataset original.")
        else:
            df_original = st.session_state["original_dataset"]
            # KPIs: Conteo de servidores por estado
            if "Estado del Sistema" in df_original.columns:
                total_counts = df_original["Estado del Sistema"].value_counts().reset_index()
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

                # Distribución de Estados (Pie Chart)
                st.markdown("#### Distribución de Estados")
                fig_pie = px.pie(total_counts, values="Cantidad", names="Estado", title="Distribución de Estados (Real)")
                st.plotly_chart(fig_pie, use_container_width=True)

                # Evolución de Estados en el Tiempo (Line Chart)
                if "Fecha" in df_original.columns:
                    df_time = df_original.groupby(["Fecha","Estado del Sistema"]).size().reset_index(name="Conteo")
                    fig_line = px.line(df_time, x="Fecha", y="Conteo", color="Estado del Sistema", title="Evolución de Estados en el Tiempo (Real)")
                    st.plotly_chart(fig_line, use_container_width=True)

                # Relación entre Uso CPU y Temperatura (Scatter Plot)
                if "Uso CPU (%)" in df_original.columns and "Temperatura (°C)" in df_original.columns:
                    st.markdown("#### Relación entre Uso de CPU y Temperatura")
                    fig_scatter = px.scatter(df_original, x="Uso CPU (%)", y="Temperatura (°C)",
                                             color="Estado del Sistema",
                                             title="Uso CPU vs Temperatura (Real)")
                    st.plotly_chart(fig_scatter, use_container_width=True)

                # Boxplot para Outliers
                st.markdown("#### Análisis de Outliers")
                fig_box, ax = plt.subplots(figsize=(8, 5))
                df_box = df_original[["Uso CPU (%)", "Temperatura (°C)"]].dropna()
                sns.boxplot(data=df_box, ax=ax)
                ax.set_title("Boxplot de Uso CPU y Temperatura (Real)")
                st.pyplot(fig_box)

                # Matriz de Correlación
                st.markdown("#### Matriz de Correlación")
                numeric_cols = df_original.select_dtypes(include=[np.number])
                if not numeric_cols.empty:
                    corr_matrix = numeric_cols.corr()
                    fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Matriz de Correlación (Real)")
                    st.plotly_chart(fig_corr, use_container_width=True)
    
    # -------- Subpestaña B: Resultados de la Clasificación --------
    with sub_tab2:
        st.markdown("### Resultados de la Clasificación (Dataset Ganador)")
        if "best_dataset" not in st.session_state:
            st.warning("No se ha definido un dataset ganador. Procesa los modelos primero.")
        else:
            df_ganador = st.session_state["best_dataset"]
            # Distribución de Estados (Pie Chart)
            if "Estado del Sistema" in df_ganador.columns:
                total_counts_ganador = df_ganador["Estado del Sistema"].value_counts().reset_index()
                total_counts_ganador.columns = ["Estado", "Cantidad"]
                fig_pie_ganador = px.pie(total_counts_ganador, values="Cantidad", names="Estado",
                                         title="Distribución de Estados (Clasificación)")
                st.plotly_chart(fig_pie_ganador, use_container_width=True)

            # KPIs: Conteo de servidores por estado (Clasificación)
            col1g, col2g, col3g, col4g = st.columns(4)
            count_c_critico = total_counts_ganador.loc[total_counts_ganador["Estado"]=="Crítico", "Cantidad"].values[0] if "Crítico" in total_counts_ganador["Estado"].values else 0
            count_c_advertencia = total_counts_ganador.loc[total_counts_ganador["Estado"]=="Advertencia", "Cantidad"].values[0] if "Advertencia" in total_counts_ganador["Estado"].values else 0
            count_c_normal = total_counts_ganador.loc[total_counts_ganador["Estado"]=="Normal", "Cantidad"].values[0] if "Normal" in total_counts_ganador["Estado"].values else 0
            count_c_inactivo = total_counts_ganador.loc[total_counts_ganador["Estado"]=="Inactivo", "Cantidad"].values[0] if "Inactivo" in total_counts_ganador["Estado"].values else 0

            col1g.metric("Crítico (Clasif.)", f"{count_c_critico}")
            col2g.metric("Advertencia (Clasif.)", f"{count_c_advertencia}")
            col3g.metric("Normal (Clasif.)", f"{count_c_normal}")
            col4g.metric("Inactivo (Clasif.)", f"{count_c_inactivo}")
            
            # Visualización de Importancia de Variables
            st.markdown("#### Importancia de Variables del Modelo Ganador")
            if "trained_models" in st.session_state and st.session_state["best_model"] in st.session_state["trained_models"]:
                model = st.session_state["trained_models"][st.session_state["best_model"]]
                if hasattr(model, "feature_importances_"):
                    importances = model.feature_importances_
                    features = st.session_state.get("feature_names", [])
                    if features:
                        df_imp = pd.DataFrame({"Variable": features, "Importancia": importances})
                        df_imp = df_imp.sort_values("Importancia", ascending=True)
                        fig_bar = px.bar(df_imp, x="Importancia", y="Variable", orientation="h",
                                         title="Importancia de Variables")
                        st.plotly_chart(fig_bar, use_container_width=True)
                    else:
                        st.info("No se encontraron nombres de variables.")
                else:
                    st.info("El modelo ganador no proporciona importancias de variables.")
            else:
                st.info("No se ha almacenado el modelo ganador.")

            # Pronóstico de Temperatura Futura
            st.markdown("#### Pronóstico de Temperatura Futura")
            if "Temperatura (°C)" in df_ganador.columns and "Fecha" in df_ganador.columns:
                forecast_df = predecir_temperatura_futura(df_ganador, horizon=7)
                fig_forecast = px.line(forecast_df, x="Fecha", y="Temperatura Predicha (°C)",
                                       title="Pronóstico de Temperatura Futura (Próximos 7 días)")
                st.plotly_chart(fig_forecast, use_container_width=True)
            else:
                st.info("No se dispone de las columnas necesarias para el pronóstico.")
