import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

# 1. CONFIGURACIÓN DE APIS Y MODELOS
# Reemplaza con tu clave real de Google AI Studio
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
    model_gemini = genai.GenerativeModel('gemini-1.5-flash')
else:
    st.error("⚠️ No se encontró la clave API en el archivo .env")

@st.cache_resource # Evita que el modelo se recargue en cada interacción
def cargar_recursos_ia():
    # Modelo para transformar texto en vectores (Embeddings)
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

embedding_model = cargar_recursos_ia()

RUTA_JSON = os.path.join(os.path.dirname(__file__), "ley_procesada.json")

# 2. CARGA DE DATOS
@st.cache_data
def cargar_datos_legales():
    if not os.path.exists(RUTA_JSON):
        st.error(f"No se encontró el archivo en: {RUTA_JSON}")
        return None
    try:
        df = pd.read_json(RUTA_JSON)
        # Convertimos las listas a vectores de Numpy para la IA
        df['Vector'] = df['Vector'].apply(lambda x: np.array(x))
        return df
    except Exception as e:
        st.error(f"Error al leer el archivo: {e}")
        return None
df_ley = cargar_datos_legales()

# 3. FUNCIONES DE LÓGICA
def buscar_asesoria(pregunta_usuario, df, top_n=3):
    # Convertimos la pregunta en vector
    vector_pregunta = embedding_model.encode([pregunta_usuario])
    # Extraemos la matriz de vectores del DataFrame
    matriz_vectores = np.stack(df['Vector'].values)
    # Calculamos similitud
    similitudes = cosine_similarity(vector_pregunta, matriz_vectores)
    df_temp = df.copy()
    df_temp['Similitud'] = similitudes[0]
    return df_temp.nlargest(top_n, 'Similitud')

def simplificar_ley(texto_legal):
    prompt = f"""
    Eres un asesor legal experto de extranjería en España. 
    Tu objetivo es ayudar a una persona migrante a entender este artículo de la ley.
    Explícalo en 2 frases cortas, usando un lenguaje muy sencillo y cercano.
    No uses tecnicismos.
    
    Artículo: {texto_legal}
    """
    try:
        respuesta = model_gemini.generate_content(prompt)
        return respuesta.text
    except Exception as e:
        return "Lo siento, no pude simplificar este texto en este momento."

# 4. INTERFAZ DE USUARIO (STREAMLIT)
st.set_page_config(page_title="Asistente Legal Migrante", layout="centered", page_icon="⚖️")

st.title("🛡️ Guía Legal para Migrantes en España")
st.markdown("Busca tus derechos en la **Ley de Extranjería** con lenguaje natural.")

if df_ley is not None:
    pregunta = st.text_input("¿En qué podemos ayudarte hoy?", placeholder="Ej: ¿Cómo renovar mi permiso de trabajo?")

    if pregunta:
        with st.spinner('Consultando la base legal...'):
            resultados = buscar_asesoria(pregunta, df_ley)
            
            st.subheader("Resultados más relevantes:")
            
            for index, row in resultados.iterrows():
                # Mostramos la referencia y el nivel de coincidencia
                with st.expander(f"📍 {row['Referencia']} (Similitud: {int(row['Similitud']*100)}%)"):
                    st.write(row['Texto_Legal'])
                    
                    # El botón de simplificar ahora es único para cada artículo
                    if st.button("✨ Simplificar explicación", key=f"btn_{index}"):
                        resumen = simplificar_ley(row['Texto_Legal'])
                        st.success(f"**En pocas palabras:** {resumen}")

st.sidebar.info("Proyecto de Análisis de Datos para Impacto Social.")