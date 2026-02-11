import pickle
import streamlit as st
import pandas as pd
import os


st.set_page_config(page_title="Recomendador de Carreras", layout="centered")

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'modelo_pipeline_final.pkl')
csv_path = os.path.join(current_dir, 'adult-census-income.csv') # Cambia el nombre al de tu CSV

@st.cache_resource
def load_resources():
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    df = pd.read_csv(csv_path)
    return model, df
try:
    model, df = load_resources()
    st.success("✅ Sistema listo para procesar")
except Exception as e:
    st.error(f"Error al cargar recursos: {e}")
    st.stop()

    
st.title("🚀 Validador de Éxito Profesional")
st.write("Introduce el ID del usuario para calcular su probabilidad de éxito estadístico.")

user_id = st.number_input("ID del Usuario", min_value=0, step=1, value=0)

if st.button("Generar Recomendación"):
    
    try:
        user_row = df.iloc[[user_id]]
    except IndexError:
        st.error("Ese ID de usuario no existe en la base de datos.")
    if user_row.empty:
        st.warning("⚠️ ID de usuario no encontrado en la base de datos.")
    else:
        X_usuario = user_row.drop(columns=['income', 'income_num', 'pure_labor_rating'], errors='ignore')
        
        
        prob = model.predict_proba(X_usuario)[0][1]
        
        
        st.divider()
        st.subheader(f"Análisis para el Usuario {user_id}")
        
        col1, col2 = st.columns(2)
        col1.metric("Probabilidad de Éxito", f"{prob:.2%}")
        
        if prob > 0.70:
            col2.success("Perfil de Alto Potencial")
        else:
            col2.info("Perfil con Potencial Estándar")
            
        
        with st.expander("Ver detalles del perfil"):
            st.write(user_row)