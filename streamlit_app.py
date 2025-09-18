import streamlit as st
import tensorflow as tf
import numpy as np
import openai
import json
import os

# Configura tu clave de API de OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Carga el modelo de Keras previamente entrenado
@st.cache_resource
def load_model():
    """Carga el modelo de Keras en caché para evitar recargas."""
    try:
        model = tf.keras.models.load_model('modelo_diabetes.keras')
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}. Asegúrate de que 'modelo_diabetes.keras' esté en el mismo directorio.")
        return None

model = load_model()

# Título y descripción de la aplicación
st.title("Diagnóstico de Diabetes basado en IA")
st.markdown("Este aplicativo usa una red neuronal para predecir si una persona tiene diabetes y luego genera recomendaciones personalizadas con la ayuda de un modelo de lenguaje avanzado.")

# Interfaz para la entrada de datos
st.sidebar.header("Parámetros del Paciente")

def user_input_features():
    """Recopila los parámetros de entrada del usuario."""
    pregnancies = st.sidebar.slider('Embarazos', 0, 17, 3)
    glucose = st.sidebar.slider('Glucosa', 0, 200, 117)
    diastolic = st.sidebar.slider('Presión sanguínea diastólica', 0, 122, 72)
    triceps = st.sidebar.slider('Grosor del pliegue cutáneo del tríceps', 0, 99, 23)
    insulin = st.sidebar.slider('Insulina', 0, 846, 30)
    bmi = st.sidebar.slider('Índice de Masa Corporal (BMI)', 0.0, 67.1, 32.0)
    dpf = st.sidebar.slider('Función de Pedigree de Diabetes', 0.078, 2.42, 0.3725)
    age = st.sidebar.slider('Edad', 21, 88, 29)
    data = {
        'pregnancies': pregnancies,
        'glucose': glucose,
        'diastolic': diastolic,
        'triceps': triceps,
        'insulin': insulin,
        'bmi': bmi,
        'dpf': dpf,
        'age': age
    }
    features = np.array(list(data.values())).reshape(1, -1)
    return features, data

# Botón para el diagnóstico
if st.sidebar.button('Realizar Diagnóstico'):
    if model:
        try:
            input_features, input_data = user_input_features()
            
            # Normalización de los datos de entrada (AJUSTAR ESTO SEGÚN TU ENTRENAMIENTO)
            # Ejemplo de normalización simple, debes usar los mismos escaladores de tu modelo
            normalized_features = input_features / np.max(input_features) 
            
            # Predicción con el modelo
            prediction = model.predict(normalized_features)
            diagnosis_proba = prediction[0][0]
            
            # Clasificación
            if diagnosis_proba > 0.5:
                diagnosis = "Diabético"
                diagnosis_text = f"Con una probabilidad del **{diagnosis_proba * 100:.2f}%**"
            else:
                diagnosis = "No Diabético"
                diagnosis_text = f"Con una probabilidad del **{(1 - diagnosis_proba) * 100:.2f}%**"
            
            st.markdown("---")
            st.header("Resultados del Diagnóstico")
            st.markdown(f"El sistema ha clasificado a la persona como: **{diagnosis}**")
            st.markdown(diagnosis_text)
            
            # Lógica para la integración con GPT-4
            if diagnosis == "Diabético":
                prompt_text = f"""
                El sistema de IA ha clasificado a un paciente como diabético. Sus parámetros son:
                {json.dumps(input_data, indent=2)}.
                Como experto en nutrición y fitness, por favor, genera una guía práctica de alimentación y un plan de ejercicios personalizado, 
                enfocados en el manejo de la diabetes para una persona con estas características. 
                Considera una dieta baja en carbohidratos simples y azúcares, alta en fibra y proteínas, 
                y un plan de ejercicios moderado que incluya cardio y fuerza.
                """
            else:
                prompt_text = f"""
                El sistema de IA ha clasificado a un paciente como no diabético, pero con los siguientes parámetros:
                {json.dumps(input_data, indent=2)}.
                Como experto en nutrición y fitness, por favor, genera recomendaciones preventivas de alimentación y un plan de ejercicios generales para 
                mantener una vida saludable y prevenir la diabetes en el futuro. Enfócate en una dieta balanceada y actividad física regular.
                """
            
            st.markdown("---")
            st.header("Recomendaciones de GPT-4")
            
            with st.spinner('Generando recomendaciones...'):
                try:
                    response = openai.chat.completions.create(
                        model="gpt-4",
                        messages=[{"role": "system", "content": "Eres un experto en nutrición y fitness."},
                                  {"role": "user", "content": prompt_text}],
                        max_tokens=500
                    )
                    recommendations = response.choices[0].message.content
                    st.markdown(recommendations)
                except Exception as e:
                    st.error(f"Error al conectar con la API de GPT-4: {e}. Asegúrate de que tu clave de API esté configurada correctamente.")
            
        except Exception as e:
            st.error(f"Ocurrió un error inesperado durante el procesamiento: {e}")
    else:
        st.warning("No se pudo cargar el modelo, no es posible realizar el diagnóstico.")

# para correr en la terminal		
# streamlit run streamlit_app.py
