import streamlit as st
import tensorflow as tf
import numpy as np
from openai import OpenAI

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

def get_user_input():
    """Recopila los parámetros de entrada del usuario."""
    
    # Interfaz para la entrada de datos
    st.header("Parámetros del Paciente")
    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.slider('Embarazos', 0, 17, 3)
        glucose = st.slider('Glucosa', 0, 200, 117)
        diastolic = st.slider('Presión sanguínea diastólica', 0, 122, 72)
        triceps = st.slider('Grosor del pliegue cutáneo del tríceps', 0, 99, 23)
    with col2:
        insulin = st.slider('Insulina', 0, 846, 30)
        bmi = st.slider('Índice de Masa Corporal (BMI)', 0.0, 67.1, 32.0)
        dpf = st.slider('Función de Pedigree de Diabetes', 0.078, 2.42, 0.3725)
        age = st.slider('Edad', 21, 88, 29)
    
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
    return data



# Configura tu clave de API de OpenAI
#openai.api_key = os.getenv("OPENAI_API_KEY")

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:
    # Create an OpenAI client.
    client = OpenAI(api_key=openai_api_key)


    # Create a session state variable to store the chat messages. This ensures that the
    # messages persist across reruns.
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display the existing chat messages via `st.chat_message`.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    # Obtiene los datos del usuario antes de la lógica del botón
    input_data = get_user_input()

    # **<-- EL BOTÓN AHORA ACTIVA LA PREDICCIÓN CON LOS DATOS YA OBTENIDOS**
    if st.button('Predecir'):
        if model:
            try:
                # Convierte los datos del diccionario a un array de numpy
                input_features = np.array(list(input_data.values())).reshape(1, -1)
                
                # Normalización de los datos de entrada (AJUSTAR ESTO SEGÚN TU ENTRENAMIENTO)
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

                # Formato de texto plano
                data_string = "\n".join([f"{key}: {value}" for key, value in input_data.items()])
                
                # Lógica para la integración con GPT-4
                if diagnosis == "Diabético":
                    prompt_text = f"""
                    El sistema de IA ha clasificado a un paciente como diabético. Sus parámetros son:
                    {data_string}.
                    Como experto en nutrición y fitness, por favor, genera una guía práctica de alimentación y un plan de ejercicios personalizado, 
                    enfocados en el manejo de la diabetes para una persona con estas características. 
                    Considera una dieta baja en carbohidratos simples y azúcares, alta en fibra y proteínas, 
                    y un plan de ejercicios moderado que incluya cardio y fuerza.
                    """
                else:
                    prompt_text = f"""
                    El sistema de IA ha clasificado a un paciente como no diabético, pero con los siguientes parámetros:
                    {data_string}.
                    Como experto en nutrición y fitness, por favor, genera recomendaciones preventivas de alimentación y un plan de ejercicios generales para 
                    mantener una vida saludable y prevenir la diabetes en el futuro. Enfócate en una dieta balanceada y actividad física regular.
                    """
                
                st.markdown("---")
                st.header("Recomendaciones de GPT-4")
                
                with st.spinner('Generando recomendaciones...'):
                    try:
                        response = client.chat.completions.create(
                            model="gpt-4.1",
                            messages=[{"role": "system", "content": "Eres un experto en nutrición y fitness."},
                                    {"role": "user", "content": prompt_text}],
                            #max_tokens=500
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
