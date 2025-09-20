import streamlit as st
import tensorflow as tf
import numpy as np
from openai import OpenAI
from sklearn.ensemble import RandomForestClassifier
import joblib

# Configuración de la página
st.set_page_config(
    page_title="Diagnóstico de Diabetes",
    layout="wide"
)

# Carga el modelo de Keras previamente entrenado
@st.cache_resource
def load_model_keras():
    """Carga el modelo de Keras en caché para evitar recargas."""
    try:
        model_keras = tf.keras.models.load_model('modelo_diabetes.keras')
        return model_keras
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}. Asegúrate de que 'modelo_diabetes.keras' esté en el mismo directorio.")
        return None

# Carga el modelo de Random Forest previamente entrenado
@st.cache_resource
def load_model_rf():
    """Carga el modelo de Random Forest en caché para evitar recargas."""
    try:
        # Cargar Random Forest
        model_rf = joblib.load('random_forest_model.pkl')
        return model_rf
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}. Asegúrate de que 'modelo_diabetes.keras' esté en el mismo directorio.")
        return None


model_keras = load_model_keras()
model_rf = load_model_rf()

# Título y descripción de la aplicación
st.title("Diagnóstico de Diabetes basado en IA")
st.markdown("Este aplicativo usa una red neuronal para predecir si una persona tiene diabetes y luego genera recomendaciones personalizadas con la ayuda de un modelo de lenguaje avanzado.")
st.markdown("---")

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

    # ---
    ## radiobutton en la barra lateral
    # El contenedor 'st.sidebar' ubica los elementos en el costado izquierdo de la página.
    with st.sidebar:
        st.header("Personalizaciones")  
        
        st.markdown("---")

        selected_model = st.radio(
            "Modelo predictor:",
            ("Red Neuronal Simple", "Ramdon Forest")
        )

        st.markdown("---")

        #st.header("Tipo de lenguaje")        
        selected_language = st.radio(
            "Tipo de lenguaje:",
            ("Coloquial", "Formal")
        )
        
        st.markdown("---")

        #st.header("Localización del paciente")        
        selected_location = st.radio(
            "Localización del paciente:",
            ("Quito", "Santo Domingo", "Las Pampas")
        )


    # **Formulario principal de la aplicación**
    # El formulario agrupa los inputs y el botón, haciendo que se procesen juntos al enviar.
    with st.form(key="diabetes_form"):
        st.header("📝 Parámetros del Paciente")
        st.markdown("Ingrese los datos del paciente para realizar el diagnóstico.")
        
        # Uso de columnas para organizar los inputs
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            pregnancies = st.slider('Número de Embarazos', 0, 17, 3, key='p')
            diastolic = st.slider('Presión Sanguínea Diastólica (mmHg)', 0, 122, 72, key='d')

        with col2:
            glucose = st.slider('Nivel de Glucosa (mg/dL)', 0, 200, 117, key='g')
            triceps = st.slider('Grosor del Pliegue Cutáneo del Tríceps (mm)', 0, 99, 23, key='t')

        with col3:
            insulin = st.slider('Nivel de Insulina (mu U/ml)', 0, 846, 30, key='i')
            bmi = st.slider('Índice de Masa Corporal (BMI)', 0.0, 67.1, 32.0, key='b')
        
        with col4:
            dpf = st.slider('Función de Pedigree de Diabetes', 0.078, 2.42, 0.3725, key='dpf')
            age = st.slider('Edad', 21, 88, 29, key='a')
        
        # **Botón para enviar el formulario**
        submit_button = st.form_submit_button(label='🚀 Realizar Predicción y Obtener Recomendaciones')

    is_diabetic = False
    diagnosis_proba = 0.00
    diagnosis_text = ""

    # Lógica principal del diagnóstico que se activa con el botón del formulario
    if submit_button:

        # Diccionario con los datos del formulario
        input_data = {
            'Número de Embarazos': pregnancies,
            'Nivel de Glucosa': glucose,
            'Presión Sanguínea Diastólica': diastolic,
            'Grosor del Pliegue Cutáneo': triceps,
            'Nivel de Insulina': insulin,
            'BMI': bmi,
            'Función de Pedigree': dpf,
            'Edad': age
        }
        # Convierte los datos del diccionario a un array de numpy
        input_features = np.array(list(input_data.values())).reshape(1, -1)

        if selected_model == "Red Neuronal Simple":
            if model_keras:
                try:                    
                    # Normalización de los datos de entrada (AJUSTAR ESTO SEGÚN TU ENTRENAMIENTO)
                    normalized_features = input_features / np.max(input_features) 
                    
                    # Predicción con el modelo
                    prediction = model_keras.predict(normalized_features)
                    diagnosis_proba = prediction[0][0]

                    # Clasificación
                    if diagnosis_proba > 0.5:
                        is_diabetic = True
                        diagnosis_text = f" con una probabilidad del **{diagnosis_proba * 100:.2f}%**."
                    else:
                        is_diabetic = False
                        diagnosis_text = f" con una probabilidad del **{(1 - diagnosis_proba) * 100:.2f}%**."
                
                except Exception as e:
                    st.error(f"Ocurrió un error inesperado durante el procesamiento dela Red Neuronal Simple: {e}")
            else:
                st.warning("No se pudo cargar el modelo Red Neurnal Simple, no es posible realizar el diagnóstico.")

        elif "Ramdon Forest":
            if model_rf:
                try:
                    # Predicción con el modelo
                    prediction = model_rf.predict(input_features)
                    diagnosis_proba = prediction
                    diagnosis_text = "."
                    # Clasificación
                    if diagnosis_proba > 0.5:
                        is_diabetic = True
                    else:
                        is_diabetic = False
                except Exception as e:
                    st.error(f"Ocurrió un error inesperado durante el procesamiento de Randm Forest: {e}")
            else:
                st.warning("No se pudo cargar el modelo Random Forest, no es posible realizar el diagnóstico.")

        # **Sección de Resultados**
        st.markdown("## 📊 Resultados del Análisis")
        st.markdown(f"**Análisis completado.**")

        step = "0"
        # Desplegar resultado
        try:
            step = "1"
            if is_diabetic:
                diagnosis = "Diabético"
                #st.error(f"El sistema clasifica al paciente como **{diagnosis}** con una probabilidad del **{diagnosis_proba * 100:.2f}%**.")
                st.error(f"El sistema clasifica al paciente como **{diagnosis}**{diagnosis_text}")
            else:
                diagnosis = "No Diabético"
                #st.success(f"El sistema clasifica al paciente como **{diagnosis}** con una probabilidad del **{(1 - diagnosis_proba) * 100:.2f}%**.")
                st.success(f"El sistema clasifica al paciente como **{diagnosis}**{diagnosis_text}")

            step = "2"
            # Formato de texto para el prompt
            data_string = "\n".join([f"{key}: {value}" for key, value in input_data.items()])
            
            step = "3"
            # **Sección de Recomendaciones**
            st.markdown("---")
            st.markdown(f"## 📋 Recomendaciones Personalizadas - Lenguaje {selected_language}")
            
            step = "4"
            # Lógica para la integración con GPT-4
            if is_diabetic:
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
            
            location_text = "Quito, capital del Ecuador"
            location_describe = ""
            
            if selected_location == "Quito":
                location_text = "Quito, capital del Ecuador"
                location_describe = ""
            elif selected_location == "Santo Domingo":
                location_text = "Santo Domingo de los Colorados, Ecuador"
                location_describe = ""
            elif selected_location == "Las Pampas":
                location_text = "Las Pampas, parroquia rural del canton Sigchos, provincia de Cotopaxi, Ecuador"
                location_describe = ""

            step = "5"
            st.markdown("---")
            #st.header("Recomendaciones de GPT-4.1")
            step = "6"
            with st.spinner('Generando recomendaciones...'):
                try:
                    step = "7"
                    response = client.chat.completions.create(
                        model="gpt-4.1",
                        messages=[{"role": "system", "content": f"Eres un experto en nutrición y fitness. Toma en cuenta que el usuario vive en {location_text} para generar tus recomendaciones. Usa un lenguaje {selected_language} en tus respuestas y nombrar siempre el lugar donde vive el usuario."},
                                {"role": "user", "content": prompt_text}],
                        temperature=0.7
                    )
                    step = "8"
                    recommendations = response.choices[0].message.content
                    st.markdown(recommendations)
                except Exception as e:
                        st.error(f"Error al conectar con la API de GPT-4.1: {e}. Asegúrate de que tu clave de API esté configurada correctamente.")
                step = "9"
        except Exception as e:
            st.error(f"Ocurrió un error inesperado durante el procesamiento (paso {step}): {e}")
        




# para correr en la terminal		
# streamlit run streamlit_app.py
