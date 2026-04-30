
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import re

# ============================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================
st.set_page_config(
    page_title="Coffee Analytics Dashboard",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CARGAR MODELOS Y PREPROCESADORES
# ============================================
@st.cache_resource
def load_all_models():
    """Carga todos los modelos y preprocesadores necesarios"""
    
    models = {}
    
    # ============================================
    # 1. PRIMERO: Cargar el dataset
    # ============================================
    try:
        # Cargar dataset
        models['df_long'] = pd.read_parquet('coffee_db.parquet')
        
        # Procesar el dataset para obtener formato largo
        year_cols = [col for col in models['df_long'].columns if '/' in col]
        id_vars = ['Country', 'Coffee type', 'Total_domestic_consumption']
        
        df_long = pd.melt(models['df_long'], id_vars=id_vars, value_vars=year_cols,
                         var_name='Year_Period', value_name='Consumption')
        df_long['Consumption'] = pd.to_numeric(df_long['Consumption'], errors='coerce')
        df_long['Year'] = df_long['Year_Period'].apply(lambda x: int(x.split('/')[0]))
        df_long = df_long.drop(columns=['Year_Period'])
        df_long = df_long[['Country', 'Coffee type', 'Year', 'Consumption', 'Total_domestic_consumption']]
        
        models['df_long'] = df_long.dropna()
        st.write("✓ Dataset cargado")
    except Exception as e:
        st.error(f"Error cargando dataset: {e}")
        models['df_long'] = None
    
    # Cargar países únicos
    if models['df_long'] is not None:
        models['unique_countries'] = models['df_long']['Country'].unique().tolist()
        models['unique_coffee_types'] = models['df_long']['Coffee type'].unique().tolist()
    else:
        models['unique_countries'] = []
        models['unique_coffee_types'] = []
    
    # ============================================
    # 2. SEGUNDO: Cargar modelos de TensorFlow
    # ============================================
    try:
        # Cargar modelo de clasificación
        models['classification_model'] = load_model('classification_model.h5')
        st.write("✓ Modelo de clasificación cargado")
    except Exception as e:
        st.error(f"Error cargando modelo de clasificación: {e}")
        models['classification_model'] = None
    
    try:
        # Cargar modelo de regresión
        models['regression_model'] = load_model('regression_model.h5')
        st.write("✓ Modelo de regresión cargado")
    except Exception as e:
        st.error(f"Error cargando modelo de regresión: {e}")
        models['regression_model'] = None
    
    # ============================================
    # 3. TERCERO: Cargar clases de café
    # ============================================
    try:
        # Cargar clases de café
        models['coffee_type_classes'] = np.load('coffee_type_classes.npy', allow_pickle=True)
        st.write("✓ Clases de café cargadas")
    except Exception as e:
        st.error(f"Error cargando clases de café: {e}")
        models['coffee_type_classes'] = None
    
    # ============================================
    # 4. CUARTO: Crear preprocesadores (después de df_long)
    # ============================================
    try:
        # Cargar preprocesador de clasificación
        # NOTA: Por incompatibilidad de versiones de sklearn, recreamos el preprocesador
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.compose import ColumnTransformer
        
        # Recrear preprocesador de clasificación
        models['preprocessor_cls'] = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), ['Country'])
            ],
            remainder='passthrough'
        )
        # Ajustar con los datos disponibles
        X_cls = models['df_long'][['Country', 'Year']].dropna()
        models['preprocessor_cls'].fit(X_cls)
        st.write("✓ Preprocesador de clasificación creado")
    except Exception as e:
        st.error(f"Error creando preprocesador de clasificación: {e}")
        models['preprocessor_cls'] = None
    
    try:
        # Cargar preprocesador de regresión
        # NOTA: Por incompatibilidad de versiones de sklearn, recreamos el preprocesador
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.compose import ColumnTransformer
        
        # Recrear preprocesador de regresión
        models['preprocessor_reg'] = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), ['Country', 'Coffee type']),
                ('num', StandardScaler(), ['Year'])
            ],
            remainder='drop'
        )
        # Ajustar con los datos disponibles
        X_reg = models['df_long'][['Country', 'Coffee type', 'Year']].dropna()
        models['preprocessor_reg'].fit(X_reg)
        st.write("✓ Preprocesador de regresión creado")
    except Exception as e:
        st.error(f"Error creando preprocesador de regresión: {e}")
        models['preprocessor_reg'] = None
    
    try:
        # Cargar scaler de regresión
        # NOTA: Por incompatibilidad de versiones de sklearn, recreamos el scaler
        from sklearn.preprocessing import StandardScaler
        
        # Recrear scaler de regresión
        models['scaler_y_reg'] = StandardScaler()
        y_reg = models['df_long']['Consumption'].dropna().values.reshape(-1, 1)
        models['scaler_y_reg'].fit(y_reg)
        st.write("✓ Scaler de regresión creado")
    except Exception as e:
        st.error(f"Error creando scaler de regresión: {e}")
        models['scaler_y_reg'] = None
    
    return models

# ============================================
# FUNCIONES AUXILIARES
# ============================================

def extract_entities_and_intent(query, countries, coffee_types):
    """Extrae entidades y intención de una consulta"""
    intent = None
    country = None
    coffee_type = None
    year = None
    
    query_lower = query.lower()
    
    # Extraer año
    year_matches = re.findall(r'\b\d{4}\b', query_lower)
    if year_matches:
        year = int(max(year_matches))
    
    # Extraer país
    for c in countries:
        if c.lower() in query_lower:
            country = c
            break
    
    # Extraer tipo de café
    for ct in coffee_types:
        if ct.lower() in query_lower:
            coffee_type = ct
            break
    
    # Inferir intención
    prediction_keywords = ['predecir', 'estimar', 'pronosticar', 'futuro', 'consumirá']
    if any(keyword in query_lower for keyword in prediction_keywords):
        intent = 'predict_consumption'
    elif (country or coffee_type or year) and ('consumo' in query_lower or 'cantidad' in query_lower or 'fue' in query_lower):
        intent = 'historical_consumption'
    elif (country or coffee_type or year) and not intent:
        intent = 'predict_consumption'
    else:
        intent = 'general_info'
    
    return {
        'intent': intent,
        'country': country,
        'coffee_type': coffee_type,
        'year': year
    }

def preprocess_new_data_for_reg(country, coffee_type, year, preprocessor_reg):
    """Preprocesa nuevos datos para el modelo de regresión"""
    new_data = pd.DataFrame({
        'Country': [country],
        'Coffee type': [coffee_type],
        'Year': [year]
    })
    # Convertir Year a float para el scaler
    new_data['Year'] = new_data['Year'].astype(float)
    return preprocessor_reg.transform(new_data)

def query_ml_models(extracted_info, models):
    """Procesa consultas usando los modelos ML"""
    intent = extracted_info['intent']
    country = extracted_info['country']
    coffee_type = extracted_info['coffee_type']
    year = extracted_info['year']
    
    response = "Lo siento, no pude procesar su consulta. Asegúrese de proporcionar un país y un año."
    
    if intent == 'predict_consumption':
        if country and year:
            if coffee_type:
                # Predicción específica
                new_input = preprocess_new_data_for_reg(country, coffee_type, year, models['preprocessor_reg'])
                predicted_consumption_scaled = models['regression_model'].predict(new_input, verbose=0)[0][0]
                predicted_consumption = models['scaler_y_reg'].inverse_transform([[predicted_consumption_scaled]])[0][0]
                response = f"Se predice que el consumo de **{coffee_type}** en **{country}** para el año **{year}** será de aproximadamente **{predicted_consumption:,.2f}** unidades."
            else:
                # Predicción total
                total_consumption = 0
                for c_type in models['unique_coffee_types']:
                    new_input = preprocess_new_data_for_reg(country, c_type, year, models['preprocessor_reg'])
                    predicted_consumption_scaled = models['regression_model'].predict(new_input, verbose=0)[0][0]
                    predicted_consumption = models['scaler_y_reg'].inverse_transform([[predicted_consumption_scaled]])[0][0]
                    total_consumption += predicted_consumption
                response = f"Se predice que el consumo general de café en **{country}** para el año **{year}** será de aproximadamente **{total_consumption:,.2f}** unidades."
        else:
            response = "Para predecir el consumo, necesito al menos un país y un año."
    
    elif intent == 'historical_consumption':
        filtered_df = models['df_long'].copy()
        if country: 
            filtered_df = filtered_df[filtered_df['Country'].str.lower() == country.lower()]
        if coffee_type: 
            filtered_df = filtered_df[filtered_df['Coffee type'].str.lower() == coffee_type.lower()]
        if year: 
            filtered_df = filtered_df[filtered_df['Year'] == year]
        
        if not filtered_df.empty:
            total_historical = filtered_df['Consumption'].sum()
            details = []
            if country: details.append(country)
            if coffee_type: details.append(coffee_type)
            if year: details.append(str(year))
            details_str = " en ".join(details)
            response = f"El consumo histórico de café {details_str} fue de **{total_historical:,.2f}** unidades."
        else:
            response = "No se encontraron datos históricos para los criterios proporcionados."
    
    return response

# ============================================
# PÁGINA PRINCIPAL
# ============================================

# Título principal
st.title("☕ Coffee Analytics Dashboard")
st.markdown("### Sistema de Predicción y Análisis de Consumo de Café")

# Sidebar para navegación
st.sidebar.title("Navegación")
page = st.sidebar.radio(
    "Seleccionar sección:",
    ["🏠 Inicio", "📊 Predicción de Consumo", "🏷️ Clasificación de Café", "💬 Consultas RAG"]
)

# Cargar modelos al inicio
if 'models' not in st.session_state:
    with st.spinner('Cargando modelos...'):
        st.session_state.models = load_all_models()

models = st.session_state.models

# ============================================
# PÁGINA: INICIO
# ============================================
if page == "🏠 Inicio":
    st.markdown("""
    ## Bienvenido al Coffee Analytics Dashboard
    
    Este dashboard te permite analizar y predecir el consumo de café a nivel mundial utilizando 
    modelos de Machine Learning y un sistema de consultas basado en NLP.
    
    ### Funcionalidades disponibles:
    
    1. **📊 Predicción de Consumo**
       - Predice el consumo de café para cualquier país y año
       - Soporta predicción por tipo de café específico o consumo total
    
    2. **🏷️ Clasificación de Café**
       - Clasifica el tipo de café predominante según el país y año
       - Utiliza una red neuronal con activación softmax
    
    3. **💬 Consultas RAG**
       - Realiza preguntas en lenguaje natural sobre el dataset
       - Combina un transformer (BERT) con recuperación de datos
    """)
    
    # Mostrar estadísticas generales
    if models['df_long'] is not None:
        st.markdown("### 📈 Estadísticas del Dataset")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Países", len(models['unique_countries']))
        with col2:
            st.metric("Tipos de Café", len(models['unique_coffee_types']))
        with col3:
            st.metric("Años de Datos", models['df_long']['Year'].nunique())
        with col4:
            st.metric("Registros Totales", len(models['df_long']))
        
        # Mostrar tipos de café disponibles
        st.markdown("### ☕ Tipos de Café Disponibles")
        for coffee in models['unique_coffee_types']:
            st.write(f"- {coffee}")

# ============================================
# PÁGINA: PREDICCIÓN DE CONSUMO
# ============================================
elif page == "📊 Predicción de Consumo":
    st.markdown("## 📊 Predicción de Consumo de Café")
    st.markdown("Utiliza el modelo de regresión para predecir el consumo futuro de café.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_country = st.selectbox(
            "Seleccionar País",
            options=models['unique_countries'],
            index=models['unique_countries'].index('Brazil') if 'Brazil' in models['unique_countries'] else 0
        )
    
    with col2:
        selected_year = st.number_input(
            "Año de Predicción",
            min_value=2020,
            max_value=2030,
            value=2025
        )
    
    # Predicción por tipo específico o total
    prediction_type = st.radio(
        "Tipo de Predicción",
        ["Consumo Total", "Por Tipo de Café Específico"],
        horizontal=True
    )
    
    if prediction_type == "Por Tipo de Café Específico":
        selected_coffee_type = st.selectbox(
            "Seleccionar Tipo de Café",
            options=models['unique_coffee_types']
        )
        
        if st.button("🔮 Predecir Consumo", type="primary"):
            with st.spinner('Realizando predicción...'):
                new_input = preprocess_new_data_for_reg(
                    selected_country, 
                    selected_coffee_type, 
                    selected_year,
                    models['preprocessor_reg']
                )
                predicted_scaled = models['regression_model'].predict(new_input, verbose=0)[0][0]
                predicted_consumption = models['scaler_y_reg'].inverse_transform([[predicted_scaled]])[0][0]
                
                st.success(f"### 📊 Resultado de Predicción")
                st.markdown(f"""
                **País:** {selected_country}
                
                **Año:** {selected_year}
                
                **Tipo de Café:** {selected_coffee_type}
                
                **Consumo Predicho:** {predicted_consumption:,.2f} unidades
                """)
    else:
        if st.button("🔮 Predecir Consumo Total", type="primary"):
            with st.spinner('Calculando predicción total...'):
                total_consumption = 0
                progress_bar = st.progress(0)
                
                for i, c_type in enumerate(models['unique_coffee_types']):
                    new_input = preprocess_new_data_for_reg(
                        selected_country, 
                        c_type, 
                        selected_year,
                        models['preprocessor_reg']
                    )
                    predicted_scaled = models['regression_model'].predict(new_input, verbose=0)[0][0]
                    predicted_consumption = models['scaler_y_reg'].inverse_transform([[predicted_scaled]])[0][0]
                    total_consumption += predicted_consumption
                    progress_bar.progress((i + 1) / len(models['unique_coffee_types']))
                
                st.success(f"### 📊 Resultado de Predicción")
                st.markdown(f"""
                **País:** {selected_country}
                
                **Año:** {selected_year}
                
                **Consumo Total Predicho:** {total_consumption:,.2f} unidades
                """)

# ============================================
# PÁGINA: CLASIFICACIÓN DE CAFÉ
# ============================================
elif page == "🏷️ Clasificación de Café":
    st.markdown("## 🏷️ Clasificación de Tipo de Café")
    st.markdown("Utiliza el modelo de clasificación para predecir el tipo de café predominante.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        cls_country = st.selectbox(
            "Seleccionar País",
            options=models['unique_countries'],
            index=0
        )
    
    with col2:
        cls_year = st.number_input(
            "Año",
            min_value=1990,
            max_value=2030,
            value=2025
        )
    
    if st.button("🏷️ Clasificar Tipo de Café", type="primary"):
        with st.spinner('Clasificando...'):
            new_data_cls = pd.DataFrame({
                'Country': [cls_country],
                'Year': [float(cls_year)]  # Convertir a float para el preprocesador
            })
            processed_data_cls = models['preprocessor_cls'].transform(new_data_cls)
            
            predictions_probabilities = models['classification_model'].predict(processed_data_cls, verbose=0)
            predicted_class_index = np.argmax(predictions_probabilities, axis=1)[0]
            predicted_coffee_type = models['coffee_type_classes'][predicted_class_index]
            probability = predictions_probabilities[0][predicted_class_index] * 100
            
            st.success(f"### 🏷️ Resultado de Clasificación")
            st.markdown(f"""
            **País:** {cls_country}
            
            **Año:** {cls_year}
            
            **Tipo de Café Predicho:** {predicted_coffee_type}
            
            **Confianza:** {probability:.2f}%
            """)
            
            # Mostrar todas las probabilidades
            st.markdown("#### Probabilidades por Tipo de Café:")
            prob_df = pd.DataFrame({
                'Tipo de Café': models['coffee_type_classes'],
                'Probabilidad': predictions_probabilities[0] * 100
            }).sort_values('Probabilidad', ascending=False)
            
            st.bar_chart(prob_df.set_index('Tipo de Café'))

# ============================================
# PÁGINA: CONSULTAS RAG
# ============================================
elif page == "💬 Consultas RAG":
    st.markdown("## 💬 Consultas al Dataset (RAG)")
    st.markdown("""
    Haz preguntas en lenguaje natural sobre el consumo de café. 
    El sistema utilizará extracción de entidades y los modelos ML para responder.
    """)
    
    # Ejemplos de consultas
    st.markdown("### 💡 Ejemplos de Consultas:")
    st.code("Predecir el consumo de café Arabica en Brazil en 2025")
    st.code("Consumo de café en Angola en 2018?")
    st.code("Datos de consumo para el año 2010 en Colombia.")
    
    # Campo de consulta
    query = st.text_input(
        "Escribe tu consulta:",
        placeholder="Ej: ¿Cuánto café consumirá Estados Unidos en 2025?"
    )
    
    if st.button("🔍 Buscar", type="primary"):
        if query:
            with st.spinner('Procesando consulta...'):
                # Extraer entidades
                extracted = extract_entities_and_intent(
                    query, 
                    models['unique_countries'], 
                    models['unique_coffee_types']
                )
                
                st.markdown("### 📌 Entidades Extraídas:")
                st.json(extracted)
                
                # Procesar consulta
                response = query_ml_models(extracted, models)
                
                st.markdown("### 💬 Respuesta:")
                st.markdown(response)
        else:
            st.warning("Por favor, escribe una consulta.")
    
    # Sección de consulta histórica directa
    st.markdown("---")
    st.markdown("### 📚 Consulta Histórica Directa")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        hist_country = st.selectbox(
            "País",
            options=models['unique_countries'],
            index=0
        )
    
    with col2:
        hist_coffee = st.selectbox(
            "Tipo de Café (opcional)",
            options=["Todos"] + models['unique_coffee_types'],
            index=0
        )
    
    with col3:
        hist_year = st.selectbox(
            "Año",
            options=sorted(models['df_long']['Year'].unique()),
            index=len(models['df_long']['Year'].unique()) - 1
        )
    
    if st.button("📊 Ver Datos Históricos"):
        filtered_df = models['df_long'][
            (models['df_long']['Country'] == hist_country) &
            (models['df_long']['Year'] == hist_year)
        ]
        
        if hist_coffee != "Todos":
            filtered_df = filtered_df[filtered_df['Coffee type'] == hist_coffee]
        
        if not filtered_df.empty:
            total = filtered_df['Consumption'].sum()
            st.success(f"Consumo en {hist_country} ({hist_year}): **{total:,.2f}** unidades")
            st.dataframe(filtered_df)
        else:
            st.warning("No se encontraron datos para los criterios seleccionados.")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "☕ Coffee Analytics Dashboard | Sistema de Inteligencia Artificial | "
    "Modelos: Clasificación (Softmax) + Regresión + RAG"
    "</div>",
    unsafe_allow_html=True
)