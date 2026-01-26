import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #e74c3c;
        color: white;
        font-weight: bold;
        padding: 0.5rem;
        border-radius: 10px;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
    }
    .disease { background-color: #ffebee; color: #c62828; }
    .healthy { background-color: #e8f5e9; color: #2e7d32; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        model_data = joblib.load("heart_disease_model.pkl")
        scaler = joblib.load("heart_scaler.pkl")
        return model_data["model"], model_data["feature_names"], scaler
    except FileNotFoundError:
        st.error("Model files not found. Run the notebook first.")
        return None, None, None

model, feature_names, scaler = load_model()

st.title("Heart Disease Prediction")
st.markdown("Predict heart disease risk using Machine Learning")
st.markdown("---")

if model is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personal Information")
        age = st.slider("Age", 20, 100, 50)
        sex = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x==1 else "Female")
        
        st.subheader("Heart Metrics")
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], 
                         format_func=lambda x: ["Typical Angina", "Atypical Angina", 
                                               "Non-anginal Pain", "Asymptomatic"][x])
        trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
        chol = st.slider("Cholesterol (mg/dl)", 100, 600, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], 
                          format_func=lambda x: "Yes" if x==1 else "No")
        restecg = st.selectbox("Resting ECG", [0, 1, 2],
                              format_func=lambda x: ["Normal", "ST-T Abnormality", "LV Hypertrophy"][x])
    
    with col2:
        thalach = st.slider("Max Heart Rate", 60, 220, 150)
        exang = st.selectbox("Exercise Induced Angina", [0, 1],
                            format_func=lambda x: "Yes" if x==1 else "No")
        oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0, 0.1)
        slope = st.selectbox("Slope of Peak Exercise ST", [0, 1, 2],
                            format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
        ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
        thal = st.selectbox("Thalassemia", [1, 2, 3],
                           format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect"][x-1])
    
    st.markdown("---")
    
    if st.button("Predict Heart Disease"):
        input_data = {
            'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
            'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
            'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
        }
        
        input_df = pd.DataFrame([input_data])
        input_df = input_df[feature_names]
        input_scaled = scaler.transform(input_df)
        
        # CORRECTION : Gérer différemment la prédiction selon le type de modèle
        try:
            prediction = model.predict(input_scaled)[0]
            
            # Essayer d'obtenir les probabilités
            try:
                proba = model.predict_proba(input_scaled)[0]
            except (AttributeError, IndexError):
                # Si predict_proba n'est pas disponible, utiliser une valeur par défaut
                proba = [0.5, 0.5] if prediction == 0 else [0.5, 0.5]
                
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if prediction == 1:
                st.markdown('<div class="prediction-box disease">DISEASE DETECTED</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="prediction-box healthy">HEALTHY</div>', unsafe_allow_html=True)
        
        with col2:
            # Calculer la confiance différemment selon la disponibilité des probabilités
            if len(proba) == 2:
                confidence = max(proba) * 100
                st.metric("Confidence", f"{confidence:.1f}%")
            else:
                st.metric("Prediction", "Positive" if prediction == 1 else "Negative")
        
        with col3:
            if len(proba) == 2:
                risk = "High" if proba[1] > 0.7 else "Medium" if proba[1] > 0.4 else "Low"
            else:
                risk = "High" if prediction == 1 else "Low"
            st.metric("Risk Level", risk)
        
        # Graphique seulement si les probabilités sont disponibles
        if len(proba) == 2:
            fig = go.Figure(data=[
                go.Bar(name='Healthy', x=['Probability'], y=[proba[0]], marker_color='#2ecc71'),
                go.Bar(name='Disease', x=['Probability'], y=[proba[1]], marker_color='#e74c3c')
            ])
            fig.update_layout(
                title="Prediction Probabilities",
                yaxis_title="Probability",
                barmode='group',
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Probability distribution not available for this model type.")
        
        if prediction == 1:
            st.error("""
            **High Risk Detected:** 
            - Consult a cardiologist immediately
            - Maintain healthy lifestyle
            - Monitor blood pressure and heart rate regularly
            - Follow medical advice regarding medication
            """)
        else:
            st.success("""
            **Low Risk:** 
            - Continue healthy habits
            - Exercise regularly (30 minutes daily)
            - Maintain balanced diet
            - Schedule regular medical checkups
            - Avoid smoking and excessive alcohol
            """)
        
        with st.expander("Input Summary"):
            st.write(input_data)
    
    with st.sidebar:
        st.header("Model Information")
        st.info(f"""
        **Model Type:** {type(model).__name__}
        **Features Used:** {len(feature_names)} attributes
        
        **Important Notes:**
        - This tool is for educational purposes
        - Model accuracy may vary
        - Always consult healthcare professionals
        """)
        
        st.header("Key Risk Factors")
        st.markdown("""
        ✅ **Major Factors:**
        - Chest pain type (Asymptomatic is high risk)
        - Age (risk increases with age)
        - Resting blood pressure
        - Cholesterol levels
        - Maximum heart rate
        
        ⚠️ **Other Factors:**
        - Exercise-induced angina
        - ST depression during exercise
        - Number of major vessels
        - Thalassemia type
        """)
        
        st.markdown("---")
        st.caption("⚠️ **Disclaimer:** This application is for educational and research purposes only.")
        st.caption("Not a substitute for professional medical diagnosis or treatment.")
        st.caption("Always consult qualified healthcare providers for medical decisions.")

else:
    st.error("""
    Model not found. Please ensure:
    1. You have run the notebook to generate the model
    2. The files are in the correct directory:
       - heart_disease_model.pkl
       - heart_scaler.pkl
    3. The files are in the same folder as this app
    """)
    
    # Instructions for troubleshooting
    with st.expander("Troubleshooting Guide"):
        st.write("""
        **How to fix this error:**
        
        1. **Check if model files exist:**
           - Look in the current directory for:
             - `heart_disease_model.pkl`
             - `heart_scaler.pkl`
        
        2. **If files don't exist:**
           - Open and run the Jupyter notebook that creates the model
           - Ensure the notebook saves the files correctly
        
        3. **If using a different directory:**
           - Update the file paths in the `load_model()` function
           - Example: `joblib.load("path/to/your/heart_disease_model.pkl")`
        
        4. **Common issues:**
           - File permissions
           - Different Python environment
           - Corrupted model files
        """)