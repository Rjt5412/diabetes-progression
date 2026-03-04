import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import shap
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide"
)

# Load artifacts
@st.cache_resource
def load_model_assets():
    model = joblib.load('artifacts/best_model.joblib')
    scaler = joblib.load('artifacts/scaler.joblib')
    processed_data = joblib.load('artifacts/processed_data.joblib')
    with open('artifacts/threshold.json', 'r') as f:
        threshold_data = json.load(f)
    return model, scaler, threshold_data['optimal_threshold'], processed_data

@st.cache_resource
def get_shap_explainer(_model, _X_train):
    # Use k-means to summarize background data for speed (10 clusters)
    background = shap.kmeans(_X_train, 10)
    # Narrow output to just class 1 probability to simplify SHAP value extraction
    return shap.KernelExplainer(lambda x: _model.predict_proba(x)[:, 1], background)

model, scaler, opt_threshold, processed_data = load_model_assets()
explainer = get_shap_explainer(model, processed_data['X_train'])

# UI Layout
st.title("🩺 Diabetes Progression Risk Predictor")
st.markdown("""
Assesses diabetes progression risk based on current metabolic and physiological markers. 
The risk score is a model-derived probability used to prioritise patients for 
clinical review — it is not a diagnostic tool.
""")

# Feature Engineering
def preprocess_input(data_dict):
    df = pd.DataFrame([data_dict])
    
    # Matching the engineering from Phase 2/4
    df['glucose_bmi_interaction'] = df['Glucose'] * df['BMI']
    df['insulin_glucose_ratio'] = df['Insulin'] / (df['Glucose'] + 1e-6)
    
    # Scale features
    # Note: scale expect same order as X_train
    feature_order = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 
        'BMI', 'DiabetesPedigreeFunction', 'Age', 'glucose_bmi_interaction', 
        'insulin_glucose_ratio'
    ]
    df = df[feature_order]
    scaled_values = scaler.transform(df)
    return pd.DataFrame(scaled_values, columns=feature_order)

# Tab Layout
tab_dashboard, tab_about = st.tabs(["📊 Prediction Dashboard", "ℹ️ About the Model"])

with tab_dashboard:
    # Sidebar Patient Parameters
    with st.sidebar:
        st.header("Patient Parameters")
        
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, 
                                      help="Number of times pregnant.")
        glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=300, value=120, 
                                  help="Plasma glucose concentration a 2 hours in an oral glucose tolerance test.")
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=150, value=70, 
                                         help="Diastolic blood pressure (mm Hg).")
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20, 
                                         help="Triceps skin fold thickness (mm).")
        insulin = st.number_input("Insulin (mu U/ml)", min_value=0, max_value=900, value=80, 
                                  help="2-Hour serum insulin (mu U/ml).")
        bmi = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=70.0, value=25.0, 
                              help="Body mass index (weight in kg/(height in m)^2).")
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.3f", 
                              help="A function which scores likelihood of diabetes based on family history.")
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=30, 
                              help="Age (years).")

    # Prediction
    if st.button("Calculate Risk Score"):
        input_data = {
            'Pregnancies': pregnancies,
            'Glucose': glucose,
            'BloodPressure': blood_pressure,
            'SkinThickness': skin_thickness,
            'Insulin': insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction': dpf,
            'Age': age
        }
        
        processed_df = preprocess_input(input_data)
        prob = model.predict_proba(processed_df)[0, 1]
        is_high_risk = prob >= opt_threshold

        # Visualization
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric("Risk Probability", f"{prob:.1%}", delta=None)
            if is_high_risk:
                st.error("⚠️ HIGH RISK IDENTIFIED")
            else:
                st.success("✅ LOW RISK IDENTIFIED")
                
        with col2:
            st.progress(float(np.clip(prob, 0.0, 1.0)))
            st.caption(f"Decision Threshold: {opt_threshold:.2f}")

        # SHAP Explainability
        st.subheader("Clinical Explanation (SHAP)")
        st.caption("How to interpret: 🔴 RED bars increase risk, 🟢 GREEN bars decrease risk relative to the average patient.", 
                   help="Values shown are SHAP (SHapley Additive exPlanations). They quantify how much each patient marker pushed the probability away from the base value (average risk).")
        
        with st.spinner("Calculating feature contributions... (first run may take 15–20 seconds)"):
            shap_values = explainer.shap_values(processed_df, silent=True)
            current_shap = shap_values[0]
            
            features = processed_df.columns
            shap_df = pd.DataFrame({
                'Feature': features,
                'Contribution': current_shap
            }).sort_values(by='Contribution', ascending=True)

            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#ff4b4b' if x > 0 else '#2fa44e' for x in shap_df['Contribution']]
            ax.barh(shap_df['Feature'], shap_df['Contribution'], color=colors)
            ax.set_xlabel("Contribution to Risk Probability")
            ax.set_title("Feature Influence on this Prediction")
            st.pyplot(fig)
            st.caption(f"Base rate (average patient risk): {explainer.expected_value:.1%}. SHAP values show how this patient's markers shift the prediction away from that baseline.")

        # Narrative explanation
        top_pos = shap_df[shap_df['Contribution'] > 0.01].sort_values(by='Contribution', ascending=False)
        top_neg = shap_df[shap_df['Contribution'] < -0.01].sort_values(by='Contribution', ascending=True)
        risk_status = "HIGH" if is_high_risk else "LOW"
        narrative = f"**Current Status:** {risk_status} Risk ({prob:.1%} probability).\n\n"
        
        if not top_pos.empty:
            pos_list = [f"{row['Feature']} (+{row['Contribution']:.2f})" for _, row in top_pos.head(3).iterrows()]
            narrative += f"**Primary Risk Drivers:** {', '.join(pos_list)}\n\n"
        
        if not top_neg.empty:
            neg_list = [f"{row['Feature']} ({row['Contribution']:.2f})" for _, row in top_neg.head(3).iterrows()]
            narrative += f"**Key Protective Factors:** {', '.join(neg_list)}\n\n"

        st.info(narrative)

        # Reliability Warnings
        if age >= 60 or glucose < 100 or bmi < 25.0:
            st.subheader("Model Reliability Advisory")
            if age >= 60:
                st.warning("**Age 60+ Advisory:** Reduced reliability for patients over 60. Model performance in this cohort is unverified due to low representation in training data.")
            if glucose < 100:
                st.warning("**Normal Glucose Advisory:** High prediction variance for baseline glucose <100 mg/dL. Metabolic stability may mask progression markers.")
            if bmi < 25.0:
                st.warning("**BMI Advisory:** Patients with BMI below 25.0 are underrepresented in the training data — predictions for this group have higher uncertainty.")

with tab_about:
    st.header("About the Diabetes Progression Risk Predictor")
    st.markdown("""
    ### 🎯 Objective
    This tool is a prototype designed to assess **diabetes onset risk** from current 
    metabolic and physiological markers. 
    It uses a Support Vector Machine (SVM) model trained on the standard Pima Indians Diabetes Database to identify clinical markers correlated with metabolic deterioration.

    ### 📊 Dataset Reference
    The model is based on the **Pima Indians Diabetes Dataset** (National Institute of Diabetes and Digestive and Kidney Diseases). 
    All patients listed are females at least 21 years old of Pima Indian heritage.

    ### 🛡️ Interpretability & Fairness
    - **SHAP (SHapley Additive exPlanations):** We use game-theoretic SHAP values to provide local feature importance for every individual prediction, ensuring clinical transparency.
    - **Fairness Audit:** The model has been audited across Age and BMI subgroups. Reliability advisories are triggered automatically for demographic segments (e.g., Age 60+) where training data was sparse.

    ### ⚠️ Disclaimer
    *This is a clinical decision support prototype and is NOT a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider.*
    """)

st.divider()
st.caption("Clinical Decision Support Prototype · Pima Indians Diabetes Database · Model: SVM (RBF) · Threshold: 0.374")
