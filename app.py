import streamlit as st
import numpy as np
import joblib

# Load saved model and scaler
scaler = joblib.load('scaler.pkl')
model = joblib.load('feature_DissionTree.pkl')

# App title and description
st.title("📊 Customer Churn Prediction")
st.divider()
st.write("This web app predicts whether a customer is likely to churn based on their inputs.")
st.divider()

# User inputs
age = st.number_input("Enter your age", min_value=18, max_value=99, value=25)
gender = st.selectbox("Select your gender", ["Male", "Female"])
tenure = st.slider("How long have you been with us?", min_value=0, max_value=130, value=10)
monthly_charges = st.number_input("Enter your monthly charges", min_value=30.0, max_value=150.0, value=75.0)

st.divider()

# Predict button
if st.button("Predict"):
    # Encode gender (Female: 1, Male: 0)
    gender_encoded = 1 if gender == "Female" else 0
    st.balloons()

    # Prepare input for model
    input_data = np.array([[age, gender_encoded, tenure, monthly_charges]])
    input_scaled = scaler.transform(input_data)

    # Make prediction
    predicted_class = model.predict(input_scaled)[0]
    result = "Churn" if predicted_class == 1 else "Not Churn"

    st.success(f"🔍 Prediction: **{result}**")

    # Show probability if available
    if hasattr(model, "predict_proba"):
        try:
            probability = model.predict_proba(input_scaled)[0][1]  # Probability of churn
            st.info(f"📈 Probability of churn: **{probability:.2%}**")
        except Exception as e:
            st.warning("⚠️ Could not calculate probability.")
else:
    st.info("👆 Fill out the inputs and click **Predict** to see results.")
