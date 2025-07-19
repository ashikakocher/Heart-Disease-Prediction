import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open("trained_model.sav", "rb") as f:
    model = pickle.load(f)

# Title
st.markdown("<h1 style='text-align: center; color: red;'>❤️ Heart Disease Prediction App</h1>", unsafe_allow_html=True)
st.markdown("This app predicts whether a person is at risk of heart disease based on health parameters.")

# Sidebar info
st.sidebar.title("About")
st.sidebar.info("Built using Streamlit\n\nModel: Machine Learning\n\nDemo project")

# Input form
st.header("🧾 Enter Patient Details")

def user_input_features():
    age = st.number_input("Age", 1, 120, 45)
    sex = st.selectbox("Sex", ['Male', 'Female'])
    cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (trestbps)", 80, 200, 120)
    chol = st.number_input("Serum Cholesterol (chol)", 100, 400, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
    restecg = st.selectbox("Resting ECG (restecg)", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved (thalach)", 70, 210, 150)
    exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
    oldpeak = st.number_input("ST depression (oldpeak)", 0.0, 6.0, 1.0, step=0.1)
    slope = st.selectbox("Slope of ST segment (slope)", [0, 1, 2])
    ca = st.selectbox("Number of major vessels (ca)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

    sex = 1 if sex == 'Male' else 0

    data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }
    features = pd.DataFrame([data])
    return features

input_df = user_input_features()

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    result = "🔴 At Risk of Heart Disease" if prediction == 1 else "🟢 Not at Risk"
    st.subheader("Prediction:")
    st.success(result)

# Bulk CSV upload
st.header("📤 Upload CSV for Bulk Prediction")
uploaded_file = st.file_uploader("Upload a CSV file with the same column format as the model", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    predictions = model.predict(df)
    df['Prediction'] = np.where(predictions == 1, "At Risk", "Not at Risk")
    st.write(df)

    # Download option
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Results",
        data=csv,
        file_name='heart_disease_predictions.csv',
        mime='text/csv',
    )
