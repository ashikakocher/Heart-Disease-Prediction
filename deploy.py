import numpy as np
import pickle
import streamlit as st

# Load the trained model
loaded_model = pickle.load(open("C:/Users/ASHIKA/Desktop/mlproject/trained_model.sav", 'rb'))

# Prediction function
def heartdisease_prediction(input_data):
    try:
        # Convert input to numpy array
        input_data_as_numpy_array = np.asarray(input_data, dtype=float)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

        prediction = loaded_model.predict(input_data_reshaped)

        if prediction[0] == 0:
            return '✅ The person does NOT have heart disease.'
        else:
            return '⚠️ The person HAS heart disease.'
    except Exception as e:
        return f"Error in prediction: {e}"

# Main function for Streamlit
def main():
    st.set_page_config(page_title="Heart Disease Prediction", layout="centered")
    st.title("❤️ Heart Disease Prediction Web App")

    st.markdown("Enter the following values to check heart disease risk:")

    # User input
    age = st.number_input("Age", min_value=1)
    sex = st.selectbox("Sex", ["0 = Female", "1 = Male"])
    cp = st.number_input("Chest Pain Type (cp)", min_value=0, max_value=3)
    trestbps = st.number_input("Resting Blood Pressure (trestbps)")
    chol = st.number_input("Cholesterol")
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", ["0 = False", "1 = True"])
    restecg = st.number_input("Resting ECG Results (restecg)", min_value=0, max_value=2)
    thalach = st.number_input("Max Heart Rate Achieved (thalach)")
    exang = st.selectbox("Exercise Induced Angina (exang)", ["0 = No", "1 = Yes"])
    oldpeak = st.number_input("ST depression (oldpeak)")
    slope = st.number_input("Slope of the peak exercise ST segment", min_value=0, max_value=2)
    ca = st.number_input("Number of major vessels (ca)", min_value=0, max_value=4)
    thal = st.number_input("Thalassemia (thal)", min_value=0, max_value=3)

    # Prepare for prediction
    if st.button("🔍 Predict Heart Disease"):
        features = [
            age,
            int(sex.split(" = ")[0]),
            cp,
            trestbps,
            chol,
            int(fbs.split(" = ")[0]),
            restecg,
            thalach,
            int(exang.split(" = ")[0]),
            oldpeak,
            slope,
            ca,
            thal
        ]
        result = heartdisease_prediction(features)
        st.success(result)

if __name__ == '__main__':
    main()
