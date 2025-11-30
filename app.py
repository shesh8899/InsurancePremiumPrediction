import streamlit as st
from main import model  # Import model class

st.set_page_config(page_title="Insurance Premium Predictor", layout="centered")
st.title("🩺 Insurance Premium Prediction")
st.markdown("Please fill in the details below:")

with st.form("prediction_form"):
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    Gender = st.selectbox("Gender", ['Male', 'Female'])
    Region = st.selectbox("Region", ['Northeast', 'Northwest', 'Southeast', 'Southwest'])
    Marital_Status = st.selectbox("Marital Status", ['Unmarried', 'Married'])
    Number_of_Dependants = st.number_input("Number of Dependants", min_value=0, max_value=10, value=0)
    BMI_Category = st.selectbox("BMI Category", ['Overweight', 'Underweight', 'Normal', 'Obesity'])
    Smoking_Status = st.selectbox("Smoking Status", ['Regular', 'No Smoking', 'Occasional', 'NO Smoking'])
    Employment_Status = st.selectbox("Employment Status", ['Self-Employed', 'Freelancer', 'Salaried'])
    Income_Lakhs = st.number_input("Annual Income (in Lakhs)", min_value=0.0, step=0.1)
    Medical_History = st.selectbox(
        "Medical History",
        [
            'High blood pressure', 'No Disease', 'Diabetes & High blood pressure',
            'Diabetes & Heart disease', 'Diabetes', 'Diabetes & Thyroid',
            'Heart disease', 'Thyroid', 'High blood pressure & Heart disease'
        ]
    )
    Insurance_Plan = st.selectbox("Insurance Plan", ['Gold', 'Silver', 'Bronze'])
    Genetical_Risk = st.slider("Genetical Risk (0 - 10)", min_value=0, max_value=10)
    Income_Level = st.selectbox("Income Level", ['> 40L', '<10L', '10L - 25L', '25L - 40L'])

    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = {
        "age": age,
        "Gender": Gender,
        "Region": Region,
        "Marital_Status": Marital_Status,  # ✅ FIXED
        "Number_of_Dependants": Number_of_Dependants,
        "BMI_Category": BMI_Category,
        "Smoking_Status": Smoking_Status,
        "Employment_Status": Employment_Status,
        "Income_Lakhs": Income_Lakhs,
        "Medical_History": Medical_History,
        "Insurance_Plan": Insurance_Plan,
        "Genetical_Risk": Genetical_Risk,
        "Income_Level": Income_Level
    }

    st.subheader("🧾 User Input Summary")
    st.json(input_data)

    try:
        mod = model(**input_data)
        prediction = mod.dataframecreation()
        st.subheader("💡 Predicted Annual Premium")
        st.success(f"₹ {prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"❌ Prediction Failed: {e}")