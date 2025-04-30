import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import os

df=pd.read_csv("loan_data.csv")

st.title("Loan Approval Predictor")

# Sidebar or button to toggle visualizations
if st.checkbox("📊 Show Data Visualizations"):
    st.subheader("Loan Approval Rate by Credit History")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x="Credit_History", hue="Loan_Status", ax=ax1)
    st.pyplot(fig1)

    st.subheader("Loan Amount Distribution")
    fig2, ax2 = plt.subplots()
    sns.histplot(df["LoanAmount"].dropna(), kde=True, ax=ax2)
    st.pyplot(fig2)

    st.subheader("Income vs Loan Amount")
    fig3, ax3 = plt.subplots()
    sns.scatterplot(data=df, x="ApplicantIncome", y="LoanAmount", hue="Loan_Status", ax=ax3)
    st.pyplot(fig3)

    st.subheader("Gender vs Loan Amount")
    fig4, ax4 = plt.subplots()
    sns.countplot(data=df, x="Gender", hue="Loan_Status")
    st.pyplot(fig4)

    fig = px.histogram(df, x="LoanAmount", color="Loan_Status", nbins=30, title="Loan Amount Distribution")
    st.plotly_chart(fig)



# Load model, scaler, and encoders
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
file_path = "scaler.pkl"
if os.path.exists(file_path):
    scaler = joblib.load(file_path)
else:
    print(f"Error: {file_path} not found!")
encoders = joblib.load("encoders.pkl")

st.subheader("Predict whether your loan gets approved or not!!")
# Sidebar inputs
gender = st.selectbox("Gender", ["Male","Female"])
married = st.selectbox("Married", ["Yes","No"])
dependents = st.selectbox("Dependents", ["0","1","2","3+"])
education = st.selectbox("Education", ["Graduate","Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes","No"])
app_income = st.number_input("Applicant Income", min_value=0)
coapp_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0)
loan_term = st.number_input("Loan Amount Term (months)", min_value=0)
credit_history = st.selectbox("Credit History (1 = meets criteria)", [1,0])
property_area = st.selectbox("Property Area", ["Urban","Rural","Semiurban"])

if st.button("Predict"):
    # Prepare DataFrame
    df = pd.DataFrame({
        'Gender':[gender],
        'Married':[married],
        'Dependents':[dependents],
        'Education':[education],
        'Self_Employed':[self_employed],
        'ApplicantIncome':[app_income],
        'CoapplicantIncome':[coapp_income],
        'LoanAmount':[loan_amount],
        'Loan_Amount_Term':[loan_term],
        'Credit_History':[credit_history],
        'Property_Area':[property_area],
    })
    df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']

    # Encode categoricals using saved encoders
    for col in ['Gender','Married','Dependents','Education','Self_Employed','Property_Area']:
        df[col] = encoders[col].transform(df[col])

    # Scale numeric features
    num_cols = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','TotalIncome']
    df[num_cols] = scaler.transform(df[num_cols])

    # Predict
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0][1]
    label = "Approved" if pred == 1 else "Rejected"

    st.write(f"**Prediction:** {label}")
    st.write(f"**Confidence:** {proba * 100:.2f}%")

