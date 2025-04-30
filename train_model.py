import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load data (place 'loan_data.csv' in this directory)
df = pd.read_csv("loan_data.csv")

# Preprocessing
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median(), inplace=True)
df['Credit_History'].fillna(0, inplace=True)
for col in ['Gender','Married','Dependents','Education','Self_Employed','Property_Area']:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Feature engineering
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df.drop(['Loan_ID'], axis=1, inplace=True)

# Encode categoricals
encoders = {}
for col in ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# Scale numeric features
scaler = StandardScaler()
num_cols = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','TotalIncome']
X[num_cols] = scaler.fit_transform(X[num_cols])

# Save scaler and encoders
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoders, "encoders.pkl")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))

# Save model
joblib.dump(model, "model.pkl")
