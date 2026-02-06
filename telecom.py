import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Title
st.title("Monthly Data Usage Prediction 📊")

# Load dataset
df = pd.read_csv("Book1.csv")

# Drop ID column
df = df.drop("customer_id", axis=1)

# Encode categorical columns
df['device_type'] = df['device_type'].map({'Android': 1, 'iOS': 0})
df['plan_type'] = df['plan_type'].map({'Postpaid': 1, 'Prepaid': 0})

# Features and target
X = df.drop('monthly_data_uses (GB)', axis=1)
y = df['monthly_data_uses (GB)']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree
dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)

# Train Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Sidebar for user input
st.sidebar.header("Input Customer Details")

avg_call_minute = st.sidebar.number_input("Average Call Minutes", min_value=0)
avg_sms_count = st.sidebar.number_input("Average SMS Count", min_value=0)
device_type = st.sidebar.selectbox("Device Type", ["Android", "iOS"])
internet_speed = st.sidebar.number_input("Internet Speed (Mbps)", min_value=0.0)
plan_type = st.sidebar.selectbox("Plan Type", ["Postpaid", "Prepaid"])
roaming_uses = st.sidebar.number_input("Roaming Uses (per month)", min_value=0)

# Encode input
device_type_enc = 1 if device_type == "Android" else 0
plan_type_enc = 1 if plan_type == "Postpaid" else 0

# Create input DataFrame
input_data = pd.DataFrame({
    'avg_call_minute': [avg_call_minute],
    'avg_sms_count': [avg_sms_count],
    'device_type': [device_type_enc],
    'internet_speed': [internet_speed],
    'plan_type': [plan_type_enc],
    'roaming_uses': [roaming_uses]
})

# Predictions
dt_pred = dt.predict(input_data)[0]
rf_pred = rf.predict(input_data)[0]

# Display results
st.subheader("Predicted Monthly Data Usage (GB)")
st.write(f"**Decision Tree Prediction:** {dt_pred:.2f} GB")
st.write(f"**Random Forest Prediction:** {rf_pred:.2f} GB")

# Model Evaluation on test set
y_pred_dt = dt.predict(X_test)
y_pred_rf = rf.predict(X_test)

st.subheader("Model Performance on Test Set")
st.write("**Decision Tree:**")
st.write(f"MSE: {mean_squared_error(y_test, y_pred_dt):.2f}")
st.write(f"R2 Score: {r2_score(y_test, y_pred_dt):.2f}")

st.write("**Random Forest:**")
st.write(f"MSE: {mean_squared_error(y_test, y_pred_rf):.2f}")
st.write(f"R2 Score: {r2_score(y_test, y_pred_rf):.2f}")