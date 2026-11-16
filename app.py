import streamlit as st
import pandas as pd
import joblib

# Page Configuration
st.set_page_config(page_title="Supply Risk Prediction System", layout="wide")

# Load saved model and artifacts
artifacts = joblib.load('supply_risk_model.pkl')
model = artifacts['model']
scaler = artifacts['scaler']
encoded_columns = artifacts['encoded_columns']
feature_names = artifacts['feature_names']
num_bounds = artifacts['num_bounds']
cat_options = artifacts['cat_options']

# Header Section
st.title("Supplier Supply Risk Prediction System")
st.markdown("""
This interactive web application utilizes a **Multinomial Logistic Regression** model (**96.32% Accuracy**) 
to predict potential supply chain risks based on supplier attributes.
""")
st.divider()

# User Input Form with Drop-downs and Bounded Numeric Inputs
st.subheader("1. Select Supplier Attributes")

user_inputs = {}
cols = st.columns(3)

for i, col_name in enumerate(feature_names):
    with cols[i % 3]:
        if col_name in cat_options:
            # Drop-down menu restricted strictly to values present in the dataset
            user_inputs[col_name] = st.selectbox(
                label=f"{col_name}",
                options=cat_options[col_name]
            )
        else:
            # Numeric input showing and enforcing exact Min and Max values from the dataset
            bounds = num_bounds[col_name]
            min_val = bounds['min']
            max_val = bounds['max']
            default_val = bounds['median']

            if bounds['is_int']:
                user_inputs[col_name] = st.number_input(
                    label=f"{col_name} (Min: {int(min_val)}, Max: {int(max_val)})",
                    min_value=int(min_val),
                    max_value=int(max_val),
                    value=int(default_val),
                    step=1
                )
            else:
                user_inputs[col_name] = st.number_input(
                    label=f"{col_name} (Min: {min_val:.2f}, Max: {max_val:.2f})",
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(default_val),
                    step=0.1
                )

st.divider()

# Prediction Button & Output
if st.button("Predict Supply Risk", type="primary", use_container_width=True):
    # Convert user inputs into a single-row DataFrame
    input_df = pd.DataFrame([user_inputs])

    # Apply the exact same One-Hot Encoding and column alignment used in training
    input_encoded = pd.get_dummies(input_df, drop_first=True)
    input_encoded = input_encoded.reindex(columns=encoded_columns, fill_value=0)

    # Scale numerical features
    input_scaled = scaler.transform(input_encoded)

    # Make prediction and get probabilities
    prediction = model.predict(input_scaled)
    probabilities = model.predict_proba(input_scaled)

    # Display Result
    st.subheader("2. Prediction Outcome")
    if prediction == "No_Risk":
        st.success(f"Predicted Outcome: {prediction} (Low or No Supply Chain Risk)")
    else:
        st.error(f"Predicted Outcome: {prediction} (High Risk Detected)")

    # Display Probability Breakdown Chart
    prob_df = pd.DataFrame({
        'Risk Category': model.classes_,
        'Probability (%)': probabilities * 100
    }).sort_values(by='Probability (%)', ascending=False)

    st.write("**Prediction Confidence Breakdown:**")
    st.bar_chart(prob_df.set_index('Risk Category'))
