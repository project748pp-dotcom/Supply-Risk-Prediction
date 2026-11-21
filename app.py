import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Supply Risk Prediction System", layout="wide")

artifacts = joblib.load('supply_risk_model.pkl')
model = artifacts['model']
scaler = artifacts['scaler']
encoded_columns = artifacts['encoded_columns']
feature_names = artifacts['feature_names']
dropdown_options = artifacts['dropdown_options']

feature_metadata = {
    "Response": {
        "title": "Supplier Response Time",
        "desc": "Time taken by the supplier to respond to inquiries or orders (1 = 1 day, 4 = >7 days).",
        "labels": {1: "1 (Within 1 day)", 2: "2 (1–3 days)", 3: "3 (4–7 days)", 4: "4 (More than 7 days)"}
    },
    "Order-type": {
        "title": "Order Type",
        "desc": "Nature of the purchase order (0 = New order, 1 = Repetitive order).",
        "labels": {0: "0 (New Order)", 1: "1 (Repetitive Order)"}
    },
    "Cycle-time": {
        "title": "Order Cycle Time",
        "desc": "Total lead time from order placement to delivery (1 = Within 1 week, 5 = >12 weeks).",
        "labels": {1: "1 (Within 1 week)", 2: "2 (2–4 weeks)", 3: "3 (5–8 weeks)", 4: "4 (9–12 weeks)", 5: "5 (More than 12 weeks)"}
    },
    "Shipment-route": {
        "title": "Shipment Route",
        "desc": "Primary transportation mode used for delivery (Air, Surface/Land, or Other).",
        "labels": {"AIR": "AIR (Air Freight)", "Surface": "Surface (Land/Sea)", "Other": "Other"}
    },
    "ISO-Certi": {
        "title": "ISO Certification",
        "desc": "Indicates whether the supplier holds valid ISO quality/management certifications.",
        "labels": {"YES": "YES (Certified)", "NO": "NO (Not Certified)"}
    },
    "Quali_ Award": {
        "title": "Quality Awards",
        "desc": "Indicates whether the supplier has received recognized industry quality awards.",
        "labels": {"YES": "YES", "NO": "NO"}
    },
    "Manu-yield": {
        "title": "Manufacturing Yield Rate",
        "desc": "Percentage of defect-free units produced (1 = 91–100% highest yield, 5 = <60% lowest yield).",
        "labels": {1: "1 (91–100%)", 2: "2 (81–90%)", 3: "3 (71–80%)", 4: "4 (61–70%)", 5: "5 (Below 60%)"}
    },
    "Warranty": {
        "title": "Product Warranty",
        "desc": "Indicates whether the supplier provides warranty coverage on supplied products.",
        "labels": {"YES": "YES", "NO": "NO"}
    },
    "Quali_ Inspec": {
        "title": "Quality Inspection Method",
        "desc": "Inspection method applied: Statistical Process Control (SPC), Batch Inspection (BI), or In-Work Inspection (IWI).",
        "labels": {"SPC": "SPC (Statistical Process Control)", "BI": "BI (Batch Inspection)", "IWI": "IWI (In-Work Inspection)"}
    },
    "Quali_Reco": {
        "title": "Quality Documentation & Records",
        "desc": "Indicates whether the supplier maintains formal quality control records and documentation.",
        "labels": {"YES": "YES", "NO": "NO"}
    },
    "Quali_impro": {
        "title": "Quality Improvement Programs",
        "desc": "Indicates whether the supplier actively implements continuous quality improvement initiatives.",
        "labels": {"YES": "YES", "NO": "NO"}
    },
    "Produ_Facility": {
        "title": "Production Facility Level",
        "desc": "Condition and modernization level of manufacturing facilities (High-Adv to Bad).",
        "labels": {"High-Adv": "High-Adv (Highly Advanced)", "Adv": "Adv (Advanced)", "Good": "Good", "Ave": "Ave (Average)", "Bad": "Bad"}
    },
    "Tech_Cap": {
        "title": "Technical Capabilities",
        "desc": "Engineering and technical expertise level of the supplier (Exce to Bad).",
        "labels": {"Exce": "Exce (Excellent)", "Good": "Good", "Ave": "Ave (Average)", "below_Ave": "below_Ave (Below Average)", "Bad": "Bad"}
    },
    "Cap_ Utili": {
        "title": "Capacity Utilization",
        "desc": "Proportion of total production capacity currently utilized (1 = 91–100%, 5 = <60%).",
        "labels": {1: "1 (91–100%)", 2: "2 (81–90%)", 3: "3 (71–80%)", 4: "4 (61–70%)", 5: "5 (Below 60%)"}
    },
    "Availability": {
        "title": "Market Supply Availability",
        "desc": "Availability of alternative suppliers or materials in the market (High, Medium, Low).",
        "labels": {"High": "High", "Med": "Med (Medium)", "Low": "Low"}
    },
    "Ex_rate": {
        "title": "Exchange Rate Volatility",
        "desc": "Level of currency exchange rate fluctuation affecting procurement costs.",
        "labels": {"High": "High", "Med": "Med (Medium)", "Low": "Low"}
    },
    "Commodity-price": {
        "title": "Commodity Price Volatility",
        "desc": "Degree of raw material and commodity price instability.",
        "labels": {"High": "High", "Med": "Med (Medium)", "Low": "Low"}
    },
    "Relationship": {
        "title": "Supplier Relationship Duration",
        "desc": "Length of business partnership with the supplier (1 = >5 years, 5 = New supplier).",
        "labels": {1: "1 (Above 5 years)", 2: "2 (4–5 years)", 3: "3 (2–3 years)", 4: "4 (1 year)", 5: "5 (New supplier)"}
    },
    "Supplier-lock": {
        "title": "Sourcing Strategy (Supplier Lock-in)",
        "desc": "Dependency structure on the supplier (Sole, Dual, or Multiple sources).",
        "labels": {"Sole": "Sole (Single Supplier)", "Dual": "Dual (Two Suppliers)", "Multi": "Multi (Multiple Suppliers)"}
    },
    "Info-sharing": {
        "title": "Information Sharing Level",
        "desc": "Transparency and frequency of operational data sharing with the supplier.",
        "labels": {"High": "High", "Med": "Med (Medium)", "Low": "Low"}
    },
    "Nat-disasters": {
        "title": "Natural Disasters Risk",
        "desc": "Exposure of the supplier's region to natural disasters (High to V-low).",
        "labels": {"High": "High", "Med": "Med (Medium)", "Low": "Low", "V-low": "V-low (Very Low)"}
    },
    "Man-Disaster": {
        "title": "Man-Made Disasters Risk",
        "desc": "Exposure to operational accidents, strikes, or man-made disruptions (High to V-low).",
        "labels": {"High": "High", "Med": "Med (Medium)", "Low": "Low", "V-low": "V-low (Very Low)"}
    },
    "Politi-envir": {
        "title": "Political Environment Stability",
        "desc": "Political stability in the supplier's operating country or region.",
        "labels": {"High": "High", "Better": "Better", "Good": "Good", "Weak": "Weak"}
    },
    "Custom-Regu": {
        "title": "Customs & Trade Regulations",
        "desc": "Efficiency and ease of customs clearance and import/export regulations.",
        "labels": {"Best": "Best", "Good": "Good", "Ave": "Ave (Average)", "Bel-Ave": "Bel-Ave (Below Average)"}
    },
    "Infrastructure": {
        "title": "Logistics Infrastructure Quality",
        "desc": "Quality of roads, ports, and logistics networks in the supplier's region.",
        "labels": {"Best": "Best", "Good": "Good", "Ave": "Ave (Average)", "Bel-Ave": "Bel-Ave (Below Average)"}
    },
    "Eco_Situ": {
        "title": "Economic Situation (Freedom Index)",
        "desc": "Macroeconomic stability and economic freedom of the supplier's country.",
        "labels": {"Most-free": "Most-free", "Free": "Free", "Mode-Free": "Mode-Free (Moderately Free)", "Unfree": "Unfree", "Repressed": "Repressed"}
    }
}

risk_labels = {
    "No_Risk": "No Supply Risk — Supplier operates within safe performance thresholds.",
    "Delivery_Risk": "Delivery Risk — Potential delay or incomplete fulfillment of purchase orders.",
    "Quali_Risk": "Quality Risk — Received order may exceed acceptable defect limits.",
    "Disruption": "Supply Disruption — High probability of severe delay, quality failure, or non-delivery."
}

st.title("Supplier Supply Risk Prediction System")
st.markdown("This interactive decision-support tool uses a **Multinomial Logistic Regression** model (**96.32% Accuracy**) to evaluate supplier attributes and predict potential supply chain risks.")
st.divider()

st.subheader("1. Select Supplier Attributes")

user_inputs = {}
cols = st.columns(3)

for i, col_name in enumerate(feature_names):
    with cols[i % 3]:
        meta = feature_metadata.get(col_name, {"title": col_name, "desc": "", "labels": {}})
        options = dropdown_options[col_name]

        user_inputs[col_name] = st.selectbox(
            label=f"{meta['title']} ({col_name})",
            options=options,
            format_func=lambda x, m=meta['labels']: m.get(x, str(x)),
            key=f"input_{col_name}"
        )
        if meta["desc"]:
            st.caption(meta["desc"])
        st.write("")

st.divider()

if st.button("Predict Supply Risk", type="primary", use_container_width=True):
    input_df = pd.DataFrame([user_inputs])

    # CRITICAL FIX: Do NOT use drop_first=True on a single row, otherwise the selected value gets dropped!
    input_encoded = pd.get_dummies(input_df, drop_first=False)
    input_encoded = input_encoded.reindex(columns=encoded_columns, fill_value=0)

    input_scaled = scaler.transform(input_encoded)

    pred_code = str(model.predict(input_scaled)[0])
    readable_result = risk_labels.get(pred_code, pred_code)
    probabilities = model.predict_proba(input_scaled)[0]

    st.subheader("2. Prediction Outcome")
    if pred_code == "No_Risk":
        st.success(f"Predicted Status: {readable_result}")
    else:
        st.error(f"Alert - Predicted Status: {readable_result}")

    prob_df = pd.DataFrame({
        "Risk Category": [c.replace("_", " ") for c in model.classes_],
        "Probability (%)": probabilities * 100
    }).sort_values(by="Probability (%)", ascending=False)

    st.write("Prediction Confidence Breakdown (%):")
    st.bar_chart(prob_df.set_index("Risk Category"))
