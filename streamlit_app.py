import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import joblib
import plotly.graph_objects as go
from datetime import datetime

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.utils.features import engineer_features

st.set_page_config(layout="wide", page_title="ChurnGuard", page_icon="🔮")

# Inject Custom CSS
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background-color: #0f2027;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if 'history' not in st.session_state:
    st.session_state['history'] = []

# Resource Loading
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'best_model.pkl')
    if not os.path.exists(model_path):
        return None, None
    pipeline = joblib.load(model_path)
    import shap
    explainer = shap.TreeExplainer(pipeline.named_steps['classifier'])
    return pipeline, explainer

@st.cache_data
def load_scored_data():
    path = os.path.join(os.path.dirname(__file__), 'data', 'processed', 'customers_scored.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

MODEL_PIPELINE, SHAP_EXPLAINER = load_model()
SCORED_DF = load_scored_data()

# Helper Functions
def get_risk_tier(prob):
    if prob > 0.75: return "Critical", "#FF4B4B" # Red
    elif prob > 0.50: return "High", "#FFA500" # Orange
    elif prob > 0.25: return "Medium", "#FFD700" # Yellow
    else: return "Low", "#00C853" # Green

def get_recommendation(tier):
    if tier == "Critical": return "Urgent: Personal retention call + 25% discount"
    elif tier == "High": return "Offer annual contract upgrade within 48 hours"
    elif tier == "Medium": return "Send targeted email with loyalty incentive"
    else: return "Upsell opportunity — customer is stable"

def render_gauge(prob):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prob * 100,
        number = {'suffix': "%"},
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Churn Probability"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkgray"},
            'steps': [
                {'range': [0, 30], 'color': "#00C853"},
                {'range': [30, 60], 'color': "#FFD700"},
                {'range': [60, 80], 'color': "#FFA500"},
                {'range': [80, 100], 'color': "#FF4B4B"}
            ]
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
    return fig

def sidebar_inputs(prefix=""):
    st.sidebar.header("Demographics")
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"], key=f"{prefix}gender")
    SeniorCitizen_str = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"], key=f"{prefix}SeniorCitizen")
    SeniorCitizen = 1 if SeniorCitizen_str == "Yes" else 0
    Partner = st.sidebar.selectbox("Partner", ["Yes", "No"], key=f"{prefix}Partner")
    Dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"], key=f"{prefix}Dependents")
    
    st.sidebar.header("Services")
    PhoneService = st.sidebar.selectbox("Phone Service", ["Yes", "No"], key=f"{prefix}PhoneService")
    MultipleLines = st.sidebar.selectbox("Multiple Lines", ["No phone service", "No", "Yes"], key=f"{prefix}MultipleLines")
    InternetService = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], key=f"{prefix}InternetService")
    OnlineSecurity = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"], key=f"{prefix}OnlineSecurity")
    OnlineBackup = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"], key=f"{prefix}OnlineBackup")
    DeviceProtection = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"], key=f"{prefix}DeviceProtection")
    TechSupport = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"], key=f"{prefix}TechSupport")
    StreamingTV = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"], key=f"{prefix}StreamingTV")
    StreamingMovies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"], key=f"{prefix}StreamingMovies")
    
    st.sidebar.header("Account")
    Contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key=f"{prefix}Contract")
    PaperlessBilling = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"], key=f"{prefix}PaperlessBilling")
    PaymentMethod = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], key=f"{prefix}PaymentMethod")
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12, key=f"{prefix}tenure")
    MonthlyCharges = st.sidebar.slider("Monthly Charges", 18.0, 120.0, 50.0, step=0.5, key=f"{prefix}MonthlyCharges")
    TotalCharges = st.sidebar.number_input("Total Charges", value=float(MonthlyCharges * tenure), key=f"{prefix}TotalCharges")

    return {
        "gender": gender, "SeniorCitizen": SeniorCitizen, "Partner": Partner, "Dependents": Dependents,
        "PhoneService": PhoneService, "MultipleLines": MultipleLines, "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity, "OnlineBackup": OnlineBackup, "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport, "StreamingTV": StreamingTV, "StreamingMovies": StreamingMovies,
        "Contract": Contract, "PaperlessBilling": PaperlessBilling, "PaymentMethod": PaymentMethod,
        "tenure": tenure, "MonthlyCharges": MonthlyCharges, "TotalCharges": TotalCharges
    }

# App Navigation
page = st.sidebar.selectbox("Navigate", ["Single Prediction", "What-If Simulator", "Batch Prediction", "Prediction History", "Model Performance"])

if page == "Single Prediction":
    st.title("🔮 Single Customer Prediction")
    st.write("Enter customer details in the sidebar and predict churn risk.")
    
    inputs = sidebar_inputs()
    
    if st.button("Predict Churn"):
        if MODEL_PIPELINE is None:
            st.error("Model pipeline is not loaded.")
        else:
            with st.spinner("Analyzing customer profile..."):
                df_raw = pd.DataFrame([inputs])
                df_eng = engineer_features(df_raw)
                prob = float(MODEL_PIPELINE.predict_proba(df_eng)[0, 1])
                tier, color = get_risk_tier(prob)
                clv = inputs["MonthlyCharges"] * max(1, 72 - inputs["tenure"]) * (1 - prob)
                
                # Save to history
                st.session_state['history'].append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "tenure": inputs["tenure"],
                    "MonthlyCharges": inputs["MonthlyCharges"],
                    "Contract": inputs["Contract"],
                    "churn_prob": prob,
                    "risk_tier": tier,
                    "CLV": clv
                })
                
                # Top Banner
                if prob > 0.5:
                    st.markdown(f'<div style="background-color: {color}; padding: 10px; border-radius: 5px; color: white; text-align: center;"><h2>🚨 Churn Predicted</h2></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div style="background-color: {color}; padding: 10px; border-radius: 5px; color: white; text-align: center;"><h2>✅ Customer Will Stay</h2></div>', unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1: st.metric("Churn Probability", f"{prob*100:.1f}%")
                with col2: st.metric("Risk Tier", tier)
                with col3: st.metric("Retention Probability", f"{(1-prob)*100:.1f}%")
                with col4: st.metric("Estimated CLV", f"${clv:,.2f}")
                
                st.info(f"**Recommendation:** {get_recommendation(tier)}")
                
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    st.plotly_chart(render_gauge(prob), use_container_width=True)
                    
                with col_chart2:
                    if SHAP_EXPLAINER:
                        preprocessor = MODEL_PIPELINE.named_steps['preprocessor']
                        X_trans = preprocessor.transform(df_eng)
                        feature_names = preprocessor.get_feature_names_out()
                        shap_vals = SHAP_EXPLAINER.shap_values(pd.DataFrame(X_trans, columns=feature_names))
                        if isinstance(shap_vals, list): shap_vals = shap_vals[1]
                        
                        clean_names = [f.split('__')[-1] for f in feature_names]
                        contribs = pd.DataFrame({"Feature": clean_names, "Contribution": shap_vals[0]})
                        contribs['abs_contrib'] = contribs['Contribution'].abs()
                        contribs = contribs.sort_values('abs_contrib', ascending=False).head(8)
                        contribs['Color'] = np.where(contribs['Contribution'] > 0, '#FF4B4B', '#00C853')
                        
                        fig = go.Figure(go.Bar(
                            x=contribs['Contribution'],
                            y=contribs['Feature'],
                            orientation='h',
                            marker_color=contribs['Color']
                        ))
                        fig.update_layout(title="Top Feature Contributions (SHAP)", height=300, yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                
                # Bonus 1: Similar Customers
                if not SCORED_DF.empty:
                    st.markdown("### 👥 Customers like this one:")
                    mask = (
                        (SCORED_DF['tenure'].between(inputs['tenure'] - 6, inputs['tenure'] + 6)) &
                        (SCORED_DF['MonthlyCharges'].between(inputs['MonthlyCharges'] - 10, inputs['MonthlyCharges'] + 10))
                    )
                    similar = SCORED_DF[mask]
                    if len(similar) > 0:
                        similar_sample = similar.sample(min(5, len(similar)), random_state=42)
                        churned_count = similar_sample['Churn'].apply(lambda x: 1 if x == 'Yes' else 0).sum()
                        st.write(f"**Customers like this one: {churned_count}/{len(similar_sample)} churned**")
                        st.dataframe(similar_sample[['tenure', 'MonthlyCharges', 'Contract', 'InternetService', 'churn_prob', 'Churn']])
                    else:
                        st.write("No similar customers found in historical data.")

elif page == "What-If Simulator":
    st.title("🎛️ What-If Simulator")
    st.write("Tweak interventions to see real-time impact on churn probability.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Current Profile")
        inputs = sidebar_inputs("wi_")
        df_raw = pd.DataFrame([inputs])
        df_eng = engineer_features(df_raw)
        orig_prob = float(MODEL_PIPELINE.predict_proba(df_eng)[0, 1]) if MODEL_PIPELINE else 0
        st.metric("Current Probability", f"{orig_prob*100:.1f}%")
        
    with col2:
        st.subheader("Simulate Changes")
        new_contract = st.selectbox("Upgrade Contract?", [inputs["Contract"], "One year", "Two year"])
        new_charges = st.slider("Discounted Monthly Charges?", 18.0, 120.0, inputs["MonthlyCharges"], step=0.5)
        new_support = st.selectbox("Add Tech Support?", [inputs["TechSupport"], "Yes"])
        
        sim_inputs = inputs.copy()
        sim_inputs["Contract"] = new_contract
        sim_inputs["MonthlyCharges"] = new_charges
        sim_inputs["TechSupport"] = new_support
        
        df_sim = pd.DataFrame([sim_inputs])
        df_sim_eng = engineer_features(df_sim)
        new_prob = float(MODEL_PIPELINE.predict_proba(df_sim_eng)[0, 1]) if MODEL_PIPELINE else 0
        
    st.markdown("---")
    # Bottom Panel
    col_g1, col_g2, col_m = st.columns([1, 1, 2])
    with col_g1: st.plotly_chart(render_gauge(orig_prob), use_container_width=True)
    with col_g2: st.plotly_chart(render_gauge(new_prob), use_container_width=True)
    
    with col_m:
        st.subheader("Impact Analysis")
        prob_diff = new_prob - orig_prob
        st.metric("Probability Delta", f"{prob_diff*100:+.1f}%", delta=f"{prob_diff*100:+.1f}%", delta_color="inverse")
        
        rem_tenure = max(1, 72 - inputs["tenure"])
        orig_clv = inputs["MonthlyCharges"] * rem_tenure * (1 - orig_prob)
        new_clv = new_charges * rem_tenure * (1 - new_prob)
        clv_diff = new_clv - orig_clv
        
        st.metric("Revenue Impact (CLV)", f"${new_clv:,.2f}", delta=f"${clv_diff:,.2f}")

elif page == "Batch Prediction":
    st.title("📂 Batch Prediction")
    
    @st.cache_data(ttl=300)
    def process_batch(df_upload):
        df_eng = engineer_features(df_upload)
        probs = MODEL_PIPELINE.predict_proba(df_eng)[:, 1]
        
        df_res = df_upload.copy()
        df_res['churn_prob'] = probs
        df_res['churn_prediction'] = df_res['churn_prob'] > 0.5
        df_res['risk_tier'] = pd.cut(df_res['churn_prob'], bins=[-0.1, 0.25, 0.50, 0.75, 1.1], labels=["Low", "Medium", "High", "Critical"])
        
        df_res['expected_rem_tenure'] = (72 - pd.to_numeric(df_res['tenure'], errors='coerce').fillna(0)).clip(lower=1)
        df_res['CLV'] = pd.to_numeric(df_res['MonthlyCharges'], errors='coerce').fillna(0) * df_res['expected_rem_tenure'] * (1 - df_res['churn_prob'])
        
        return df_res

    # Provide template
    template_cols = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod", "tenure", "MonthlyCharges", "TotalCharges"]
    template_df = pd.DataFrame(columns=template_cols)
    csv = template_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Sample Template", data=csv, file_name="batch_template.csv", mime="text/csv")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file and MODEL_PIPELINE:
        with st.spinner("Scoring batch..."):
            df = pd.read_csv(uploaded_file)
            missing = [c for c in template_cols if c not in df.columns]
            if missing:
                st.error(f"Missing columns in CSV: {', '.join(missing)}")
            else:
                scored = process_batch(df)
                
                churners = scored['churn_prediction'].sum()
                total_clv_risk = scored[scored['churn_prediction']]['CLV'].sum()
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Customers", len(scored))
                col2.metric("Predicted Churners", f"{churners} ({churners/len(scored)*100:.1f}%)")
                col3.metric("CLV at Risk", f"${total_clv_risk:,.2f}")
                
                # Risk Tier Bar
                tier_counts = scored['risk_tier'].value_counts()
                fig = go.Figure(go.Bar(x=tier_counts.index, y=tier_counts.values, marker_color=['#00C853', '#FFD700', '#FFA500', '#FF4B4B']))
                fig.update_layout(title="Risk Tier Breakdown", height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                st.write("Top 20 Riskiest Customers:")
                st.dataframe(scored.sort_values('churn_prob', ascending=False).head(20))
                
                csv_out = scored.to_csv(index=False).encode('utf-8')
                st.download_button("Download Scored Results", data=csv_out, file_name=f"churnguard_scored_{int(time.time())}.csv", mime="text/csv")

elif page == "Prediction History":
    st.title("🕰️ Prediction History Log")
    
    if st.session_state['history']:
        hist_df = pd.DataFrame(st.session_state['history'])
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Predictions", len(hist_df))
        col2.metric("Avg Churn Prob", f"{hist_df['churn_prob'].mean()*100:.1f}%")
        col3.metric("High/Critical Risk Count", len(hist_df[hist_df['risk_tier'].isin(['High', 'Critical'])]))
        
        tier_filter = st.selectbox("Filter by Risk Tier", ["All", "Low", "Medium", "High", "Critical"])
        if tier_filter != "All":
            hist_df = hist_df[hist_df['risk_tier'] == tier_filter]
            
        st.dataframe(hist_df)
        
        fig = go.Figure(go.Scatter(x=hist_df['timestamp'], y=hist_df['churn_prob'], mode='lines+markers'))
        fig.update_layout(title="Churn Probability Trend (Session)", yaxis_title="Probability")
        st.plotly_chart(fig, use_container_width=True)
        
        if st.button("Clear History"):
            st.session_state['history'] = []
            st.rerun()
    else:
        st.info("No predictions made in this session yet. Go to 'Single Prediction' to start.")

elif page == "Model Performance":
    st.title("📈 Model Performance Dashboard")
    
    st.markdown("### Executive Metrics (Tuned XGBoost)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ROC-AUC", "0.88")
    col2.metric("F1 Score", "0.65")
    col3.metric("Recall", "0.61")
    col4.metric("Precision", "0.71")
    
    st.markdown("---")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        img_path = os.path.join(os.path.dirname(__file__), 'reports', 'figures', '06_roc_curves.png')
        if os.path.exists(img_path): st.image(img_path, caption="ROC Curve")
    with col_c2:
        img_path = os.path.join(os.path.dirname(__file__), 'reports', 'figures', '07_pr_curves.png')
        if os.path.exists(img_path): st.image(img_path, caption="Precision-Recall Curve")
        
    st.markdown("---")
    col_c3, col_c4 = st.columns(2)
    with col_c3:
        img_path = os.path.join(os.path.dirname(__file__), 'reports', 'figures', 'XGBoost_Tuned_confusion_matrix.png')
        if os.path.exists(img_path): st.image(img_path, caption="Confusion Matrix")
    with col_c4:
        img_path = os.path.join(os.path.dirname(__file__), 'reports', 'figures', '06_shap_global_full.png')
        if os.path.exists(img_path): st.image(img_path, caption="Global Feature Importance (SHAP)")
        
    if not SCORED_DF.empty:
        st.markdown("### Churn Rate by Segment")
        seg_stats = SCORED_DF.groupby('customer_segment').agg({
            'customer_segment': 'count',
            'CLV': 'mean',
            'churn_prob': 'mean'
        }).rename(columns={'customer_segment': 'Count', 'CLV': 'Avg CLV', 'churn_prob': 'Avg Churn Prob'})
        seg_stats['Churn Rate %'] = seg_stats['Avg Churn Prob'] * 100
        st.dataframe(seg_stats.style.format({'Avg CLV': '${:,.2f}', 'Avg Churn Prob': '{:.1%}', 'Churn Rate %': '{:.1f}%'}))
