import os
import time
import uuid
import joblib
import pandas as pd
import shap
from flask import Flask, request, jsonify
from flask_cors import CORS
from pydantic import ValidationError
from datetime import datetime

from schemas import CustomerInput, BatchCustomerInput, WhatIfInput
from db import init_db, log_prediction, get_logs, get_stats
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.features import engineer_features

app = Flask(__name__)
CORS(app)

# Global variables for model and explainer
MODEL_PIPELINE = None
SHAP_EXPLAINER = None
START_TIME = time.time()

def load_resources():
    global MODEL_PIPELINE, SHAP_EXPLAINER
    if MODEL_PIPELINE is None:
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'best_model.pkl')
        try:
            MODEL_PIPELINE = joblib.load(model_path)
            # Create TreeExplainer using the XGBoost classifier step
            xgb_model = MODEL_PIPELINE.named_steps['classifier']
            SHAP_EXPLAINER = shap.TreeExplainer(xgb_model)
            print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")

# Initialize DB and model on startup
with app.app_context():
    init_db()
    load_resources()

def get_risk_tier(prob):
    if prob > 0.75: return "Critical"
    elif prob > 0.50: return "High"
    elif prob > 0.25: return "Medium"
    else: return "Low"

def get_recommendation(tier):
    if tier == "Critical": return "Urgent: Personal retention call + 25% discount"
    elif tier == "High": return "Offer annual contract upgrade within 48 hours"
    elif tier == "Medium": return "Send targeted email with loyalty incentive"
    else: return "Upsell opportunity — customer is stable"

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": MODEL_PIPELINE is not None,
        "model_version": "XGBoost-tuned-v1",
        "uptime_seconds": int(time.time() - START_TIME)
    })

@app.route('/predict', methods=['POST'])
def predict():
    start_req = time.time()
    try:
        # 1. Validate Input
        data = request.json
        try:
            validated_data = CustomerInput(**data)
        except ValidationError as e:
            return jsonify({
                "error": "Validation failed",
                "code": "VALIDATION_ERROR",
                "details": e.errors(),
                "timestamp": datetime.utcnow().isoformat()
            }), 422
            
        # 2. Feature Engineering
        df_raw = pd.DataFrame([validated_data.model_dump()])
        df_eng = engineer_features(df_raw)
        
        # 3. Model Prediction
        prob = float(MODEL_PIPELINE.predict_proba(df_eng)[0, 1])
        
        # 4. CLV Calculation
        monthly = validated_data.MonthlyCharges
        tenure = validated_data.tenure
        expected_remaining = max(1, 72 - tenure)
        clv = monthly * expected_remaining * (1 - prob)
        
        # Format result
        tier = get_risk_tier(prob)
        prediction_id = f"pred_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        response_ms = int((time.time() - start_req) * 1000)
        
        result = {
            "churn": prob > 0.5,
            "probability": round(prob, 4),
            "risk_tier": tier,
            "recommendation": get_recommendation(tier),
            "clv_estimate": round(clv, 2),
            "response_ms": response_ms,
            "prediction_id": prediction_id
        }
        
        # 5. Log to SQLite
        log_prediction(prediction_id, validated_data.model_dump(), result, response_ms)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "code": "INTERNAL_ERROR",
            "timestamp": datetime.utcnow().isoformat()
        }), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    start_req = time.time()
    try:
        data = request.json
        try:
            validated_batch = BatchCustomerInput(**data)
        except ValidationError as e:
            return jsonify({
                "error": "Batch validation failed",
                "code": "VALIDATION_ERROR",
                "details": e.errors(),
                "timestamp": datetime.utcnow().isoformat()
            }), 422
            
        customers = [c.model_dump() for c in validated_batch.customers]
        df_raw = pd.DataFrame(customers)
        df_eng = engineer_features(df_raw)
        
        probs = MODEL_PIPELINE.predict_proba(df_eng)[:, 1]
        
        results = []
        total_clv = 0
        churners = 0
        
        for i, prob in enumerate(probs):
            prob = float(prob)
            monthly = customers[i]['MonthlyCharges']
            tenure = customers[i]['tenure']
            expected_remaining = max(1, 72 - tenure)
            clv = monthly * expected_remaining * (1 - prob)
            
            tier = get_risk_tier(prob)
            is_churn = prob > 0.5
            if is_churn:
                churners += 1
                total_clv += clv
                
            results.append({
                "index": i,
                "churn": is_churn,
                "probability": round(prob, 4),
                "risk_tier": tier,
                "clv_estimate": round(clv, 2)
            })
            
        response_ms = int((time.time() - start_req) * 1000)
        
        return jsonify({
            "total": len(customers),
            "churners": churners,
            "churn_rate_pct": round((churners / len(customers)) * 100, 1),
            "total_clv_at_risk": round(total_clv, 2),
            "results": results,
            "processing_ms": response_ms
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "code": "INTERNAL_ERROR",
            "timestamp": datetime.utcnow().isoformat()
        }), 500

@app.route('/admin/logs', methods=['GET'])
def admin_logs():
    limit = int(request.args.get('limit', 100))
    risk_tier = request.args.get('risk_tier')
    churn_only = request.args.get('churn_only', 'false').lower() == 'true'
    
    logs = get_logs(limit=limit, risk_tier=risk_tier, churn_only=churn_only)
    stats = get_stats()
    
    return jsonify({
        "stats": stats,
        "logs": logs
    })

@app.route('/what-if', methods=['POST'])
def what_if():
    try:
        data = request.json
        try:
            validated_data = WhatIfInput(**data)
        except ValidationError as e:
            return jsonify({"error": "Validation failed", "code": "VALIDATION_ERROR"}), 422
            
        orig_cust = validated_data.customer.model_dump()
        df_orig = pd.DataFrame([orig_cust])
        df_orig_eng = engineer_features(df_orig)
        orig_prob = float(MODEL_PIPELINE.predict_proba(df_orig_eng)[0, 1])
        
        monthly = orig_cust['MonthlyCharges']
        tenure = orig_cust['tenure']
        rem_tenure = max(1, 72 - tenure)
        orig_clv = monthly * rem_tenure * (1 - orig_prob)
        
        # Apply interventions
        new_cust = orig_cust.copy()
        interventions = validated_data.interventions.model_dump(exclude_none=True)
        for k, v in interventions.items():
            new_cust[k] = v
            
        df_new = pd.DataFrame([new_cust])
        df_new_eng = engineer_features(df_new)
        new_prob = float(MODEL_PIPELINE.predict_proba(df_new_eng)[0, 1])
        
        new_monthly = new_cust['MonthlyCharges']
        new_clv = new_monthly * rem_tenure * (1 - new_prob)
        
        orig_tier = get_risk_tier(orig_prob)
        new_tier = get_risk_tier(new_prob)
        
        return jsonify({
            "original": {
                "probability": round(orig_prob, 4),
                "risk_tier": orig_tier,
                "clv_estimate": round(orig_clv, 2)
            },
            "simulated": {
                "probability": round(new_prob, 4),
                "risk_tier": new_tier,
                "clv_estimate": round(new_clv, 2)
            },
            "delta": {
                "probability_change": round(new_prob - orig_prob, 4),
                "clv_change": round(new_clv - orig_clv, 2),
                "risk_tier_change": f"{orig_tier} -> {new_tier}"
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e), "code": "INTERNAL_ERROR"}), 500

@app.route('/explain', methods=['POST'])
def explain():
    try:
        data = request.json
        try:
            validated_data = CustomerInput(**data)
        except ValidationError as e:
            return jsonify({"error": "Validation failed"}), 422
            
        df_raw = pd.DataFrame([validated_data.model_dump()])
        df_eng = engineer_features(df_raw)
        
        # SHAP needs the preprocessed data
        preprocessor = MODEL_PIPELINE.named_steps['preprocessor']
        X_trans = preprocessor.transform(df_eng)
        feature_names = preprocessor.get_feature_names_out()
        
        shap_values = SHAP_EXPLAINER.shap_values(pd.DataFrame(X_trans, columns=feature_names))
        if isinstance(shap_values, list):
            shap_values = shap_values[1] # positive class
            
        shap_vals = shap_values[0]
        
        clean_names = [f.split('__')[-1] for f in feature_names]
        
        feature_contributions = []
        for name, val in zip(clean_names, shap_vals):
            feature_contributions.append({
                "feature": name,
                "contribution": float(val),
                "direction": "increases risk" if val > 0 else "decreases risk"
            })
            
        top_5 = sorted(feature_contributions, key=lambda x: abs(x['contribution']), reverse=True)[:5]
        
        return jsonify({"top_5_shap_values": top_5})
        
    except Exception as e:
        return jsonify({"error": str(e), "code": "INTERNAL_ERROR"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
