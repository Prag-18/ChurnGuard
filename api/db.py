import sqlite3
import os
import json
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), 'predictions.db')

def init_db():
    """Create the SQLite database and tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id TEXT UNIQUE,
            timestamp DATETIME,
            input_json TEXT,
            churn_result BOOLEAN,
            probability REAL,
            risk_tier TEXT,
            clv_estimate REAL,
            response_ms INTEGER
        )
    ''')
    conn.commit()
    conn.close()

def log_prediction(prediction_id, input_dict, result_dict, response_ms):
    """Log a single prediction to the audit database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predictions (
            prediction_id, timestamp, input_json, churn_result, 
            probability, risk_tier, clv_estimate, response_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        prediction_id,
        datetime.utcnow().isoformat(),
        json.dumps(input_dict),
        result_dict.get('churn', False),
        result_dict.get('probability', 0.0),
        result_dict.get('risk_tier', 'Unknown'),
        result_dict.get('clv_estimate', 0.0),
        response_ms
    ))
    conn.commit()
    conn.close()

def get_logs(limit=100, risk_tier=None, churn_only=False):
    """Retrieve recent prediction logs with optional filtering."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    query = "SELECT * FROM predictions WHERE 1=1"
    params = []
    
    if risk_tier:
        query += " AND risk_tier = ?"
        params.append(risk_tier)
    
    if churn_only:
        query += " AND churn_result = 1"
        
    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    
    # Convert input_json back to dict for API response
    result = []
    for row in rows:
        d = dict(row)
        d['input_json'] = json.loads(d['input_json'])
        result.append(d)
    return result

def get_stats():
    """Get high-level statistics of the API usage."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*), AVG(probability) FROM predictions")
    total, avg_prob = cursor.fetchone()
    
    cursor.execute("SELECT risk_tier, COUNT(*) FROM predictions GROUP BY risk_tier")
    tiers = {row[0]: row[1] for row in cursor.fetchall()}
    
    conn.close()
    
    return {
        "total_predictions": total or 0,
        "average_probability": round(avg_prob or 0, 4),
        "tier_breakdown": tiers
    }
