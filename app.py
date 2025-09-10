from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import sqlite3
import pandas as pd
import json
from datetime import datetime
import uuid

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('diabetes_model.h5')

# Feature names for reference
FEATURE_NAMES = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 
                'insulin', 'bmi', 'diabetes_pedigree', 'age']

def create_database():
    """Create SQLite database and table if not exists"""
    conn = sqlite3.connect('diabetes.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS diabetes_records (
            id TEXT PRIMARY KEY,
            pregnancies INTEGER,
            glucose REAL,
            blood_pressure REAL,
            skin_thickness REAL,
            insulin REAL,
            bmi REAL,
            diabetes_pedigree REAL,
            age INTEGER,
            outcome INTEGER,
            created_at TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

def preprocess_data(data):
    """Preprocess input data for model prediction"""
    # Convert to numpy array and reshape for single prediction
    if isinstance(data, list):
        features = np.array(data).reshape(1, -1)
    else:
        features = np.array([data[col] for col in FEATURE_NAMES]).reshape(1, -1)
    
    return features

def predict_diabetes(features):
    """Make prediction using the loaded model"""
    prediction = model.predict(features)
    probability = float(prediction[0][0])
    predicted_class = int(probability > 0.5)
    
    return {
        'prediction': predicted_class,
        'probability': probability,
        'risk_level': 'High' if predicted_class == 1 else 'Low'
    }

@app.route('/')
def index():
    """Main page with both manual and SQL inference options"""
    return render_template('index.html')

@app.route('/manual_inference', methods=['POST'])
def manual_inference():
    """Endpoint for manual data input inference"""
    try:
        # Get data from form
        data = {
            'pregnancies': float(request.form.get('pregnancies', 0)),
            'glucose': float(request.form.get('glucose', 0)),
            'blood_pressure': float(request.form.get('blood_pressure', 0)),
            'skin_thickness': float(request.form.get('skin_thickness', 0)),
            'insulin': float(request.form.get('insulin', 0)),
            'bmi': float(request.form.get('bmi', 0)),
            'diabetes_pedigree': float(request.form.get('diabetes_pedigree', 0)),
            'age': float(request.form.get('age', 0))
        }
        
        # Preprocess and predict
        features = preprocess_data(data)
        result = predict_diabetes(features)
        
        # Add input data to result
        result['input_data'] = data
        result['method'] = 'Manual Input'
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/sql_inference', methods=['POST'])
def sql_inference():
    """Endpoint for SQL-based inference"""
    try:
        record_id = request.form.get('record_id')
        
        if not record_id:
            return jsonify({
                'success': False,
                'error': 'Record ID is required'
            }), 400
        
        # Connect to database
        conn = sqlite3.connect('diabetes.db')
        cursor = conn.cursor()
        
        # Query the record
        cursor.execute('''
            SELECT pregnancies, glucose, blood_pressure, skin_thickness, 
                   insulin, bmi, diabetes_pedigree, age
            FROM diabetes_records 
            WHERE id = ?
        ''', (record_id,))
        
        record = cursor.fetchone()
        conn.close()
        
        if not record:
            return jsonify({
                'success': False,
                'error': 'Record not found'
            }), 404
        
        # Prepare data for prediction
        data = {
            'pregnancies': record[0],
            'glucose': record[1],
            'blood_pressure': record[2],
            'skin_thickness': record[3],
            'insulin': record[4],
            'bmi': record[5],
            'diabetes_pedigree': record[6],
            'age': record[7]
        }
        
        # Preprocess and predict
        features = preprocess_data(list(record))
        result = predict_diabetes(features)
        
        # Add input data to result
        result['input_data'] = data
        result['method'] = 'SQL Query'
        result['record_id'] = record_id
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/inference', methods=['POST'])
def api_inference():
    """REST API endpoint for inference with JSON input"""
    try:
        data = request.json
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'JSON data is required'
            }), 400
        
        # Validate required fields
        missing_fields = [field for field in FEATURE_NAMES if field not in data]
        if missing_fields:
            return jsonify({
                'success': False,
                'error': f'Missing required fields: {missing_fields}'
            }), 400
        
        # Preprocess and predict
        features = preprocess_data(data)
        result = predict_diabetes(features)
        
        # Add input data to result
        result['input_data'] = data
        result['method'] = 'API'
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/load_sample_data', methods=['POST'])
def load_sample_data():
    """Load sample data from CSV to database"""
    try:
        # Read CSV file
        df = pd.read_csv('diabetes.csv')
        
        # Connect to database
        conn = sqlite3.connect('diabetes.db')
        cursor = conn.cursor()
        
        # Clear existing data
        cursor.execute('DELETE FROM diabetes_records')
        
        # Insert sample data
        for _, row in df.head(10).iterrows():  # Load first 10 records
            record_id = str(uuid.uuid4())
            cursor.execute('''
                INSERT INTO diabetes_records 
                (id, pregnancies, glucose, blood_pressure, skin_thickness, 
                 insulin, bmi, diabetes_pedigree, age, outcome, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                record_id,
                int(row['Pregnancies']),
                float(row['Glucose']),
                float(row['BloodPressure']),
                float(row['SkinThickness']),
                float(row['Insulin']),
                float(row['BMI']),
                float(row['DiabetesPedigreeFunction']),
                int(row['Age']),
                int(row['Outcome']),
                datetime.now()
            ))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': 'Sample data loaded successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/get_records')
def get_records():
    """Get all records from database"""
    try:
        conn = sqlite3.connect('diabetes.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, pregnancies, glucose, blood_pressure, skin_thickness, 
                   insulin, bmi, diabetes_pedigree, age, outcome
            FROM diabetes_records 
            ORDER BY created_at DESC
        ''')
        
        records = cursor.fetchall()
        conn.close()
        
        # Format records
        formatted_records = []
        for record in records:
            formatted_records.append({
                'id': record[0],
                'pregnancies': record[1],
                'glucose': record[2],
                'blood_pressure': record[3],
                'skin_thickness': record[4],
                'insulin': record[5],
                'bmi': record[6],
                'diabetes_pedigree': record[7],
                'age': record[8],
                'outcome': record[9]
            })
        
        return jsonify({
            'success': True,
            'records': formatted_records
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    # Create database on startup
    create_database()
    
    print("🚀 Diabetes Prediction App Starting...")
    print("📊 Model loaded successfully!")
    print("🔗 Access the app at: http://localhost:7575")
    
    app.run(debug=True, host='0.0.0.0', port=7575)
