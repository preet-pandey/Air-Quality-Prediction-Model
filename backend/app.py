import os
import time
import serial
import joblib
import pandas as pd
import mysql.connector
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_mail import Mail, Message
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-key-123')
CORS(app)

# --- Mail Configuration ---
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER', 'smtp.gmail.com')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT', 587))
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS', 'True') == 'True'
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_USERNAME')
mail = Mail(app)

# --- Database Connection ---
def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host=os.getenv('MYSQL_HOST', 'localhost'),
            user=os.getenv('MYSQL_USER', 'root'),
            password=os.getenv('MYSQL_PASSWORD', ''),
            database=os.getenv('MYSQL_DB', 'air_quality_db')
        )
        return conn
    except Exception as e:
        print(f"Database Connection Error: {e}")
        return None

# --- ML Model Configuration ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'aqi_model_final.pkl')
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- Serial Configuration ---
SERIAL_PORT = os.getenv('SERIAL_PORT', 'COM5')
BAUD_RATE = int(os.getenv('BAUD_RATE', 9600))
ser = None

def init_serial():
    global ser
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"Connected to Arduino on {SERIAL_PORT}")
    except Exception as e:
        print(f"Warning: Arduino not found on {SERIAL_PORT}. Entering Simulation Mode.")
        ser = None

init_serial()

# --- Helper Functions ---
def get_aqi_category(aqi):
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Moderate"
    elif aqi <= 150: return "Poor"
    elif aqi <= 200: return "Unhealthy"
    elif aqi <= 300: return "Severe"
    else: return "Hazardous"

def send_aqi_alert(user_email, aqi_value, category):
    try:
        msg = Message("AQI Alert: Poor Air Quality Detected",
                      recipients=[user_email])
        
        advice = {
            "Good": "Air quality is satisfactory. No health risk.",
            "Moderate": "Air quality is acceptable. Sensitive groups should monitor.",
            "Poor": "Members of sensitive groups may experience health effects.",
            "Unhealthy": "Everyone may begin to experience health effects.",
            "Severe": "Health warnings of emergency conditions. The entire population is more likely to be affected.",
            "Hazardous": "Health alert: everyone may experience more serious health effects."
        }

        msg.body = f"""
        AQI ALERT!
        -----------
        Value: {aqi_value}
        Category: {category}
        Health Advice: {advice.get(category, "Stay indoors and use air purifiers.")}
        Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}
        """
        mail.send(msg)
        print(f"Alert email sent to {user_email}")
        
        # Log alert in database
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO alerts (user_email, aqi_value, category, message) VALUES (%s, %s, %s, %s)",
                (user_email, aqi_value, category, msg.body)
            )
            conn.commit()
            conn.close()
            
    except Exception as e:
        print(f"Failed to send email: {e}")

def read_arduino_data():
    if ser and ser.is_open:
        try:
            line = ser.readline().decode('utf-8').strip()
            if line:
                parts = line.split(',')
                if len(parts) == 3:
                    return {
                        "gas": float(parts[0]),
                        "temp": float(parts[1]),
                        "humidity": float(parts[2])
                    }
        except Exception as e:
            print(f"Serial Read Error: {e}")
    
    # Simulation fallback
    return {
        "gas": 150.0 + (time.time() % 100),
        "temp": 28.0 + (time.time() % 5),
        "humidity": 60.0 + (time.time() % 10)
    }

# --- API Endpoints ---

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE uid = %s", (data['uid'],))
        user = cursor.fetchone()
        
        if not user:
            cursor.execute(
                "INSERT INTO users (uid, name, email) VALUES (%s, %s, %s)",
                (data['uid'], data['name'], data['email'])
            )
            conn.commit()
        conn.close()
        return jsonify({"status": "Success", "message": "User logged in"}), 200
    return jsonify({"status": "Error", "message": "DB connection failed"}), 500

@app.route('/live-data', methods=['GET'])
def get_live_data():
    sensor_data = read_arduino_data()
    
    features = pd.DataFrame([[sensor_data['gas'], sensor_data['temp'], sensor_data['humidity']]], 
                            columns=['Gas', 'Temperature', 'Humidity'])
    
    prediction = model.predict(features)[0] if model else 0
    category = get_aqi_category(prediction)
    
    # Store reading in DB
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO sensor_readings (gas, temperature, humidity, predicted_aqi, category) VALUES (%s, %s, %s, %s, %s)",
            (sensor_data['gas'], sensor_data['temp'], sensor_data['humidity'], prediction, category)
        )
        conn.commit()
        
        # Check alerts for all active users
        if prediction > 150:
            cursor.execute("SELECT email FROM users WHERE alerts_enabled = TRUE")
            users = cursor.fetchall()
            for user in users:
                send_aqi_alert(user[0], round(prediction, 2), category)
                
        conn.close()
    
    return jsonify({
        "aqi": round(prediction, 2),
        "category": category,
        "temperature": sensor_data['temp'],
        "humidity": sensor_data['humidity'],
        "gas_mq": sensor_data['gas'],
        "status": "Live" if ser else "Simulated"
    })

@app.route('/history', methods=['GET'])
def get_history():
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM sensor_readings ORDER BY timestamp DESC LIMIT 50")
        rows = cursor.fetchall()
        conn.close()
        return jsonify(rows)
    return jsonify([])

@app.route('/predict', methods=['POST'])
def predict_manual():
    try:
        data = request.get_json()
        features = pd.DataFrame([[
            float(data.get('gas', 0)),
            float(data.get('temp', 0)),
            float(data.get('humidity', 0))
        ]], columns=['Gas', 'Temperature', 'Humidity'])
        
        if model:
            prediction = model.predict(features)[0]
            category = get_aqi_category(prediction)
            return jsonify({
                "predicted_aqi": round(prediction, 2),
                "category": category
            })
        return jsonify({"error": "Model not loaded"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/toggle-alerts', methods=['POST'])
def toggle_alerts():
    data = request.get_json()
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE users SET alerts_enabled = %s WHERE uid = %s",
            (data['enabled'], data['uid'])
        )
        conn.commit()
        conn.close()
        return jsonify({"status": "Success"})
    return jsonify({"status": "Error"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
