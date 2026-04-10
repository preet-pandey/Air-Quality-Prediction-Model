
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

def train_and_save_model():
    # 1. Create a synthetic dataset that maps Gas, Temp, and Humidity to an AQI value
    # In a real scenario, you would use your own collected sensor data here.
    np.random.seed(42)
    num_samples = 1000
    
    # Gas (MQ sensor reading, e.g., 0-1023 or ppm)
    gas = np.random.uniform(50, 800, num_samples)
    # Temperature (Celsius)
    temp = np.random.uniform(15, 45, num_samples)
    # Humidity (%)
    humidity = np.random.uniform(30, 90, num_samples)
    
    # Simple formula to simulate AQI based on these factors:
    # High gas increases AQI significantly.
    # High temp and humidity can slightly increase AQI (stagnant air).
    aqi = (gas * 0.4) + (temp * 0.5) + (humidity * 0.2) + np.random.normal(0, 5, num_samples)
    
    df = pd.DataFrame({
        'Gas': gas,
        'Temperature': temp,
        'Humidity': humidity,
        'AQI': aqi
    })
    
    # 2. Train the Random Forest Model
    X = df[['Gas', 'Temperature', 'Humidity']]
    y = df['AQI']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # 3. Save the model
    model_path = os.path.join(os.path.dirname(__file__), 'aqi_model_final.pkl')
    joblib.dump(model, model_path)
    print(f"Success: New ML model trained and saved to {model_path}")
    print("Features used: ['Gas', 'Temperature', 'Humidity']")

if __name__ == "__main__":
    train_and_save_model()
