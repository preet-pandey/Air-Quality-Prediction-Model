import joblib
from src.data_loader import prepare_features

# Load model
model = joblib.load("aqi_random_forest_model.pkl")

try:
    scaler = joblib.load("scaler.pkl")
    use_scaler = True
except:
    scaler = None
    use_scaler = False

def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"

def predict_aqi(pm25, pm10, no2, so2):
    features = prepare_features(pm25, pm10, no2, so2)
    
    if use_scaler:
        features = scaler.transform(features)
        
    prediction = model.predict(features)[0]
    category = get_aqi_category(prediction)
    
    return round(prediction, 2), category

