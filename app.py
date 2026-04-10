# app.py

import gradio as gr
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("aqi_random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")  # comment out if not used

# Default PCA values (PC4-PC13)
default_values = {
    'PC4': -1.23,
    'PC5': 0.56,
    'PC6': -0.89,
    'PC7': 1.02,
    'PC8': 0.13,
    'PC9': -0.44,
    'PC10': 0.77,
    'PC11': 0.08,
    'PC12': -0.11,
    'PC13': 0.33
}

# AQI Category function
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

# Prediction logic
def predict_aqi(pc1, pc2, pc3):
    full_input = [pc1, pc2, pc3] + [default_values[f'PC{i}'] for i in range(4, 14)]
    features = np.array(full_input).reshape(1, -1)
    scaled = scaler.transform(features)
    prediction = model.predict(scaled)[0]
    category = get_aqi_category(prediction)
    return f"{prediction:.2f}", category

# Gradio UI
iface = gr.Interface(
    fn=predict_aqi,
    inputs=[
        gr.Number(label="PC1", value=0.0),
        gr.Number(label="PC2", value=0.0),
        gr.Number(label="PC3", value=0.0)
    ],
    outputs=[
        gr.Textbox(label="Predicted AQI"),
        gr.Textbox(label="AQI Category")
    ],
    title="AQI Predictor App",
    description="Enter PC1-PC3 values to get predicted AQI and its environmental category using Random Forest."
)

iface.launch()
