import numpy as np

def prepare_features(pm25, pm10, no2, so2):
    """
    Prepares features for AQI prediction.
    """
    return np.array([[pm25, pm10, no2, so2]])

