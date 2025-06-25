import pandas as pd
import numpy as np
import warnings
from collections import OrderedDict
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# Logging Setup
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filemode='w',
                    filename='air_quality_model.log',
                    force=True)

def load_data(path='city_day.csv'):
    df = pd.read_csv(path)
    return df.sample(frac=1).reset_index(drop=True)  # Shuffle

def handle_outliers(df):
    for col in df.select_dtypes(include='number').columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = (df[col] < lower) | (df[col] > upper)
        if outliers.sum() > 0:
            df.loc[outliers, col] = df[col].median()
    return df

def preprocess_data(df, target_col='AQI'):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    for col in X.select_dtypes(include='object').columns:
        if X[col].nunique() > 20:
            X[col] = LabelEncoder().fit_transform(X[col])
        else:
            X = pd.get_dummies(X, columns=[col], drop_first=True)

    X.fillna(X.mean(), inplace=True)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA()
    pca.fit(X_scaled)
    evr = np.cumsum(pca.explained_variance_ratio_)
    pcs = np.argmax(evr >= 0.90) + 1

    logging.info(f"Number of PCA components selected: {pcs}")

    pca = PCA(n_components=pcs)
    X_pca = pca.fit_transform(X_scaled)

    pca_columns = [f'PC{i+1}' for i in range(pcs)]
    pca_df = pd.DataFrame(X_pca, columns=pca_columns)
    pca_df[target_col] = y.reset_index(drop=True)
    return pca_df

def train_and_evaluate_rf(df):
    df = df.dropna(subset=['AQI'])
    X = df.drop(columns='AQI')
    y = df['AQI']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"Random Forest → R²: {r2:.3f}, RMSE: {rmse:.2f}")
    logging.info(f"Random Forest Evaluation → R²: {r2:.3f}, RMSE: {rmse:.2f}")

if __name__ == "__main__":
    df = load_data()
    df = handle_outliers(df)
    pca_df = preprocess_data(df)
    train_and_evaluate_rf(pca_df)
