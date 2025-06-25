# air_quality_model.py

# 1. Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import logging
import time
from collections import OrderedDict

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('air_quality_model.log', mode='w'),
        logging.StreamHandler()
    ]
)

# Load Dataset
try:
    df = pd.read_csv('city_day.csv')
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    logging.info(f"Dataset loaded with shape: {df.shape}")
except FileNotFoundError:
    print("Error: 'city_day.csv' file not found. Please ensure the file exists in the current directory.")
    logging.error("Dataset file not found")
    exit(1)

# Shuffle and display info
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print("\nDataset Info:")
df.info()

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Handle missing values for numerical columns before calculating stats
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

# Descriptive Stats
print("\nCalculating descriptive statistics...")
stats = []
for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        try:
            mode_val = df[col].mode()[0] if not df[col].mode().empty else None
            numerical_stats = OrderedDict({
                'Feature': col,
                'Minimum': df[col].min(),
                'Maximum': df[col].max(),
                'Mean': df[col].mean(),
                'Mode': mode_val,
                '25%': df[col].quantile(0.25),
                '75%': df[col].quantile(0.75),
                'IQR': df[col].quantile(0.75) - df[col].quantile(0.25),
                'Standard Deviation': df[col].std(),
                'Skewness': df[col].skew(),
                'Kurtosis': df[col].kurt()
            })
            stats.append(numerical_stats)
        except Exception as e:
            print(f"Error calculating stats for {col}: {e}")
            logging.error(f"Error calculating stats for {col}: {e}")

report = pd.DataFrame(stats)
print(f"\nDescriptive statistics calculated for {len(stats)} numerical features")

# Outlier Detection
print("\nDetecting outliers...")
outlier_label = []
for col in report['Feature']:
    try:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        LW = Q1 - 1.5 * IQR
        UW = Q3 + 1.5 * IQR
        outliers = df[(df[col] < LW) | (df[col] > UW)]
        outlier_label.append("Has Outliers" if not outliers.empty else "No Outliers")
    except Exception as e:
        print(f"Error detecting outliers for {col}: {e}")
        outlier_label.append("Error")

report["Outlier Comment"] = outlier_label

# Replace Outliers with Median
print("\nReplacing outliers with median values...")
for col in df.select_dtypes(include=['int64', 'float64']).columns:
    try:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
        
        if outliers.sum() > 0:
            median_value = df[col].median()
            df.loc[outliers, col] = median_value
            print(f"Replaced {outliers.sum()} outliers in '{col}' with median ({median_value:.2f})")
            logging.info(f"Replaced {outliers.sum()} outliers in '{col}' with median")
        else:
            print(f"No outliers found in '{col}'")
    except Exception as e:
        print(f"Error processing outliers for {col}: {e}")
        logging.error(f"Error processing outliers for {col}: {e}")

# Check if AQI column exists
if 'AQI' not in df.columns:
    print("Warning: 'AQI' column not found. Available columns:")
    print(df.columns.tolist())
    # Try to find similar column names
    possible_targets = [col for col in df.columns if 'aqi' in col.lower() or 'quality' in col.lower()]
    if possible_targets:
        print(f"Possible target columns: {possible_targets}")
    exit(1)

# VIF Calculation
print("\nCalculating VIF (Variance Inflation Factor)...")
try:
    numerical_features = df.select_dtypes(include=['int64', 'float64']).drop(columns=['AQI'])
    
    # Handle infinite values
    numerical_features.replace([np.inf, -np.inf], np.nan, inplace=True)
    numerical_features.fillna(numerical_features.median(), inplace=True)
    
    # Remove columns with zero variance
    numerical_features = numerical_features.loc[:, numerical_features.var() != 0]
    
    def calculate_vif(dataset):
        if dataset.empty or dataset.shape[1] == 0:
            return pd.DataFrame(columns=['features', 'VIF_Values'])
        
        vif = pd.DataFrame()
        vif['features'] = dataset.columns
        vif_values = []
        
        for i in range(dataset.shape[1]):
            try:
                vif_val = variance_inflation_factor(dataset.values, i)
                vif_values.append(vif_val)
            except:
                vif_values.append(np.nan)
        
        vif['VIF_Values'] = vif_values
        vif['VIF_Values'] = vif['VIF_Values'].round(2)
        return vif.sort_values(by='VIF_Values', ascending=False)

    vif_report = calculate_vif(numerical_features)
    print(f"VIF calculated for {len(vif_report)} features")
    
except Exception as e:
    print(f"Error calculating VIF: {e}")
    logging.error(f"Error calculating VIF: {e}")
    vif_report = pd.DataFrame(columns=['features', 'VIF_Values'])

# Preprocessing
print("\nStarting preprocessing...")
target_col = 'AQI'
X = df.drop(columns=[target_col])
y = df[target_col]

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Handle categorical variables
categorical_cols = X.select_dtypes(include=['object']).columns
print(f"Categorical columns found: {list(categorical_cols)}")

for col in categorical_cols:
    try:
        if X[col].nunique() > 20:
            # Use LabelEncoder for high cardinality
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            print(f"Label encoded column: {col}")
        else:
            # Use one-hot encoding for low cardinality
            X = pd.get_dummies(X, columns=[col], drop_first=True)
            print(f"One-hot encoded column: {col}")
    except Exception as e:
        print(f"Error encoding column {col}: {e}")
        # Drop problematic columns
        X = X.drop(columns=[col])

# Handle remaining missing values
X.fillna(X.median(numeric_only=True), inplace=True)
for col in X.select_dtypes(include=['object']).columns:
    X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'unknown', inplace=True)

print(f"Final features shape after preprocessing: {X.shape}")

# Feature Scaling
print("\nScaling features...")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
print("Feature scaling completed")

# PCA
print("\nPerforming PCA...")
start = time.time()

try:
    pca_full = PCA()
    pca_full.fit(X_scaled)
    evr = np.cumsum(pca_full.explained_variance_ratio_)
    pcs = np.argmax(evr >= 0.90) + 1

    print(f"Explained Variance Ratio (Cumulative) - First 10 components: {evr[:10]}")
    print(f"Number of Components Selected (90% variance): {pcs}")

    pca = PCA(n_components=pcs)
    pca_data = pca.fit_transform(X_scaled)
    pca_columns = [f'PC{i+1}' for i in range(pcs)]
    pca_df = pd.DataFrame(pca_data, columns=pca_columns)
    pca_df[target_col] = y.reset_index(drop=True)

    end = time.time()
    print(f"PCA completed in {round(end - start, 2)} seconds")
    print(f"Final PCA DataFrame shape: {pca_df.shape}")
    
except Exception as e:
    print(f"Error during PCA: {e}")
    logging.error(f"Error during PCA: {e}")
    # Fallback: use original scaled data
    pca_df = pd.DataFrame(X_scaled, columns=[f'Feature_{i}' for i in range(X_scaled.shape[1])])
    pca_df[target_col] = y.reset_index(drop=True)

# Train-Test Split
print("\nSplitting data for training and testing...")
try:
    X_pca = pca_df.drop(columns=[target_col])
    y_pca = pca_df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y_pca, test_size=0.2, random_state=42
    )
    
    # Create smaller training set for demonstration
    X_train_small, _, y_train_small, _ = train_test_split(
        X_train, y_train, test_size=0.8, random_state=42
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Small training set shape: {X_train_small.shape}")
    
except Exception as e:
    print(f"Error during train-test split: {e}")
    logging.error(f"Error during train-test split: {e}")
    exit(1)

# Random Forest Regressor
print("\nTraining Random Forest Regressor...")
try:
    rf_underfit = RandomForestRegressor(
        n_estimators=10, 
        max_depth=3, 
        max_features='sqrt',  # Changed from 3 to 'sqrt' for better compatibility
        random_state=42
    )
    
    rf_underfit.fit(X_train_small, y_train_small)
    y_pred = rf_underfit.predict(X_test)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n=== Random Forest Evaluation ===")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    logging.info(f"Model performance - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}")
    
    # Feature importance (if available)
    if hasattr(rf_underfit, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X_train_small.columns,
            'importance': rf_underfit.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Feature Importances:")
        print(feature_importance.head(10))

except Exception as e:
    print(f"Error during model training: {e}")
    logging.error(f"Error during model training: {e}")

print("\nScript execution completed!")
logging.info("Script execution completed successfully")