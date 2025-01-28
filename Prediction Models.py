# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Step 1: Generate Synthetic SKAN 4.0 Data
def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    
    # Generate synthetic data
    data = pd.DataFrame({
        'source_identifier': np.random.choice(['campaign1', 'campaign2', 'campaign3'], num_samples),
        'coarse_conversion': np.random.choice(['low', 'medium', 'high'], num_samples),
        'revenue': np.random.exponential(scale=100, size=num_samples),
        'engagement': np.random.normal(loc=50, scale=10, size=num_samples),
        'install_volume': np.random.randint(50, 1000, size=num_samples),
        'ad_spend': np.random.uniform(100, 1000, size=num_samples),
        'install_date': pd.date_range(start='2023-01-01', periods=num_samples, freq='D'),
        'postback1_revenue': np.random.exponential(scale=50, size=num_samples),
        'postback2_revenue': np.random.exponential(scale=30, size=num_samples),
        'postback3_revenue': np.random.exponential(scale=20, size=num_samples),
        'postback1_engagement': np.random.normal(loc=20, scale=5, size=num_samples),
        'postback2_engagement': np.random.normal(loc=15, scale=5, size=num_samples),
        'postback3_engagement': np.random.normal(loc=10, scale=5, size=num_samples),
    })
    
    return data

# Step 2: Clean and Preprocess Data
def clean_and_preprocess_data(data):
    # Handle missing values
    data.fillna(0, inplace=True)
    
    # Encode categorical variables (source_identifier, coarse_conversion)
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_features = encoder.fit_transform(data[['source_identifier', 'coarse_conversion']])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['source_identifier', 'coarse_conversion']))
    
    # Normalize numerical features (revenue, engagement)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[['revenue', 'engagement']])
    scaled_df = pd.DataFrame(scaled_features, columns=['revenue', 'engagement'])
    
    # Combine all features
    cleaned_data = pd.concat([encoded_df, scaled_df, data[['install_volume', 'ad_spend']]], axis=1)
    
    return cleaned_data, data

# Step 3: Feature Engineering
def feature_engineering(data):
    # Aggregate postback data
    data['total_revenue'] = data['postback1_revenue'] + data['postback2_revenue'] + data['postback3_revenue']
    data['total_engagement'] = data['postback1_engagement'] + data['postback2_engagement'] + data['postback3_engagement']
    
    # Add crowd anonymity tier as a feature
    data['crowd_anonymity_tier'] = np.where(data['install_volume'] > 1000, 3, 
                                            np.where(data['install_volume'] > 100, 2, 1))
    
    # Add time-based features
    data['days_since_install'] = (pd.to_datetime('today') - pd.to_datetime(data['install_date'])).dt.days
    
    return data

# Step 4: Prepare Data for Modeling
def prepare_data(cleaned_data, data):
    # Define features (X) and target variables (y)
    X = cleaned_data
    y_cac = data['ad_spend'] / data['install_volume']  # CAC = Ad Spend / Installs
    y_roas = data['total_revenue'] / data['ad_spend']  # ROAS = Revenue / Ad Spend
    y_ltv = data['total_revenue']  # LTV = Total Revenue (for simplicity)
    
    # Split data into training and testing sets
    X_train, X_test, y_cac_train, y_cac_test = train_test_split(X, y_cac, test_size=0.2, random_state=42)
    _, _, y_roas_train, y_roas_test = train_test_split(X, y_roas, test_size=0.2, random_state=42)
    _, _, y_ltv_train, y_ltv_test = train_test_split(X, y_ltv, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_cac_train, y_cac_test, y_roas_train, y_roas_test, y_ltv_train, y_ltv_test

# Step 5: Train Models
def train_models(X_train, y_cac_train, y_roas_train, y_ltv_train):
    # Train CAC model (Random Forest)
    cac_model = RandomForestRegressor(random_state=42)
    cac_model.fit(X_train, y_cac_train)
    
    # Train ROAS model (XGBoost)
    roas_model = XGBRegressor(random_state=42)
    roas_model.fit(X_train, y_roas_train)
    
    # Train LTV model (XGBoost)
    ltv_model = XGBRegressor(random_state=42)
    ltv_model.fit(X_train, y_ltv_train)
    
    return cac_model, roas_model, ltv_model

# Step 6: Evaluate Models
def evaluate_models(cac_model, roas_model, ltv_model, X_test, y_cac_test, y_roas_test, y_ltv_test):
    # Evaluate CAC model
    cac_pred = cac_model.predict(X_test)
    print("CAC Model Evaluation:")
    print("MSE:", mean_squared_error(y_cac_test, cac_pred))
    print("R-squared:", r2_score(y_cac_test, cac_pred))
    
    # Evaluate ROAS model
    roas_pred = roas_model.predict(X_test)
    print("\nROAS Model Evaluation:")
    print("MSE:", mean_squared_error(y_roas_test, roas_pred))
    print("R-squared:", r2_score(y_roas_test, roas_pred))
    
    # Evaluate LTV model
    ltv_pred = ltv_model.predict(X_test)
    print("\nLTV Model Evaluation:")
    print("MSE:", mean_squared_error(y_ltv_test, ltv_pred))
    print("R-squared:", r2_score(y_ltv_test, ltv_pred))

# Main Workflow
if __name__ == '__main__':
    # Step 1: Generate synthetic data
    synthetic_data = generate_synthetic_data(num_samples=1000)
    
    # Step 2: Clean and preprocess data
    cleaned_data, raw_data = clean_and_preprocess_data(synthetic_data)
    
    # Step 3: Feature engineering
    raw_data = feature_engineering(raw_data)
    
    # Step 4: Prepare data for modeling
    X_train, X_test, y_cac_train, y_cac_test, y_roas_train, y_roas_test, y_ltv_train, y_ltv_test = prepare_data(cleaned_data, raw_data)
    
    # Step 5: Train models
    cac_model, roas_model, ltv_model = train_models(X_train, y_cac_train, y_roas_train, y_ltv_train)
    
    # Step 6: Evaluate models
    evaluate_models(cac_model, roas_model, ltv_model, X_test, y_cac_test, y_roas_test, y_ltv_test)
