import pandas as pd
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
from scipy.stats import t


df = pd.read_csv("master_with_year.csv")  # Replace with your actual filename
df.dropna(inplace=True)  # Simple cleanup, adjust as needed

# Define input & target columns
FEATURES = [
    "latitude", "longitude", "PRECTOTCORR", "PS", "QV2M", "T2M", 
    "U10M", "V10M", "sst", "iod", "nao", "ao", "DOY", "year"
]

TARGETS = ["PRECTOTCORR", "T2M", "U10M", "V10M", "QV2M", "PS"]  # Adjust later if needed

def load_trend_data():
    file_path = 'regression_results.pickle' # Replace with the actual path to your .pkl file
    with open(file_path, 'rb') as file:
        return pickle.load(file)
    return None

def train_test():
    # Split into train/test based on year
    df_train = df[df["year"] < 2024]
    df_test = df[df["year"] == 2024]

    X_train = df_train[FEATURES]
    y_train = df_train[TARGETS]

    X_test = df_test[FEATURES]
    y_test = df_test[TARGETS]

    base_model = xgb.XGBRegressor(
        device='cuda',
        objective='reg:squarederror',
        tree_method="hist",
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
    )

    model = MultiOutputRegressor(base_model)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n===== Evaluation Metrics (2024 Test Set) =====\n")

    tolerance = 1.0  # Define tolerance level for accuracy

    for i, target in enumerate(TARGETS):
        true_vals = y_test.iloc[:, i].values
        pred_vals = y_pred[:, i]
        residuals = true_vals - pred_vals

        rmse = np.sqrt(mean_squared_error(true_vals, pred_vals))
        mae = np.mean(np.abs(residuals))
        r2 = r2_score(true_vals, pred_vals)
        accuracy = np.mean(np.abs(residuals) <= tolerance)

        print(f"🔹 Target: {target}")
        print(f"   RMSE = {rmse:.4f}")
        print(f"   MAE  = {mae:.4f}")
        print(f"   R²   = {r2:.4f}")
        print(f"   Accuracy@±{tolerance} = {accuracy*100:.2f}%\n")
    
    return model

def train():
    X = df[FEATURES]
    y = df[TARGETS]

    base_model = xgb.XGBRegressor(
        device='cuda',
        objective='reg:squarederror',
        tree_method="hist",
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
    )

    model = MultiOutputRegressor(base_model)
    model.fit(X, y)

    return model

def save_model(model, filename="xgb_multioutput_model.pkl"):
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def load_model(filename="xgb_multioutput_model.pkl"):
    model = joblib.load(filename)
    print(f"Model loaded from {filename}")
    return model

def predict_from_trends(lat, lon, doy, trend_map):
    # (lat, lon, year, target)

    nearest_keys = sorted(trend_map.keys(), key=lambda x: (abs(x[0] - lat), abs(x[1] - lon)))[:5]
    targets = {}
    for key in nearest_keys:
        trend = trend_map[key]
        for target in trend:
            if target not in targets:
                targets[target] = []
            targets[target].append(trend[target])

    average_targets = {}
    for target in targets:
        average_targets[target] = 0
        for line in targets[target]:
            coeffs = line['coefficients']
            intercept = line['intercept']
            value = coeffs[2] * doy ** 2 + coeffs[1] * doy + intercept
            average_targets[target] += value
        average_targets[target] /= len(targets[target])
        if target in ['PRECTOTCORR', 'QV2M']:
            average_targets[target] = max(0, average_targets[target])
    return average_targets

def build_feature_vector(lat, lon, doy, year, trend_map):
    trend_values = predict_from_trends(lat, lon, doy, trend_map)
    x_input = []
    
    for feat in FEATURES:
        if feat in TARGETS:
            x_input.append(trend_values.get(feat, 0.0))  # use trend
        elif feat == "latitude":
            x_input.append(lat)
        elif feat == "longitude":
            x_input.append(lon)
        elif feat == "DOY":
            x_input.append(doy)
        elif feat == "year":
            x_input.append(year)
        else:
            # fill other features with mean value from the dataset
            x_input.append(df[feat].mean())
    
    return np.array(x_input).reshape(1, -1)

def predict_and_plot_with_uncertainty(trend_map, df, lat, lon, target_year, targets=TARGETS, confidence=0.95):
    # Historical data for residuals (exclude target year)
    hist_df = df[(df['latitude'] == lat) & (df['longitude'] == lon) & (df['year'] != target_year)].sort_values('DOY')
    
    if hist_df.empty:
        print(f"No historical data available for lat={lat}, lon={lon}")
        return
    
    # Trend predictions on historical data
    y_pred_hist = []
    y_true_hist = []
    for _, row in hist_df.iterrows():
        pred = predict_from_trends(lat, lon, row['DOY'], trend_map)
        y_pred_hist.append([pred[t] for t in targets])
        y_true_hist.append([row[t] for t in targets])
    
    y_pred_hist = np.array(y_pred_hist)
    y_true_hist = np.array(y_true_hist)
    
    # Residuals for uncertainty
    residuals = y_true_hist - y_pred_hist
    t_vals = [t.ppf((1 + confidence)/2, df=len(residuals)-1) for _ in targets]
    sigma = [np.std(residuals[:, i], ddof=1) for i in range(len(targets))]
    
    # Predictions for target year
    target_df = df[(df['latitude'] == lat) & (df['longitude'] == lon) & (df['year'] == target_year)].sort_values('DOY')
    
    if target_df.empty:
        print(f"No data found for target year={target_year}")
        return
    
    y_pred_target = []
    for _, row in target_df.iterrows():
        pred = predict_from_trends(lat, lon, row['DOY'], trend_map)
        y_pred_target.append([pred[t] for t in targets])
    y_pred_target = np.array(y_pred_target)
    
    # Plot predicted vs uncertainty
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 10))
    
    for i, target in enumerate(targets):
        lower = y_pred_target[:, i] - t_vals[i] * sigma[i]
        upper = y_pred_target[:, i] + t_vals[i] * sigma[i]
        
        plt.subplot(3, 2, i+1)
        plt.plot(target_df['DOY'], y_pred_target[:, i], marker='x', linestyle='--', label='Predicted')
        plt.fill_between(target_df['DOY'], lower, upper, color='orange', alpha=0.3, label=f'{int(confidence*100)}% Prediction Interval')
        plt.title(f'{target} - Trend Prediction with Uncertainty (lat={lat}, lon={lon}, {target_year})')
        plt.xlabel('Day of Year (DOY)')
        plt.ylabel(target)
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'predictions_with_uncertainty_lat{lat}_lon{lon}_year{target_year}.png')
    plt.show()
    
    return y_pred_target, lower, upper

def predict_with_uncertainty_for_arbitrary_year(trend_map, df, lat, lon, doy_list, target_year, targets=TARGETS, confidence=0.95):
    # Historical data for residuals (exclude target year if exists)
    hist_df = df[(df['latitude'] == lat) & (df['longitude'] == lon)].sort_values('DOY')
    
    if hist_df.empty:
        print(f"No historical data available for lat={lat}, lon={lon}")
        return

    # Compute residuals for each target using historical data
    residuals = []
    for _, row in hist_df.iterrows():
        pred = predict_from_trends(lat, lon, row['DOY'], trend_map)
        residuals.append([row[t] - pred[t] for t in targets])
    
    residuals = np.array(residuals)
    
    # Compute sigma and t-value for confidence interval
    sigma = [np.std(residuals[:, i], ddof=1) for i in range(len(targets))]
    t_vals = [t.ppf((1 + confidence)/2, df=len(residuals)-1) for _ in targets]
    
    # Predictions for target year (arbitrary)
    y_pred_target = []
    lower_bounds = []
    upper_bounds = []
    
    for doy in doy_list:
        pred = predict_from_trends(lat, lon, doy, trend_map)
        y_row = [pred[t] for t in targets]
        y_pred_target.append(y_row)
        
        # Compute prediction intervals
        lower = [y_row[i] - t_vals[i]*sigma[i] for i in range(len(targets))]
        upper = [y_row[i] + t_vals[i]*sigma[i] for i in range(len(targets))]
        lower_bounds.append(lower)
        upper_bounds.append(upper)
    
    # Convert to numpy arrays
    y_pred_target = np.array(y_pred_target)
    lower_bounds = np.array(lower_bounds)
    upper_bounds = np.array(upper_bounds)
    
    # Return as a dictionary for easy access
    result = {
        'DOY': doy_list,
        'Predicted': y_pred_target,
        'Lower': lower_bounds,
        'Upper': upper_bounds
    }
    
    return result


trend_map = load_trend_data()
# train_test()
# save_model(train())
# model = load_model()

# Define Karachi coordinates
lat_karachi = 24.8607
lon_karachi = 67.0011
year_target = 2024

doy_list = list(range(1, 366))
result = predict_with_uncertainty_for_arbitrary_year(trend_map, df, lat_karachi, lon_karachi, doy_list, 2026)

# Example: predicted, lower, upper for PRECTOTCORR on day 50
day_idx = 49  # DOY 50
print(f"Predicted: {result['Predicted'][day_idx, 0]:.2f}")
print(f"Lower: {result['Lower'][day_idx, 0]:.2f}")
print(f"Upper: {result['Upper'][day_idx, 0]:.2f}")
