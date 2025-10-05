import pandas as pd
import xgboost as xgb
from typing import List, Dict, Any
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib
import pickle
from scipy.stats import t, norm


df = pd.read_csv("master_with_year.csv")
df.dropna(inplace=True)

FEATURES = [
    "latitude", "longitude", "PRECTOTCORR", "PS", "QV2M", "T2M", 
    "U10M", "V10M", "sst", "iod", "nao", "ao", "DOY", "year"
]

TARGETS = ["PRECTOTCORR", "T2M", "U10M", "V10M", "QV2M", "PS"]

def load_trend_data():
    file_path = 'regression_results.pickle'
    with open(file_path, 'rb') as file:
        return pickle.load(file)
    return None

def load_residuals():
    file_path = 'residuals_result.pickle'
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

def predict_with_model(lat, lon, doy, year, trend_map, model):
    x_input = build_feature_vector(lat, lon, doy, year, trend_map)
    y_pred = model.predict(x_input)
    return dict(zip(TARGETS, y_pred[0]))

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

trend_map = load_trend_data()
# train_test()
# save_model(train())
# model = load_model()

DEFAULT_TARGETS = ['PRECTOTCORR', 'PS', 'QV2M', 'T2M', 'U10M', 'V10M']

DEFAULT_THRESHOLDS = {
    'PRECTOTCORR': 10.0,   # rain > 30 mm
    'T2M': 28.0,           # temperature > 35 C
    'QV2M': 20.0,          # humidity > 12 (units used in dataset)
    'wind_speed': 5.0     # wind speed > 10 m/s (derived from U10M & V10M)
}

def _wind_from_uv(u, v):
    speed = np.sqrt(u**2 + v**2)
    # Use vectorized arctan2; handle single values and arrays:
    direction = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
    return speed, direction

def compute_location_residuals_all_years(
    trend_map,
    model,
    df: pd.DataFrame,
    lat: float,
    lon: float,
    targets: List[str]=DEFAULT_TARGETS
):
    # find nearest available (lat, lon) in the dataset
    df['dist'] = np.sqrt((df['latitude'] - lat)**2 + (df['longitude'] - lon)**2)
    nearest = df.loc[df['dist'].idxmin()]
    nearest_lat, nearest_lon = nearest['latitude'], nearest['longitude']

    hist_df = df[(df['latitude'] == nearest_lat) & (df['longitude'] == nearest_lon)].sort_values('DOY')
    if hist_df.empty:
        return None

    residuals_raw = []
    wind_residuals = []  # will store observed_speed - predicted_speed

    for _, row in hist_df.iterrows():
        pred = predict_with_model(lat, lon, row['DOY'], row['year'], trend_map, model)
        # raw residuals: observed - predicted for each target
        residuals_raw.append([row[t] - pred[t] for t in targets])

        # derived wind: observed speed vs predicted speed
        obs_u = row['U10M']
        obs_v = row['V10M']
        obs_speed, _ = _wind_from_uv(obs_u, obs_v)

        pred_u = pred['U10M']
        pred_v = pred['V10M']
        pred_speed, _ = _wind_from_uv(pred_u, pred_v)

        wind_residuals.append(obs_speed - pred_speed)

    residuals_raw = np.array(residuals_raw)   # shape (n_samples, n_targets)
    wind_residuals = np.array(wind_residuals) # shape (n_samples,)
    return residuals_raw, wind_residuals, len(hist_df)


def find_residuals_for_location(lat, lon, residuals):
    df['dist'] = np.sqrt((df['latitude'] - lat)**2 + (df['longitude'] - lon)**2)
    nearest = df.loc[df['dist'].idxmin()]
    nearest_lat, nearest_lon = nearest['latitude'], nearest['longitude']

    return residuals[(nearest_lat, nearest_lon)] if (nearest_lat, nearest_lon) in residuals else None

def predict_with_full_stats_and_probs_single(
    trend_map,
    model,
    df: pd.DataFrame,
    lat: float,
    lon: float,
    doy: int,
    year: int,
    targets: List[str]=DEFAULT_TARGETS,
    thresholds: Dict[str, float]=None,
    confidence: float=0.95,
    rr=None
) -> Dict[str, Any]:
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS.copy()

    if rr is None:
        rr = compute_location_residuals_all_years(trend_map, model, df, lat, lon, targets)
        if rr is None:
            raise ValueError(f"No historical data found at lat={lat}, lon={lon}")

    residuals_raw, wind_residuals, n_samples = rr

    residual_mean = np.mean(residuals_raw, axis=0)
    residual_std = np.std(residuals_raw, axis=0, ddof=1)

    wind_res_mean = np.mean(wind_residuals)
    wind_res_std = np.std(wind_residuals, ddof=1)

    dfree = max(n_samples - 1, 1)
    t_val = t.ppf((1 + confidence) / 2.0, df=dfree)

    # --- Prediction ---
    pred = predict_with_model(lat, lon, doy, year, trend_map, model)
    y_row = np.array([pred[t] for t in targets], dtype=float)

    lower = y_row - t_val * residual_std
    upper = y_row + t_val * residual_std

    # Derived wind
    pred_u = pred['U10M']
    pred_v = pred['V10M']
    speed, direction = _wind_from_uv(pred_u, pred_v)
    lower_w = speed - t_val * wind_res_std
    upper_w = speed + t_val * wind_res_std

    # Probability calculations
    idx_map = {t: i for i, t in enumerate(targets)}
    rain_idx = idx_map.get('PRECTOTCORR')
    temp_idx = idx_map.get('T2M')
    hum_idx = idx_map.get('QV2M')

    # Rain
    mean_rain = y_row[rain_idx]
    sigma_rain = residual_std[rain_idx] if residual_std[rain_idx] > 1e-6 else 1e-6
    thr_rain = thresholds.get('PRECTOTCORR', 30.0)
    prob_rain = 1.0 - norm.cdf(thr_rain, loc=mean_rain, scale=sigma_rain)

    # Heavy Rain
    thr_heavy = max(50.0, thresholds.get('PRECTOTCORR', 30.0))
    prob_heavy_rain = 1.0 - norm.cdf(thr_heavy, loc=mean_rain, scale=sigma_rain)

    # Temperature
    mean_temp = y_row[temp_idx]
    sigma_temp = residual_std[temp_idx] if residual_std[temp_idx] > 1e-6 else 1e-6
    thr_t = thresholds.get('T2M', 35.0)
    prob_temp = 1.0 - norm.cdf(thr_t, loc=mean_temp, scale=sigma_temp)

    # Humidity
    mean_q = y_row[hum_idx]
    sigma_q = residual_std[hum_idx] if residual_std[hum_idx] > 1e-6 else 1e-6
    thr_q = thresholds.get('QV2M', 12.0)
    prob_humidity = 1.0 - norm.cdf(thr_q, loc=mean_q, scale=sigma_q)

    # Wind
    sigma_ws = wind_res_std if wind_res_std > 1e-6 else 1e-6
    thr_ws = thresholds.get('wind_speed', 10.0)
    prob_wind_speed = 1.0 - norm.cdf(thr_ws, loc=speed, scale=sigma_ws)

    # Heatwave
    mean_hi = mean_temp + 0.5 * (mean_q / 10.0)
    sigma_hi = np.sqrt((sigma_temp**2) + ((0.5/10.0)**2) * (sigma_q**2))
    prob_heatwave = 1.0 - norm.cdf(40.0, loc=mean_hi, scale=sigma_hi)

    return {
        "DOY": doy,
        "Targets": targets,
        "Predicted": y_row,
        "Lower": lower,
        "Upper": upper,
        "Derived": {
            "wind_speed": speed,
            "wind_dir": direction
        },
        "Derived_Lower": {
            "wind_speed": lower_w
        },
        "Derived_Upper": {
            "wind_speed": upper_w
        },
        "Probability": {
            "rain_gt_threshold": prob_rain,
            "heavy_rain_gt_50mm": prob_heavy_rain,
            "temp_gt_threshold": prob_temp,
            "humidity_gt_threshold": prob_humidity,
            "wind_speed_gt_threshold": prob_wind_speed,
            "heatwave_hi_gt_40C": prob_heatwave
        },
        "Residual_Mean": residual_mean,
        "Residual_STD": residual_std,
        "Residual_Mean_WindSpeed": wind_res_mean,
        "Residual_STD_WindSpeed": wind_res_std,
        "confidence": confidence,
        "thresholds_used": thresholds
    }

if __name__ == "__main__":
    import pickle
    import os
    try:
        with open("regression_results.pickle", "rb") as f:
            trend_map = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("regression_results.pickle not found. Place your trend pickle in the same folder or call the function from your code passing trend_map directly.")

    if not os.path.exists("master_with_year.csv"):
        raise FileNotFoundError("master_with_year.csv not found in folder. Put your dataframe CSV here or call the function passing the df object.")

    df = pd.read_csv("master_with_year.csv")
    df.dropna(inplace=True)

    # single test day
    lat_karachi = 24
    lon_karachi = 67
    doy_test = 233
    year = 2025

    model = load_model("xgb_multioutput_model.pkl")
    residuals = load_residuals()
    rr = find_residuals_for_location(lat_karachi, lon_karachi, residuals)

    result = predict_with_full_stats_and_probs_single(
        trend_map,
        model,
        df,
        lat_karachi,
        lon_karachi,
        doy_test,
        year,
        thresholds={
            'PRECTOTCORR': 10.0,
            'T2M': 29.0,
            'QV2M': 15.0,
            'wind_speed': 5.0
        },
        rr=rr
    )

    print(result)

