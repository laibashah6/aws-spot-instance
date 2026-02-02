import requests
import json
import pandas as pd
from datetime import datetime
from prophet import Prophet
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix
import numpy as np
import os
import time

# =============================
# CONFIGURATION
# =============================
REGION = "us-west-1"           # More volatile region
INSTANCE_TYPE = "c5.large"     # Volatile instance type
ON_DEMAND_PRICE = 0.0850       # USD/hr (adjust for new instance type)
DATA_FILE = r"E:\6 semester\ML\PythonProject\spot_data.csv"  # absolute path
LOG_FILE = r"E:\6 semester\ML\PythonProject\spot_log.csv"   # absolute path
MIN_ROWS_FOR_PREDICTION = 5
FETCH_INTERVAL = 300  # seconds = 5 minutes

# =============================
# FUNCTIONS
# =============================
def fetch_spot_price():
    url = "https://spot-price.s3.amazonaws.com/spot.js"
    response = requests.get(url)
    text = response.text.replace("callback(", "").rstrip(");")
    data = json.loads(text)
    for region_data in data["config"]["regions"]:
        if region_data["region"] == REGION:
            for inst in region_data["instanceTypes"]:
                for size in inst["sizes"]:
                    if size["size"] == INSTANCE_TYPE:
                        return float(size["valueColumns"][0]["prices"]["USD"])
    return None


def append_to_csv(price, timestamp):
    # Convert timestamp to full string format (YYYY-MM-DD HH:MM:SS.mmm)
    ts_str = timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")

    new_row = pd.DataFrame({"ds": [ts_str], "y": [price]})

    if os.path.exists(DATA_FILE):
        df_existing = pd.read_csv(DATA_FILE)
        # Avoid duplicate timestamps
        if ts_str in df_existing["ds"].values:
            return
        new_row.to_csv(DATA_FILE, mode="a", header=False, index=False)
    else:
        new_row.to_csv(DATA_FILE, index=False)


def append_to_log(timestamp, current_price, predicted_price, decision):
    # Convert timestamp to full string format
    ts_str = timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")

    log_row = pd.DataFrame([{
        "timestamp": ts_str,
        "current_price": current_price,
        "predicted_price": predicted_price,
        "decision": decision
    }])

    if os.path.exists(LOG_FILE):
        df_existing = pd.read_csv(LOG_FILE)
        # Avoid duplicate timestamps
        if ts_str in df_existing["timestamp"].values:
            return
        log_row.to_csv(LOG_FILE, mode="a", header=False, index=False)
    else:
        log_row.to_csv(LOG_FILE, index=False)

def prepare_data():
    df = pd.read_csv(DATA_FILE, usecols=["ds","y"])
    df["ds"] = pd.to_datetime(df["ds"], format="%Y-%m-%d %H:%M:%S.%f")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna()
    df = df.drop_duplicates(subset="ds")  # remove duplicate timestamps
    return df

def feature_engineering(df):
    df["current_price"] = df["y"]
    df["predicted_price"] = df["current_price"].shift(-1)
    df["price_change"] = df["predicted_price"] - df["current_price"]
    df["volatility"] = df["current_price"].rolling(5).std().fillna(0)
    df["spot_vs_ondemand"] = df["current_price"] / ON_DEMAND_PRICE
    df.dropna(inplace=True)
    return df

def label_decision(row):
    if row.current_price < ON_DEMAND_PRICE and row.predicted_price < ON_DEMAND_PRICE and row.volatility < 0.002:
        return "BUY"
    elif row.current_price < ON_DEMAND_PRICE and row.volatility >= 0.002:
        return "WAIT"
    else:
        return "SWITCH"

# =============================
# MAIN LOOP
# =============================
print("Starting Automated Spot Price Fetch & Decision System with Log CSV...\n")

while True:
    timestamp = datetime.now()
    try:
        current_price = fetch_spot_price()
        if current_price is None:
            print(f"{timestamp} - Could not fetch price, retrying...")
            time.sleep(60)
            continue

        print(f"{timestamp} - Current Spot Price: {current_price:.5f}")
        append_to_csv(current_price, timestamp)

        df_raw = prepare_data()
        if len(df_raw) < MIN_ROWS_FOR_PREDICTION:
            print(f"Not enough data yet ({len(df_raw)} rows). Waiting for {MIN_ROWS_FOR_PREDICTION} rows.")
            time.sleep(FETCH_INTERVAL)
            continue

        # Prophet Regression
        model = Prophet()
        model.fit(df_raw)
        future = model.make_future_dataframe(periods=1, freq="min")
        forecast = model.predict(future)
        predicted_price = forecast.iloc[-1]["yhat"]
        print(f"Predicted Spot Price: {predicted_price:.5f}")

        # Feature Engineering
        df = feature_engineering(df_raw)
        df["decision"] = df.apply(label_decision, axis=1)

        # Train Random Forest
        features = ["current_price","predicted_price","price_change","volatility","spot_vs_ondemand"]
        X = df[features]
        y = df["decision"]
        rf = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
        rf.fit(X,y)

        # Make decision for current row
        latest_features = pd.DataFrame([{
            "current_price": current_price,
            "predicted_price": predicted_price,
            "price_change": predicted_price - current_price,
            "volatility": df["volatility"].iloc[-1],
            "spot_vs_ondemand": current_price / ON_DEMAND_PRICE
        }])
        final_decision = rf.predict(latest_features)[0]
        print(f"Decision: {final_decision}")

        # Append to log CSV
        append_to_log(timestamp, current_price, predicted_price, final_decision)

        # Regression metrics
        actual_prices = df["current_price"].values[1:]
        predicted_prices = df["predicted_price"].values[:-1]
        mae = mean_absolute_error(actual_prices, predicted_prices)
        rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
        mape = np.mean(np.abs((actual_prices - predicted_prices)/actual_prices))*100
        print(f"Regression Metrics: MAE={mae:.6f}, RMSE={rmse:.6f}, MAPE={mape:.2f}%")

        # Classification metrics
        y_pred = rf.predict(X)
        cm = confusion_matrix(y,y_pred)
        print("Classification Metrics:")
        print("Confusion Matrix:\n", cm)

    except Exception as e:
        print(f"Error: {e}")

    # Wait before next fetch
    time.sleep(FETCH_INTERVAL)