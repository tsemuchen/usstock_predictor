
import os
from typing import List, Dict, Union

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# =========================================================
# Config
# =========================================================
DEFAULT_START_DATE = "2015-01-01"
DEFAULT_SEQ_LENGTH = 10
DEFAULT_MIN_ROWS_REQUIRED = 250


# =========================================================
# Model
# =========================================================
class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


# =========================================================
# Utilities
# =========================================================
def create_sequences(X, y, closes, seq_length=10):
    X_seq, y_seq = [], []
    prev_close_seq, true_next_close_seq = [], []

    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length])
        prev_close_seq.append(closes[i + seq_length - 1])
        true_next_close_seq.append(closes[i + seq_length])

    return (
        np.array(X_seq),
        np.array(y_seq),
        np.array(prev_close_seq),
        np.array(true_next_close_seq),
    )


def create_tabular_sequences(X, y, closes, seq_length=10):
    X_tab, y_tab = [], []
    prev_close_tab, true_next_close_tab = [], []

    for i in range(len(X) - seq_length):
        X_tab.append(X[i:i + seq_length].flatten())
        y_tab.append(y[i + seq_length])
        prev_close_tab.append(closes[i + seq_length - 1])
        true_next_close_tab.append(closes[i + seq_length])

    return (
        np.array(X_tab),
        np.array(y_tab).reshape(-1, 1),
        np.array(prev_close_tab),
        np.array(true_next_close_tab),
    )


def evaluate_price_model(y_true_price, y_pred_price, prev_close, threshold=0.001):
    mae = mean_absolute_error(y_true_price, y_pred_price)
    rmse = np.sqrt(mean_squared_error(y_true_price, y_pred_price))
    mape = np.mean(np.abs((y_true_price - y_pred_price) / y_true_price)) * 100
    r2 = r2_score(y_true_price, y_pred_price)

    actual_direction = np.sign(y_true_price - prev_close)
    pred_direction = np.sign(y_pred_price - prev_close)
    directional_accuracy = np.mean(actual_direction == pred_direction) * 100

    actual_return_from_prev = (y_true_price - prev_close) / prev_close
    pred_return_from_prev = (y_pred_price - prev_close) / prev_close

    actual_dir_thresh = np.where(
        actual_return_from_prev > threshold, 1,
        np.where(actual_return_from_prev < -threshold, -1, 0)
    )
    pred_dir_thresh = np.where(
        pred_return_from_prev > threshold, 1,
        np.where(pred_return_from_prev < -threshold, -1, 0)
    )

    mask = actual_dir_thresh != 0
    directional_accuracy_thresh = (
        np.mean(actual_dir_thresh[mask] == pred_dir_thresh[mask]) * 100
        if mask.sum() > 0 else np.nan
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "R2": float(r2),
        "DA": float(directional_accuracy),
        "DA_thr": float(directional_accuracy_thresh) if np.isfinite(directional_accuracy_thresh) else np.nan,
    }


def _clean_and_engineer(df: pd.DataFrame, start_date: str) -> pd.DataFrame:
    required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Date"] + numeric_cols).reset_index(drop=True)
    df = df.sort_values("Date").reset_index(drop=True)
    df = df[df["Date"] >= start_date].reset_index(drop=True)

    df["Return"] = df["Close"].pct_change()

    df["HL_pct"] = np.where(
        df["Close"] != 0,
        (df["High"] - df["Low"]) / df["Close"],
        np.nan
    )

    df["OC_pct"] = np.where(
        df["Open"] != 0,
        (df["Close"] - df["Open"]) / df["Open"],
        np.nan
    )

    df["Volume_Change"] = df["Volume"].pct_change()

    ma5 = df["Close"].rolling(5).mean()
    ma10 = df["Close"].rolling(10).mean()

    df["MA5_ratio"] = np.where(ma5 != 0, df["Close"] / ma5, np.nan)
    df["MA10_ratio"] = np.where(ma10 != 0, df["Close"] / ma10, np.nan)

    df["Volatility5"] = df["Return"].rolling(5).std()

    df["Return_lag1"] = df["Return"].shift(1)
    df["Return_lag2"] = df["Return"].shift(2)
    df["Return_lag3"] = df["Return"].shift(3)

    # Predict next-day return, then reconstruct price
    df["Target_Return"] = df["Return"] * 100

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=True)
    return df


def run_one_stock(
    file_path: str,
    start_date: str = DEFAULT_START_DATE,
    seq_length: int = DEFAULT_SEQ_LENGTH,
    min_rows_required: int = DEFAULT_MIN_ROWS_REQUIRED,
    device: Union[str, torch.device, None] = None,
) -> Dict:
    symbol = os.path.splitext(os.path.basename(file_path))[0]

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return {"symbol": symbol, "status": f"read_error: {e}"}

    try:
        df = _clean_and_engineer(df, start_date=start_date)
    except Exception as e:
        return {"symbol": symbol, "status": f"feature_error: {e}"}

    if len(df) < min_rows_required:
        return {"symbol": symbol, "status": "too_few_rows_after_cleaning"}

    feature_cols = [
        "Return",
        "HL_pct",
        "OC_pct",
        "Volume_Change",
        "MA5_ratio",
        "MA10_ratio",
        "Volatility5",
        "Return_lag1",
        "Return_lag2",
        "Return_lag3",
    ]

    numeric_block = df[feature_cols + ["Target_Return"]].to_numpy()
    if not np.isfinite(numeric_block).all():
        return {"symbol": symbol, "status": "non_finite_values_after_cleaning"}

    X_all = df[feature_cols].values
    y_all = df[["Target_Return"]].values
    close_all = df["Close"].values
    date_all = df["Date"].values

    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train_raw = X_all[:train_end]
    X_val_raw = X_all[train_end:val_end]
    X_test_raw = X_all[val_end:]

    y_train_raw = y_all[:train_end]
    y_val_raw = y_all[train_end:val_end]
    y_test_raw = y_all[val_end:]

    close_train = close_all[:train_end]
    close_val = close_all[train_end:val_end]
    close_test = close_all[val_end:]

    date_test = df["Date"].values[val_end:]

    if min(len(X_train_raw), len(X_val_raw), len(X_test_raw)) <= seq_length:
        return {"symbol": symbol, "status": "split_too_small_for_sequence"}

    if not np.isfinite(X_train_raw).all() or not np.isfinite(y_train_raw).all():
        return {"symbol": symbol, "status": "non_finite_train_data"}

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train_raw)
    X_val_scaled = scaler_X.transform(X_val_raw)
    X_test_scaled = scaler_X.transform(X_test_raw)

    y_train_scaled = scaler_y.fit_transform(y_train_raw)
    y_val_scaled = scaler_y.transform(y_val_raw)
    y_test_scaled = scaler_y.transform(y_test_raw)

    X_train_seq, y_train_seq, prev_close_train, true_close_train = create_sequences(
        X_train_scaled, y_train_scaled, close_train, seq_length
    )
    X_val_seq, y_val_seq, prev_close_val, true_close_val = create_sequences(
        X_val_scaled, y_val_scaled, close_val, seq_length
    )
    X_test_seq, y_test_seq, prev_close_test, true_close_test = create_sequences(
        X_test_scaled, y_test_scaled, close_test, seq_length
    )

    test_dates_seq = date_test[seq_length:]

    if min(len(X_train_seq), len(X_val_seq), len(X_test_seq)) == 0:
        return {"symbol": symbol, "status": "empty_sequence_set"}

    X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_seq, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_seq, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=32,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val_tensor, y_val_tensor),
        batch_size=64,
        shuffle=False,
    )

    # LSTM
    model = LSTMModel(input_size=len(feature_cols), hidden_size=32).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float("inf")
    best_state = None
    patience = 8
    patience_counter = 0
    epochs = 50

    for _ in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            preds = model(batch_X)
            loss = criterion(preds, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                preds = model(batch_X)
                val_loss_sum += criterion(preds, batch_y).item()

        avg_val_loss = val_loss_sum / len(val_loader)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred_test_scaled_lstm = model(X_test_tensor.to(device)).cpu().numpy()

    pred_test_return_lstm = scaler_y.inverse_transform(pred_test_scaled_lstm).flatten() / 100.0
    y_pred_price_lstm = prev_close_test * (1 + pred_test_return_lstm)

    # XGBoost
    X_train_tab, y_train_tab, _, _ = create_tabular_sequences(
        X_train_scaled, y_train_scaled, close_train, seq_length
    )
    X_val_tab, y_val_tab, _, _ = create_tabular_sequences(
        X_val_scaled, y_val_scaled, close_val, seq_length
    )
    X_test_tab, y_test_tab, prev_close_test_xgb, true_close_test_xgb = create_tabular_sequences(
        X_test_scaled, y_test_scaled, close_test, seq_length
    )

    xgb_model = XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
    )

    xgb_model.fit(
        X_train_tab,
        y_train_tab.ravel(),
        eval_set=[(X_val_tab, y_val_tab.ravel())],
        verbose=False,
    )

    pred_test_scaled_xgb = xgb_model.predict(X_test_tab).reshape(-1, 1)
    pred_test_return_xgb = scaler_y.inverse_transform(pred_test_scaled_xgb).flatten() / 100.0
    y_pred_price_xgb = prev_close_test_xgb * (1 + pred_test_return_xgb)

    y_true_price = true_close_test
    naive_pred_price = prev_close_test

    naive_metrics = evaluate_price_model(y_true_price, naive_pred_price, prev_close_test)
    lstm_metrics = evaluate_price_model(y_true_price, y_pred_price_lstm, prev_close_test)
    xgb_metrics = evaluate_price_model(y_true_price, y_pred_price_xgb, prev_close_test)

    lstm_mae_gain = naive_metrics["MAE"] - lstm_metrics["MAE"]
    xgb_mae_gain = naive_metrics["MAE"] - xgb_metrics["MAE"]

    if lstm_metrics["MAE"] <= xgb_metrics["MAE"]:
        best_model = "LSTM"
        best_metrics = lstm_metrics
    else:
        best_model = "XGBoost"
        best_metrics = xgb_metrics

    # Create prediction dataframe using best model
    if best_model == "LSTM":
        best_pred_price = y_pred_price_lstm
    else:
        best_pred_price = y_pred_price_xgb

    prediction_df = pd.DataFrame({
        "Date": test_dates_seq,
        "actual_price": y_true_price,
        "predicted_price": best_pred_price,
    })

    if lstm_metrics["DA"] >= xgb_metrics["DA"]:
        best_da_model = "LSTM"
        best_da = lstm_metrics["DA"]
    else:
        best_da_model = "XGBoost"
        best_da = xgb_metrics["DA"]

    return {
        "symbol": symbol,
        "status": "ok",
        "prediction_df": prediction_df,
        "test_size": int(len(y_true_price)),

        "naive_mae": naive_metrics["MAE"],
        "naive_rmse": naive_metrics["RMSE"],

        "lstm_mae": lstm_metrics["MAE"],
        "lstm_rmse": lstm_metrics["RMSE"],
        "lstm_mape": lstm_metrics["MAPE"],
        "lstm_r2": lstm_metrics["R2"],
        "lstm_da": lstm_metrics["DA"],
        "lstm_da_thr": lstm_metrics["DA_thr"],
        "lstm_vs_naive_mae_gain": lstm_mae_gain,

        "xgb_mae": xgb_metrics["MAE"],
        "xgb_rmse": xgb_metrics["RMSE"],
        "xgb_mape": xgb_metrics["MAPE"],
        "xgb_r2": xgb_metrics["R2"],
        "xgb_da": xgb_metrics["DA"],
        "xgb_da_thr": xgb_metrics["DA_thr"],
        "xgb_vs_naive_mae_gain": xgb_mae_gain,

        "best_model": best_model,
        "best_model_mae": best_metrics["MAE"],
        "best_model_rmse": best_metrics["RMSE"],
        "best_model_mape": best_metrics["MAPE"],
        "best_model_r2": best_metrics["R2"],
        "best_model_da": best_metrics["DA"],
        "best_model_da_thr": best_metrics["DA_thr"],
        "best_model_vs_naive_mae_gain": max(lstm_mae_gain, xgb_mae_gain),

        "best_da_model": best_da_model,
        "best_da": best_da,
    }


def run_many_stocks(
    file_paths: List[str],
    start_date: str = DEFAULT_START_DATE,
    seq_length: int = DEFAULT_SEQ_LENGTH,
    min_rows_required: int = DEFAULT_MIN_ROWS_REQUIRED,
    device: Union[str, torch.device, None] = None,
) -> pd.DataFrame:
    results = []

    for idx, file_path in enumerate(file_paths, start=1):
        print(f"[{idx}/{len(file_paths)}] Running {os.path.basename(file_path)} ...")
        try:
            result = run_one_stock(
                file_path=file_path,
                start_date=start_date,
                seq_length=seq_length,
                min_rows_required=min_rows_required,
                device=device,
            )
        except Exception as e:
            symbol = os.path.splitext(os.path.basename(file_path))[0]
            result = {"symbol": symbol, "status": f"runtime_error: {e}"}
        results.append(result)

    return pd.DataFrame(results)
