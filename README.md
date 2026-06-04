# Stock Prediction Pipeline

## Files

### `selected_tickers.txt`

List the tickers you want to run.

Example:

```text
AAPL
MSFT
NVDA
GOOG
AMZN
```

---

### `get_kline.py`

Downloads/updates daily stock data from Yahoo Finance.

Output:

```text
selected_daily_kline/
```

---

### `stock_predictor.py`

Core prediction logic.

Contains:

- Feature engineering
- LSTM model
- XGBoost model
- Evaluation
- Prediction generation

Normally you do **not** run this file directly.

---

### `update_and_predict.py`

Main entry point.

Workflow:

```text
selected_tickers.txt
        ↓
get_kline.py
        ↓
stock_predictor.py
        ↓
prediction/backtest files
```

---

## How to Run

### Generate prediction files

```bash
python update_and_predict.py --output-mode prediction
```

Output:

```text
pred_price/
├── AAPL_pred.csv
├── MSFT_pred.csv
└── ...
```

---

### Generate backtest files

```bash
python update_and_predict.py --output-mode backtest
```

Output:

```text
backtest_files/
├── AAPL_backtest.csv
├── MSFT_backtest.csv
└── ...
```

---

## Output Format

```csv
Date,actual_price,predicted_price
2026-05-29,516.10,518.33
```

Meaning:

- `Date` = target trading day
- `actual_price` = actual close price on that day
- `predicted_price` = model prediction for that day

---

## Useful Options

### Use a different ticker file

```bash
python update_and_predict.py --ticker-file my_tickers.txt
```

### Change prediction start date

```bash
python update_and_predict.py --start-date 2018-01-01
```

### Change sequence length

```bash
python update_and_predict.py --seq-length 20
```

### Save summary metrics

```bash
python update_and_predict.py --save-results results.csv
```

---

## Typical Workflow

1. Edit `selected_tickers.txt`

```text
AAPL
MSFT
NVDA
```

2. Run

```bash
python update_and_predict.py --output-mode backtest
```

3. Use generated files

```text
backtest_files/AAPL_backtest.csv
backtest_files/MSFT_backtest.csv
backtest_files/NVDA_backtest.csv
```

for downstream backtesting.