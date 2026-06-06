import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# =========================================================
# Config
# =========================================================

INITIAL_CASH = 10_000
DATA_DIR = Path(__file__).parent
OUTPUT_DIR = DATA_DIR / "backtest_results"
OUTPUT_DIR.mkdir(exist_ok=True)

# =========================================================
# Load prediction CSV
# =========================================================

def load_pred(ticker: str) -> pd.DataFrame:
    f = DATA_DIR / f"{ticker}_backtest.csv"
    df = pd.read_csv(f)
    df = df.rename(columns={"Date": "date", "actual_price": "actual", "predicted_price": "predicted"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df

# =========================================================
# Generate signals
# Signal on day t is based on: predicted[t+1] vs actual[t]
#   predicted[t+1] > actual[t]  →  BUY 1 share at actual[t]
#   predicted[t+1] < actual[t]  →  SELL 1 share at actual[t]
# =========================================================

# 訊號邏輯已移到 strategies.py（策略註冊表）。
# 保留 generate_signals 別名以相容舊呼叫，預設使用隔日預測價策略。
from strategies import signal_next_day_prediction as generate_signals

# =========================================================
# Backtest engine
# =========================================================

def run_backtest(
    df: pd.DataFrame,
    initial_cash: float = INITIAL_CASH,
    position_pct: float = 0.01,
) -> pd.DataFrame:
    """
    每次交易金額 = 當前 portfolio 總值 (現金 + 持股市值) 的 position_pct。
    使用碎股 (fractional shares)，買賣金額不足整股時仍可成交。
    """
    cash = initial_cash
    shares = 0.0
    records = []

    for _, row in df.iterrows():
        price = row["actual"]
        signal = row["signal"]
        trade = ""

        total_before = cash + shares * price
        target_value = total_before * position_pct

        if signal == "BUY":
            spend = min(target_value, cash)          # 受限於現金
            if spend > 0:
                cash -= spend
                shares += spend / price
                trade = "BUY"
            else:
                trade = "BUY_FAILED"

        elif signal == "SELL":
            proceeds = min(target_value, shares * price)  # 受限於持股市值
            if proceeds > 0:
                cash += proceeds
                shares -= proceeds / price
                trade = "SELL"
            else:
                trade = "SELL_FAILED"

        total = cash + shares * price

        records.append({
            "date": row["date"],
            "actual": round(price, 4),
            "predicted": round(row["predicted"], 4),
            "signal": signal if signal else "",
            "trade": trade,
            "cash": round(cash, 2),
            "shares": round(shares, 4),
            "stock_value": round(shares * price, 2),
            "total": round(total, 2),
        })

    return pd.DataFrame(records)

# =========================================================
# Metrics
# =========================================================

def calc_metrics(result: pd.DataFrame) -> dict:
    equity = result["total"]
    daily_ret = equity.pct_change().fillna(0)

    total_return = equity.iloc[-1] / INITIAL_CASH - 1
    mdd = (equity / equity.cummax() - 1).min()
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)
              if daily_ret.std() > 0 else 0)

    trades = result[result["trade"].isin(["BUY", "SELL"])]
    n_buy = (result["trade"] == "BUY").sum()
    n_sell = (result["trade"] == "SELL").sum()

    return {
        "total_return": total_return,
        "final_balance": equity.iloc[-1],
        "sharpe": sharpe,
        "mdd": mdd,
        "n_buy": int(n_buy),
        "n_sell": int(n_sell),
        "final_shares": round(float(result["shares"].iloc[-1]), 2),
    }

# =========================================================
# Plot
# =========================================================

def plot_equity(result: pd.DataFrame, ticker: str):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(result["date"], result["total"], linewidth=1.5, label="Portfolio")
    ax.axhline(INITIAL_CASH, color="gray", linestyle="--", linewidth=1, label="Initial")
    ax.set_title(f"{ticker} — ML Backtest Equity Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{ticker}_equity.png")
    plt.close(fig)

# =========================================================
# Run single ticker
# =========================================================

def backtest_ticker(ticker: str) -> dict:
    df = load_pred(ticker)
    df = generate_signals(df)
    result = run_backtest(df)
    metrics = calc_metrics(result)
    result.to_csv(OUTPUT_DIR / f"{ticker}_balance.csv", index=False)
    plot_equity(result, ticker)
    return metrics

# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    tickers = [
        f.stem.replace("_backtest", "")
        for f in DATA_DIR.glob("*_backtest.csv")
    ]
    tickers.sort()

    print(f"{'Ticker':<8} {'Return':>8} {'Final $':>10} {'Sharpe':>7} {'MDD':>8} {'Buys':>6} {'Sells':>6} {'Shares':>7}")
    print("-" * 65)

    summary_rows = []

    for ticker in tickers:
        try:
            m = backtest_ticker(ticker)
            print(
                f"{ticker:<8} "
                f"{m['total_return']:>8.2%} "
                f"{m['final_balance']:>10.2f} "
                f"{m['sharpe']:>7.2f} "
                f"{m['mdd']:>8.2%} "
                f"{m['n_buy']:>6} "
                f"{m['n_sell']:>6} "
                f"{m['final_shares']:>7}"
            )
            summary_rows.append({"ticker": ticker, **m})
        except FileNotFoundError:
            print(f"{ticker:<8} (file not found)")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    print(f"\nResults saved to: {OUTPUT_DIR}/")
