"""
策略註冊表 (Strategy Registry)
=================================
每個策略 = 一個「產生買賣訊號」的函式 signal_fn(df, threshold) -> df(含 signal 欄)。
回測引擎 (run_backtest) 共用，負責部位管理與損益計算。

要新增策略：寫一個 signal_fn，然後在 STRATEGIES 加一筆即可，介面選單會自動出現。
"""

import pandas as pd


# =========================================================
# Strategy 1: 隔日預測價策略
#   今天看「明天的 predicted」對比「今天的 actual」
#   明天預測 > 今天實際 ×(1+門檻) → BUY
#   明天預測 < 今天實際 ×(1−門檻) → SELL
# =========================================================

def signal_next_day_prediction(df: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    df = df.copy()
    df["next_predicted"] = df["predicted"].shift(-1)
    df["signal"] = None
    df.loc[df["next_predicted"] > df["actual"] * (1 + threshold), "signal"] = "BUY"
    df.loc[df["next_predicted"] < df["actual"] * (1 - threshold), "signal"] = "SELL"
    df.loc[df.index[-1], "signal"] = None   # 最後一天沒有明天的預測
    return df


# =========================================================
# Registry
# =========================================================

STRATEGIES = {
    "next_day_pred": {
        "label": "Next-Day Prediction",
        "signal_fn": signal_next_day_prediction,
        "desc": "明天預測價 vs 今天實際價，價差超過門檻則進出場（每次下單為 portfolio 的 1%）。",
    },
    # 未來新增策略範例：
    # "ma_cross": {
    #     "label": "MA Cross",
    #     "signal_fn": signal_ma_cross,
    #     "desc": "...",
    # },
}

DEFAULT_STRATEGY = "next_day_pred"
