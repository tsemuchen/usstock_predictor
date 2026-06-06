import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from ml_backtest import load_pred, run_backtest, calc_metrics
from strategies import STRATEGIES, DEFAULT_STRATEGY

# =========================================================
# Config
# =========================================================

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "backtest_results"
INITIAL_CASH = 10_000

st.set_page_config(
    page_title="ML Stock Backtest",
    page_icon="📈",
    layout="wide",
)

# Trim the large default top padding
st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; }
      header[data-testid="stHeader"] { height: 0; }
      section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
      /* shrink metric numbers so full values fit inside the cards */
      [data-testid="stMetricValue"] { font-size: 1.3rem; }
      [data-testid="stMetricLabel"] p { font-size: 0.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# Helpers
# =========================================================

@st.cache_data
def load_summary() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / "summary.csv")
    return df

@st.cache_data
def get_balance(ticker: str, strategy_key: str, threshold: float, initial_cash: float) -> pd.DataFrame:
    df = load_pred(ticker)
    df = STRATEGIES[strategy_key]["signal_fn"](df, threshold)
    return run_backtest(df, initial_cash=initial_cash)

def available_tickers() -> list[str]:
    return sorted([
        f.stem.replace("_backtest", "")
        for f in BASE_DIR.glob("*_backtest.csv")
    ])

def fmt_pct(v: float) -> str:
    return f"{v:+.2%}"

def fmt_dollar(v: float) -> str:
    return f"${v:,.2f}"

def fmt_dollar0(v: float) -> str:
    return f"${v:,.0f}"

# =========================================================
# Pages
# =========================================================

def page_stock_backtest():
    # Strategy selector
    strat_keys = list(STRATEGIES.keys())
    strategy_key = st.sidebar.selectbox(
        "Strategy",
        options=strat_keys,
        index=strat_keys.index(DEFAULT_STRATEGY),
        format_func=lambda k: STRATEGIES[k]["label"],
    )
    st.sidebar.caption(STRATEGIES[strategy_key]["desc"])

    tickers = available_tickers()
    ticker = st.sidebar.selectbox("Select Stock", tickers)

    threshold_label = st.sidebar.selectbox(
        "Signal Threshold",
        options=["0.1%", "0.2%", "0.3%", "0.4%", "0.5%"],
        index=0,
    )
    threshold = float(threshold_label.replace("%", "")) / 100

    df = get_balance(ticker, strategy_key, threshold, INITIAL_CASH)

    # ── Metrics ──────────────────────────────────────────
    final = df["total"].iloc[-1]
    total_ret = final / INITIAL_CASH - 1
    mdd = (df["total"] / df["total"].cummax() - 1).min()
    daily_ret = df["total"].pct_change().fillna(0)
    sharpe = (daily_ret.mean() / daily_ret.std() * (252 ** 0.5)
              if daily_ret.std() > 0 else 0)
    n_buy = (df["trade"] == "BUY").sum()
    n_sell = (df["trade"] == "SELL").sum()
    final_shares = float(df["shares"].iloc[-1])

    # Win rate: 買進隔天漲 / 賣出隔天跌 算贏
    next_actual = df["actual"].shift(-1)
    is_trade = df["trade"].isin(["BUY", "SELL"]) & next_actual.notna()
    is_win = (
        ((df["trade"] == "BUY") & (next_actual > df["actual"])) |
        ((df["trade"] == "SELL") & (next_actual < df["actual"]))
    )
    n_trades = int(is_trade.sum())
    n_wins = int((is_win & is_trade).sum())
    win_rate = n_wins / n_trades if n_trades > 0 else 0

    # Buy & Hold benchmark
    bh_shares = int(INITIAL_CASH / df["actual"].iloc[0])
    bh_cost = bh_shares * df["actual"].iloc[0]
    bh_curve = bh_cost + bh_shares * (df["actual"] - df["actual"].iloc[0])
    bh_final = bh_curve.iloc[-1]
    bh_ret = bh_final / INITIAL_CASH - 1
    bh_daily_ret = bh_curve.pct_change().fillna(0)
    bh_sharpe = (bh_daily_ret.mean() / bh_daily_ret.std() * (252 ** 0.5)
                 if bh_daily_ret.std() > 0 else 0)
    bh_mdd = (bh_curve / bh_curve.cummax() - 1).min()

    # ── Header ───────────────────────────────────────────
    st.markdown(f"### 📈 {ticker}")
    st.caption(
        f"{STRATEGIES[strategy_key]['label']}  ·  Threshold {threshold_label}"
        f"  ·  Initial {fmt_dollar(INITIAL_CASH)}"
    )

    # ── Two comparison cards: Strategy vs Buy & Hold ─────
    left, right = st.columns(2, gap="medium")

    with left:
        with st.container(border=True):
            st.markdown("##### 🤖 Strategy")
            a, b, c, d = st.columns(4)
            a.metric("Final Balance", fmt_dollar0(final), fmt_pct(total_ret))
            b.metric("Sharpe", f"{sharpe:.2f}")
            c.metric("Max Drawdown", fmt_pct(mdd))
            d.metric("Win Rate", f"{win_rate:.1%}")
            st.caption(
                f"Trades: {n_buy} buy / {n_sell} sell   ·   "
                f"Wins: {n_wins}/{n_trades} ({win_rate:.1%})   ·   "
                f"Shares held: {final_shares:,.2f}"
            )

    with right:
        with st.container(border=True):
            st.markdown("##### 💎 Buy & Hold")
            a, b, c = st.columns(3)
            a.metric("Final Balance", fmt_dollar0(bh_final), fmt_pct(bh_ret))
            b.metric("Sharpe", f"{bh_sharpe:.2f}")
            c.metric("Max Drawdown", fmt_pct(bh_mdd))
            st.caption(f"Shares held: {bh_shares:,} (buy once, hold)")

    # ── Equity curve ─────────────────────────────────────
    st.markdown("##### Equity Curve")

    df["cash_pct"] = df["cash"] / df["total"] * 100
    df["stock_pct"] = df["stock_value"] / df["total"] * 100
    df["trade_label"] = df["trade"].map(
        lambda t: "🟢 Buy" if t == "BUY" else ("🔴 Sell" if t == "SELL" else "—")
    )

    fig = go.Figure()

    # Strategy line
    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["total"],
        name="Strategy",
        line=dict(color="#2196F3", width=2),
        customdata=df[["total", "trade_label", "shares", "cash_pct", "stock_pct", "cash", "stock_value"]].values,
        hovertemplate=(
            "<b>%{x|%b %d, %Y}</b><br>"
            "Strategy  : <b>$%{customdata[0]:,.2f}</b><br>"
            "Trade     : %{customdata[1]}<br>"
            "Shares    : %{customdata[2]:,.2f}<br>"
            "Cash      : $%{customdata[5]:,.2f} (%{customdata[3]:.1f}%)<br>"
            "Stock     : $%{customdata[6]:,.2f} (%{customdata[4]:.1f}%)"
            "<extra></extra>"
        ),
    ))

    # Buy & Hold line
    fig.add_trace(go.Scatter(
        x=df["date"],
        y=bh_curve,
        name="Buy & Hold",
        line=dict(color="#FF9800", width=1.5, dash="dot"),
        hovertemplate=(
            "<b>%{x|%b %d, %Y}</b><br>"
            "Buy & Hold: <b>$%{y:,.2f}</b>"
            "<extra></extra>"
        ),
    ))

    fig.add_hline(
        y=INITIAL_CASH, line_dash="dash",
        line_color="gray", annotation_text=f"Initial {fmt_dollar(INITIAL_CASH)}"
    )

    # Buy/sell markers
    buys = df[df["trade"] == "BUY"]
    sells = df[df["trade"] == "SELL"]
    fig.add_trace(go.Scatter(
        x=buys["date"], y=buys["total"],
        mode="markers", name="Buy",
        marker=dict(color="green", symbol="triangle-up", size=7),
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=sells["date"], y=sells["total"],
        mode="markers", name="Sell",
        marker=dict(color="red", symbol="triangle-down", size=7),
        hoverinfo="skip",
    ))

    fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", y=1.02),
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode="x",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Trade log ────────────────────────────────────────
    with st.expander("Trade Log", expanded=False):
        trades = df[df["trade"].isin(["BUY", "SELL"])].copy()
        trades = trades[["date", "trade", "actual", "shares", "cash", "total"]].rename(columns={
            "date": "Date", "trade": "Action", "actual": "Price",
            "shares": "Shares After", "cash": "Cash After", "total": "Portfolio"
        })
        st.dataframe(trades.reset_index(drop=True), use_container_width=True)

    # ── Full daily table ──────────────────────────────────
    with st.expander("Daily Balance Table", expanded=False):
        st.dataframe(df, use_container_width=True)


def page_summary():
    st.title("📊 All Stocks Summary")

    df = load_summary()

    df_display = df.copy()
    df_display["total_return"] = df_display["total_return"].map(fmt_pct)
    df_display["mdd"] = df_display["mdd"].map(fmt_pct)
    df_display["final_balance"] = df_display["final_balance"].map(fmt_dollar)
    df_display["sharpe"] = df_display["sharpe"].map(lambda x: f"{x:.2f}")
    df_display = df_display.rename(columns={
        "ticker": "Ticker",
        "total_return": "Return",
        "final_balance": "Final Balance",
        "sharpe": "Sharpe",
        "mdd": "MDD",
        "n_buy": "Buys",
        "n_sell": "Sells",
        "final_shares": "Shares",
    })

    st.dataframe(df_display, use_container_width=True, hide_index=True)

    # Bar chart — total return
    raw = load_summary()
    fig = go.Figure(go.Bar(
        x=raw["ticker"],
        y=raw["total_return"] * 100,
        marker_color=[
            "#4CAF50" if v >= 0 else "#F44336"
            for v in raw["total_return"]
        ],
        text=[fmt_pct(v) for v in raw["total_return"]],
        textposition="outside",
    ))
    fig.update_layout(
        title="Total Return by Stock",
        yaxis_title="Return (%)",
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# Navigation
# =========================================================

PAGES = {
    "📈 Stock Backtest": page_stock_backtest,
    "📊 Summary": page_summary,
}

selection = st.sidebar.radio(
    "Menu", list(PAGES.keys()), label_visibility="collapsed"
)
st.sidebar.markdown("---")

PAGES[selection]()
