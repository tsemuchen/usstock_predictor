"""
Usage:
python update_and_predict.py
python update_and_predict.py --output-mode prediction
python update_and_predict.py --output-mode backtest
"""

import argparse
import os

import pandas as pd

from get_kline import get_kline
from stock_predictor import run_one_stock, run_many_stocks


def read_tickers(ticker_file: str) -> list[str]:
    with open(ticker_file, "r") as f:
        tickers = [
            line.strip().upper()
            for line in f
            if line.strip() and not line.startswith("#")
        ]

    return tickers


def print_single_result(result: dict) -> None:
    print(f"\nSymbol: {result['symbol']}")
    print(f"Status: {result['status']}")

    if result["status"] != "ok":
        return

    print("\nNaive Baseline:")
    print(f"MAE  : {result['naive_mae']:.4f}")
    print(f"RMSE : {result['naive_rmse']:.4f}")

    print("\nLSTM:")
    print(f"MAE  : {result['lstm_mae']:.4f}")
    print(f"RMSE : {result['lstm_rmse']:.4f}")
    print(f"MAPE : {result['lstm_mape']:.2f}%")
    print(f"R²   : {result['lstm_r2']:.4f}")
    print(f"DA   : {result['lstm_da']:.2f}%")
    print(f"MAE Gain vs Naive: {result['lstm_vs_naive_mae_gain']:.4f}")

    print("\nXGBoost:")
    print(f"MAE  : {result['xgb_mae']:.4f}")
    print(f"RMSE : {result['xgb_rmse']:.4f}")
    print(f"MAPE : {result['xgb_mape']:.2f}%")
    print(f"R²   : {result['xgb_r2']:.4f}")
    print(f"DA   : {result['xgb_da']:.2f}%")
    print(f"MAE Gain vs Naive: {result['xgb_vs_naive_mae_gain']:.4f}")

    print("\nBest:")
    print(f"Best Performance Model         : {result['best_model']}")
    print(f"Best Performance MAE           : {result['best_model_mae']:.4f}")
    print(f"Best Comparison vs Naive (MAE) : {result['best_model_vs_naive_mae_gain']:.4f}")
    print(f"Best Directional Model         : {result['best_da_model']}")
    print(f"Best Directional Accuracy      : {result['best_da']:.2f}%")


def print_multi_summary(df: pd.DataFrame) -> None:
    ok_df = df[df["status"] == "ok"].copy()
    fail_df = df[df["status"] != "ok"].copy()

    print("\n====================")
    print("BATCH RUN SUMMARY")
    print("====================")
    print(f"Successful: {len(ok_df)}")
    print(f"Failed    : {len(fail_df)}")

    if len(ok_df) == 0:
        print("\nNo successful runs.")
        if len(fail_df) > 0:
            print(fail_df[["symbol", "status"]].to_string(index=False))
        return

    best_perf = ok_df.sort_values("best_model_mae", ascending=True).iloc[0]
    best_vs_naive = ok_df.sort_values("best_model_vs_naive_mae_gain", ascending=False).iloc[0]
    best_direction = ok_df.sort_values("best_da", ascending=False).iloc[0]

    print("\n1. Best Performance")
    print(
        f"{best_perf['symbol']} | Model={best_perf['best_model']} | "
        f"MAE={best_perf['best_model_mae']:.4f} | RMSE={best_perf['best_model_rmse']:.4f}"
    )

    print("\n2. Best Comparison Against Naive")
    print(
        f"{best_vs_naive['symbol']} | Model={best_vs_naive['best_model']} | "
        f"MAE Gain={best_vs_naive['best_model_vs_naive_mae_gain']:.4f} | "
        f"Model MAE={best_vs_naive['best_model_mae']:.4f} | Naive MAE={best_vs_naive['naive_mae']:.4f}"
    )

    print("\n3. Best Directional Accuracy")
    print(
        f"{best_direction['symbol']} | Model={best_direction['best_da_model']} | "
        f"Directional Accuracy={best_direction['best_da']:.2f}%"
    )


def save_prediction_output(result: dict, output_mode: str) -> None:
    if result["status"] != "ok" or "prediction_df" not in result:
        return

    if output_mode == "backtest":
        output_dir = "backtest_files"
        filename = f"{result['symbol']}_backtest.csv"
        message = "Saved backtest file to"
    else:
        output_dir = "pred_price"
        filename = f"{result['symbol']}_pred.csv"
        message = "Saved prediction to"

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, filename)

    result["prediction_df"].to_csv(output_path, index=False)

    print(f"\n{message}: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Update k-line data, then run stock prediction.")
    parser.add_argument("--ticker-file", default="selected_tickers.txt", help="File containing tickers, one per line")
    parser.add_argument("--data-dir", default="selected_daily_kline", help="Folder to save k-line CSV files")
    parser.add_argument("--kline-start-date", default="2010-01-01", help="Start date for downloaded k-line data")
    parser.add_argument("--start-date", default="2015-01-01", help="Prediction training start date")
    parser.add_argument("--seq-length", type=int, default=10)
    parser.add_argument("--min-rows", type=int, default=250)
    parser.add_argument("--save-results", default="", help="Optional CSV path to save batch results")
    parser.add_argument(
        "--output-mode",
        choices=["prediction", "backtest"],
        default="prediction",
        help="Choose whether to save prediction files or backtest files"
    )

    args = parser.parse_args()

    tickers = read_tickers(args.ticker_file)

    if len(tickers) == 0:
        print(f"No tickers found in {args.ticker_file}")
        return

    file_paths = get_kline(
        tickers=tickers,
        output_dir=args.data_dir,
        start_date=args.kline_start_date,
    )

    if len(file_paths) == 0:
        print("No k-line files downloaded.")
        return

    if len(file_paths) == 1:
        result = run_one_stock(
            file_path=file_paths[0],
            start_date=args.start_date,
            seq_length=args.seq_length,
            min_rows_required=args.min_rows,
        )

        save_prediction_output(result, args.output_mode)
        print_single_result(result)
        return

    results = []

    for idx, file_path in enumerate(file_paths, start=1):
        print(f"[{idx}/{len(file_paths)}] Running {os.path.basename(file_path)} ...")

        result = run_one_stock(
            file_path=file_path,
            start_date=args.start_date,
            seq_length=args.seq_length,
            min_rows_required=args.min_rows,
        )

        results.append(result)

        save_prediction_output(result, args.output_mode)

    results_df = pd.DataFrame(results)

    print_multi_summary(results_df)

    if args.save_results:
        results_df.to_csv(args.save_results, index=False)
        print(f"\nSaved results to: {args.save_results}")


if __name__ == "__main__":
    main()