"""
Walk-Forward Backtesting Engine — Indian Stock Market (Nifty 50)
Simulates historical trading day-by-day to prevent lookahead bias.
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta

# Import your existing engine components
from trading_engine import (
    NIFTY_50, ALPHA_FILE, fetch_data, load_alphas, 
    AlphaEvaluator, generate_signals_from_cache, PaperPortfolio
)

# ─────────────────────────────────────────────
# BACKTEST CONFIGURATION
# ─────────────────────────────────────────────
BACKTEST_MONTHS = 6
CAPITAL         = 1_000_000
TOP_N           = 5
BOTTOM_N        = 5
VOTE_THRESHOLD  = 0.5

# Separate log files so we don't overwrite your live paper trading data
BACKTEST_CSV    = "backtest_log.csv"
BACKTEST_JSON   = "backtest_trades.json"
BACKTEST_DAILY_CSV = "backtest_daily_log.csv"


def timestamped_output_path(path: str, run_stamp: str) -> str:
    stem, ext = os.path.splitext(path)
    return f"{stem}_{run_stamp}{ext}"

# ─────────────────────────────────────────────
# MUTE PRINTS FOR SPEED (Optional)
# ─────────────────────────────────────────────
class HiddenPrints:
    """Context manager to suppress the massive amount of terminal prints during backtesting."""
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

# ─────────────────────────────────────────────
# MAIN BACKTEST LOOP
# ─────────────────────────────────────────────
def run_backtest():
    print("\n" + "=" * 60)
    print(f"  ALPHA ENGINE - {BACKTEST_MONTHS}-MONTH WALK-FORWARD BACKTEST")
    print("=" * 60 + "\n")

    # 1. Each run gets its own output files so completed logs never mix with partial runs.
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backtest_csv = timestamped_output_path(BACKTEST_CSV, run_stamp)
    backtest_json = timestamped_output_path(BACKTEST_JSON, run_stamp)
    backtest_daily_csv = timestamped_output_path(BACKTEST_DAILY_CSV, run_stamp)

    alphas_df = load_alphas(ALPHA_FILE)
    supported_count = int(alphas_df["is_supported"].sum()) if "is_supported" in alphas_df else len(alphas_df)
    unsupported = alphas_df.loc[~alphas_df["is_supported"], ["alpha_id", "support_reason"]] if "is_supported" in alphas_df else pd.DataFrame()
    print(f"[ALPHAS] Loaded {len(alphas_df)} alphas. Supported now: {supported_count}.")
    if not unsupported.empty:
        unsupported_ids = ", ".join(unsupported["alpha_id"].tolist())
        print(f"[ALPHAS] Skipping {len(unsupported)} alphas for this run due to missing data support: {unsupported_ids}")
    
    # 2. Fetch historical data with a large enough buffer for 250-day rolling alphas.
    total_days = (30 * BACKTEST_MONTHS) + 400
    full_data = fetch_data(NIFTY_50, lookback_days=total_days)
    
    if not full_data:
        print("[ERROR] No data fetched.")
        return

    # 3. Determine the trading days within the 6-month window
    # We use Reliance as a proxy to get a clean list of NSE trading days (excluding weekends/holidays)
    proxy_df = full_data[NIFTY_50[0]]
    all_dates = proxy_df.index
    backtest_start_date = all_dates[-1] - pd.Timedelta(days=30 * BACKTEST_MONTHS)
    trading_days = all_dates[all_dates >= backtest_start_date]

    print(f"\n[BACKTEST] Starting simulation over {len(trading_days)} trading days...")

    print("[BACKTEST] Precomputing alpha score histories...")
    full_evaluator = AlphaEvaluator(full_data)
    alpha_cache = {}
    supported_alphas = alphas_df[alphas_df["is_supported"]] if "is_supported" in alphas_df else alphas_df
    for _, row in supported_alphas.iterrows():
        with HiddenPrints():
            result = full_evaluator.evaluate_full(row["expression"])
        if isinstance(result, pd.DataFrame) and not result.empty:
            alpha_cache[row["alpha_id"]] = result
    print(f"[BACKTEST] Alpha histories ready for {len(alpha_cache)} alphas.")
    
    # Initialize the portfolio tracker
    portfolio = PaperPortfolio(CAPITAL, backtest_csv, backtest_json)
    daily_rows = []

    # 4. The Walk-Forward Loop
    for i, current_date in enumerate(trading_days):
        print(f"\r  [SIMULATING] Date: {current_date.date()}  ({i+1}/{len(trading_days)})", end="")
        
        # SLICE DATA: Cut off all data after the current simulated date (Zero Lookahead Bias)
        sliced_data = {}
        for ticker, df in full_data.items():
            past_data = df.loc[:current_date]
            if not past_data.empty and len(past_data) >= 20:
                sliced_data[ticker] = past_data
                
        if not sliced_data:
            continue

        # Suppress the hundreds of "[EVAL]" print statements to speed up the loop
        signals = generate_signals_from_cache(
            alphas_df, alpha_cache, current_date, list(sliced_data.keys()),
            TOP_N, BOTTOM_N, VOTE_THRESHOLD
        )
        portfolio.execute(signals, sliced_data, simulated_date=current_date)

        snapshot = portfolio.snapshot(sliced_data)
        daily_rows.append({
            "simulated_date": current_date.strftime("%Y-%m-%d"),
            "signal_count": len(signals),
            "free_capital": snapshot["free_capital"],
            "reserved_margin": snapshot["reserved_margin"],
            "realized_pnl": snapshot["realized_pnl"],
            "unrealized_pnl": snapshot["unrealized_pnl"],
            "total_equity": snapshot["total_equity"],
            "open_positions": snapshot["open_positions"],
        })

    print("\n\n[BACKTEST] Simulation Complete!")
    portfolio.summary()
    daily_df = pd.DataFrame(daily_rows)
    daily_df.to_csv(backtest_daily_csv, index=False)
    if not daily_df.empty:
        equity_curve = daily_df["total_equity"]
        running_peak = equity_curve.cummax()
        drawdown = ((equity_curve - running_peak) / running_peak).min() * 100
        total_return = ((equity_curve.iloc[-1] / CAPITAL) - 1) * 100
        print(f"[RESULT] Final Equity  : Rs {equity_curve.iloc[-1]:,.2f}")
        print(f"[RESULT] Total Return  : {total_return:.2f}%")
        print(f"[RESULT] Max Drawdown : {drawdown:.2f}%")
    print(f"\n[LOG] Backtest log saved to {backtest_csv}")
    print(f"[LOG] Dashboard data saved to {backtest_json}\n")
    print(f"[LOG] Daily equity log saved to {backtest_daily_csv}\n")

if __name__ == "__main__":
    run_backtest()
