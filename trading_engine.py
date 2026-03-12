"""
Alpha-Driven Paper Trading Engine — Indian Stock Market (Nifty 50)
==================================================================
• Reads alpha expressions from alphas.csv
• Downloads OHLCV data via yfinance
• Evaluates alphas using a robust Namespace Dictionary
• Executes paper trades and logs everything to CSV/JSON
"""

import os
import json
import math
import re
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")

ALPHA_FILE      = "alphas.csv"
TRADE_LOG_CSV   = "trade_log.csv"
TRADES_JSON     = "trades.json"
CAPITAL         = 1_000_000          
TOP_N           = 5                  
BOTTOM_N        = 5                  
LOOKBACK_DAYS   = 120                
VOTE_THRESHOLD  = 0.5                

NIFTY_50 = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "BAJFINANCE.NS",
    "KOTAKBANK.NS", "AXISBANK.NS", "LT.NS", "ASIANPAINT.NS", "HCLTECH.NS",
    "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS",
    "NTPC.NS", "POWERGRID.NS", "TECHM.NS", "NESTLEIND.NS", "ONGC.NS",
    "JSWSTEEL.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "M&M.NS", "ADANIENT.NS",
    "ADANIPORTS.NS", "COALINDIA.NS", "GRASIM.NS", "BPCL.NS", "DIVISLAB.NS",
    "DRREDDY.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "HINDALCO.NS", "APOLLOHOSP.NS",
    "CIPLA.NS", "BAJAJFINSV.NS", "BAJAJ-AUTO.NS", "BRITANNIA.NS", "SBILIFE.NS",
    "HDFCLIFE.NS", "UPL.NS", "TATACONSUM.NS", "INDUSINDBK.NS", "LTIM.NS",
]

UNSUPPORTED_ALPHA_PATTERNS = (
    ("indneutralize", "requires industry classification data"),
    ("cap", "requires market capitalization data"),
)


def _balanced_outer_parens(text: str) -> str:
    while text.startswith("(") and text.endswith(")"):
        depth = 0
        balanced = True
        for idx, char in enumerate(text):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            if depth == 0 and idx != len(text) - 1:
                balanced = False
                break
        if balanced:
            text = text[1:-1].strip()
        else:
            break
    return text


def _split_ternary(expression: str):
    depth = 0
    question_index = None
    colon_index = None

    for idx, char in enumerate(expression):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "?" and depth == 0 and question_index is None:
            question_index = idx
        elif char == ":" and depth == 0 and question_index is not None:
            colon_index = idx
            break

    if question_index is None or colon_index is None:
        return None

    return (
        expression[:question_index].strip(),
        expression[question_index + 1 : colon_index].strip(),
        expression[colon_index + 1 :].strip(),
    )


def convert_ternary(expression: str) -> str:
    expression = expression.strip()
    expression = _balanced_outer_parens(expression)
    parts = _split_ternary(expression)
    if not parts:
        return expression
    condition, when_true, when_false = parts
    return f"where({convert_ternary(condition)}, {convert_ternary(when_true)}, {convert_ternary(when_false)})"


def convert_all_ternaries(expression: str) -> str:
    pieces = []
    idx = 0

    while idx < len(expression):
        char = expression[idx]
        if char != "(":
            pieces.append(char)
            idx += 1
            continue

        depth = 1
        end = idx + 1
        while end < len(expression) and depth > 0:
            if expression[end] == "(":
                depth += 1
            elif expression[end] == ")":
                depth -= 1
            end += 1

        inner = expression[idx + 1 : end - 1]
        pieces.append("(" + convert_all_ternaries(inner) + ")")
        idx = end

    rebuilt = "".join(pieces)
    converted = convert_ternary(rebuilt)
    return converted


def normalize_alpha_expression(expression: str) -> str:
    expr = " ".join(str(expression).split())
    expr = convert_all_ternaries(expr)

    replacements = [
        (r"\bTs_ArgMax\b", "ts_argmax"),
        (r"\bTs_ArgMin\b", "ts_argmin"),
        (r"\bTs_Rank\b", "ts_rank"),
        (r"\bSignedPower\b", "signedpower"),
        (r"\bIndNeutralize\b", "indneutralize"),
        (r"\bstddev\b", "ts_std"),
        (r"\bLog\b", "log"),
        (r"\bSign\b", "sign"),
    ]
    for pattern, replacement in replacements:
        expr = re.sub(pattern, replacement, expr, flags=re.IGNORECASE)

    expr = expr.replace("^", "**")
    expr = expr.replace("||", "|")
    expr = expr.replace("&&", "&")
    expr = re.sub(r"\bdelta\s*\(\s*close\s*-\s*1\s*,", "delta((close - 1),", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\badv(\d+(?:\.\d+)?)\b", r"adv(\1)", expr, flags=re.IGNORECASE)
    return expr

class AlphaEvaluator:
    """Evaluates WorldQuant-style alpha expressions safely."""

    def __init__(self, data: dict[str, pd.DataFrame]):
        self.tickers = list(data.keys())
        dates = sorted(set.intersection(*[set(df.index) for df in data.values()]))
        self.dates = dates

        self.close  = pd.DataFrame({t: data[t]["Close"]  for t in self.tickers}).loc[dates]
        self.open_  = pd.DataFrame({t: data[t]["Open"]   for t in self.tickers}).loc[dates]
        self.high   = pd.DataFrame({t: data[t]["High"]   for t in self.tickers}).loc[dates]
        self.low    = pd.DataFrame({t: data[t]["Low"]    for t in self.tickers}).loc[dates]
        self.volume = pd.DataFrame({t: data[t]["Volume"] for t in self.tickers}).loc[dates]
        self.vwap   = (self.close + self.high + self.low) / 3
        self.returns= self.close.pct_change()
        self._adv_cache = {}

    @staticmethod
    def _window(n) -> int:
        return max(1, int(math.floor(float(n))))

    def _adv(self, n) -> pd.DataFrame:
        n = self._window(n)
        if n not in self._adv_cache:
            dollar_volume = self.close * self.volume
            self._adv_cache[n] = dollar_volume.rolling(n).mean()
        return self._adv_cache[n]

    @staticmethod
    def _rank(df: pd.DataFrame) -> pd.DataFrame: return df.rank(axis=1, pct=True)
    @staticmethod
    def _delay(df: pd.DataFrame, n) -> pd.DataFrame: return df.shift(AlphaEvaluator._window(n))
    @staticmethod
    def _delta(df: pd.DataFrame, n) -> pd.DataFrame: return df - df.shift(AlphaEvaluator._window(n))
    @staticmethod
    def _ts_max(df: pd.DataFrame, n) -> pd.DataFrame: return df.rolling(AlphaEvaluator._window(n)).max()
    @staticmethod
    def _ts_min(df: pd.DataFrame, n) -> pd.DataFrame: return df.rolling(AlphaEvaluator._window(n)).min()
    @staticmethod
    def _ts_mean(df: pd.DataFrame, n) -> pd.DataFrame: return df.rolling(AlphaEvaluator._window(n)).mean()
    @staticmethod
    def _ts_std(df: pd.DataFrame, n) -> pd.DataFrame: return df.rolling(AlphaEvaluator._window(n)).std()
    @staticmethod
    def _ts_sum(df: pd.DataFrame, n) -> pd.DataFrame:
        return df.rolling(AlphaEvaluator._window(n)).sum()
    @staticmethod
    def _ts_product(df: pd.DataFrame, n) -> pd.DataFrame:
        return df.rolling(AlphaEvaluator._window(n)).apply(np.prod, raw=True)
    @staticmethod
    def _ts_rank(df: pd.DataFrame, n) -> pd.DataFrame:
        n = AlphaEvaluator._window(n)
        return df.rolling(n).apply(lambda x: pd.Series(x).rank().iloc[-1] / n, raw=False)
    @staticmethod
    def _ts_argmin(df: pd.DataFrame, n) -> pd.DataFrame:
        n = AlphaEvaluator._window(n)
        return df.rolling(n).apply(lambda x: n - np.argmin(x) - 1, raw=True)
    @staticmethod
    def _ts_argmax(df: pd.DataFrame, n) -> pd.DataFrame:
        n = AlphaEvaluator._window(n)
        return df.rolling(n).apply(lambda x: n - np.argmax(x) - 1, raw=True)
    @staticmethod
    def _decay_linear(df: pd.DataFrame, n) -> pd.DataFrame:
        n = AlphaEvaluator._window(n)
        w = np.arange(1, n + 1)
        w = w / w.sum()
        return df.rolling(n).apply(lambda x: np.dot(x, w), raw=True)
    @staticmethod
    def _scale(df: pd.DataFrame, a: float = 1.0) -> pd.DataFrame:
        return df.div(df.abs().sum(axis=1), axis=0) * a
    @staticmethod
    def _sign(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(np.sign(df), index=df.index, columns=df.columns)
    @staticmethod
    def _abs(df):
        return np.abs(df)
    @staticmethod
    def _log(df):
        return np.log(df)
    @staticmethod
    def _signedpower(df, a):
        return np.sign(df) * np.power(np.abs(df), a)
    def _max(self, x, y):
        if np.isscalar(y):
            return self._ts_max(x, y)
        return pd.DataFrame(np.maximum(x, y), index=self.close.index, columns=self.close.columns)
    def _min(self, x, y):
        if np.isscalar(y):
            return self._ts_min(x, y)
        return pd.DataFrame(np.minimum(x, y), index=self.close.index, columns=self.close.columns)
    @staticmethod
    def _indneutralize(df, _group):
        raise NotImplementedError("indneutralize requires industry classification data")

    def _where(self, condition: pd.DataFrame, x, y) -> pd.DataFrame:
        return pd.DataFrame(np.where(condition, x, y), index=self.close.index, columns=self.close.columns)

    @staticmethod
    def _correlation(a: pd.DataFrame, b: pd.DataFrame, n) -> pd.DataFrame:
        n = AlphaEvaluator._window(n)
        result = pd.DataFrame(index=a.index, columns=a.columns, dtype=float)
        for i in range(n - 1, len(a)):
            result.iloc[i] = a.iloc[i - n + 1 : i + 1].corrwith(b.iloc[i - n + 1 : i + 1], axis=0)
        return result

    @staticmethod
    def _covariance(a: pd.DataFrame, b: pd.DataFrame, n) -> pd.DataFrame:
        n = AlphaEvaluator._window(n)
        result = pd.DataFrame(index=a.index, columns=a.columns, dtype=float)
        for i in range(n - 1, len(a)):
            wa = a.iloc[i - n + 1 : i + 1]
            wb = b.iloc[i - n + 1 : i + 1]
            result.iloc[i] = (wa * wb).mean() - wa.mean() * wb.mean()
        return result

    def _safe_locals(self):
        class _IndClass:
            sector = "sector"
            industry = "industry"
            subindustry = "subindustry"

        return {
            "close": self.close, "open": self.open_, "high": self.high, 
            "low": self.low, "volume": self.volume, "vwap": self.vwap, "returns": self.returns,
            "adv": self._adv,
            "adv5": self._adv(5), "adv10": self._adv(10), "adv15": self._adv(15), "adv20": self._adv(20),
            "adv30": self._adv(30), "adv40": self._adv(40), "adv50": self._adv(50), "adv60": self._adv(60),
            "adv81": self._adv(81), "adv120": self._adv(120), "adv150": self._adv(150), "adv180": self._adv(180),
            "rank": self._rank, "delay": self._delay, "delta": self._delta,
            "ts_max": self._ts_max, "ts_min": self._ts_min, "ts_mean": self._ts_mean,
            "ts_std": self._ts_std, "ts_sum": self._ts_sum, "ts_product": self._ts_product,
            "sum": self._ts_sum, "product": self._ts_product,
            "ts_rank": self._ts_rank, "ts_argmin": self._ts_argmin,
            "ts_argmax": self._ts_argmax, "correlation": self._correlation, 
            "covariance": self._covariance, "decay_linear": self._decay_linear,
            "scale": self._scale, "sign": self._sign, "abs": self._abs, "log": self._log,
            "signedpower": self._signedpower, "where": self._where, "max": self._max,
            "min": self._min, "indneutralize": self._indneutralize, "IndClass": _IndClass,
            "cap": None,
        }

    def evaluate_full(self, expression: str):
        """Evaluates the full time-series output for an alpha expression."""
        try:
            result = eval(expression, {"__builtins__": None}, self._safe_locals())
            if isinstance(result, np.ndarray):
                result = pd.DataFrame(result, index=self.close.index, columns=self.close.columns)
            if isinstance(result, (bool, np.bool_)):
                result = pd.Series(float(result), index=self.close.columns)
            if np.isscalar(result):
                result = pd.Series(float(result), index=self.close.columns)
            return result
        except Exception as e:
            print(f"  [WARN] Could not evaluate '{expression}': {e}")
            return pd.Series(dtype=float)

    def evaluate(self, expression: str) -> pd.Series:
        """Parses the expression natively using a local dictionary."""
        try:
            result = self.evaluate_full(expression)
            if isinstance(result, pd.Series):
                return result.dropna()
            return result.iloc[-1].dropna()
        except Exception as e:
            print(f"  [WARN] Could not evaluate '{expression}': {e}")
            return pd.Series(dtype=float)


def fetch_data(tickers: list[str], lookback_days: int) -> dict[str, pd.DataFrame]:
    end = datetime.today()
    start = end - timedelta(days=lookback_days + 15)
    print(f"[DATA] Fetching {len(tickers)} tickers from {start.date()} to {end.date()} ...")
    raw = yf.download(
        tickers, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
        group_by="ticker", auto_adjust=True, progress=False, threads=True,
    )
    data = {}
    if raw is None:
        return data
    for t in tickers:
        try:
            df = raw[t].dropna()
            df = df[~df.index.duplicated(keep='first')]
            if len(df) >= 20:
                data[t] = df
        except Exception:
            pass
    return data

def load_alphas(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df["weight"] = df["weight"].astype(float)
    df["raw_expression"] = df["expression"]
    df["expression"] = df["expression"].map(normalize_alpha_expression)

    def support_reason(expr: str) -> str:
        lowered = expr.lower()
        for pattern, reason in UNSUPPORTED_ALPHA_PATTERNS:
            if re.search(rf"\b{pattern}\b", lowered):
                return reason
        return ""

    df["support_reason"] = df["expression"].map(support_reason)
    df["is_supported"] = df["support_reason"].eq("")
    return df

def generate_signals(alphas_df, evaluator, top_n, bottom_n, vote_threshold):
    all_tickers = evaluator.tickers
    buy_votes   = {t: 0.0 for t in all_tickers}
    sell_votes  = {t: 0.0 for t in all_tickers}
    if "is_supported" in alphas_df.columns:
        supported_alphas = alphas_df[alphas_df["is_supported"]]
    else:
        supported_alphas = alphas_df
    total_weight = supported_alphas["weight"].sum()
    if total_weight <= 0:
        return {}

    for _, row in supported_alphas.iterrows():
        scores = evaluator.evaluate(row["expression"])
        if scores.empty: continue
        ranked = scores.rank(ascending=False)
        for t in scores.index:
            if t not in buy_votes: continue
            r = ranked[t]
            if r <= top_n: buy_votes[t] += row["weight"]
            if r >= (len(scores) - bottom_n + 1): sell_votes[t] += row["weight"]

    signals = {}
    for t in all_tickers:
        b_frac = buy_votes[t] / total_weight
        s_frac = sell_votes[t] / total_weight
        if b_frac >= vote_threshold and b_frac > s_frac: signals[t] = "BUY"
        elif s_frac >= vote_threshold and s_frac > b_frac: signals[t] = "SELL"
    return signals


def generate_signals_from_cache(alphas_df, alpha_cache, current_date, tickers, top_n, bottom_n, vote_threshold):
    buy_votes = {t: 0.0 for t in tickers}
    sell_votes = {t: 0.0 for t in tickers}

    if "is_supported" in alphas_df.columns:
        working_alphas = alphas_df[alphas_df["is_supported"]].copy()
    else:
        working_alphas = alphas_df.copy()

    working_alphas = working_alphas[working_alphas["alpha_id"].isin(alpha_cache)]
    total_weight = working_alphas["weight"].sum()
    if total_weight <= 0:
        return {}

    for _, row in working_alphas.iterrows():
        score_frame = alpha_cache[row["alpha_id"]]
        if current_date not in score_frame.index:
            continue
        scores = score_frame.loc[current_date].dropna()
        if scores.empty:
            continue
        ranked = scores.rank(ascending=False)
        for ticker in scores.index:
            if ticker not in buy_votes:
                continue
            rank_value = ranked[ticker]
            if rank_value <= top_n:
                buy_votes[ticker] += row["weight"]
            if rank_value >= (len(scores) - bottom_n + 1):
                sell_votes[ticker] += row["weight"]

    signals = {}
    for ticker in tickers:
        buy_fraction = buy_votes[ticker] / total_weight
        sell_fraction = sell_votes[ticker] / total_weight
        if buy_fraction >= vote_threshold and buy_fraction > sell_fraction:
            signals[ticker] = "BUY"
        elif sell_fraction >= vote_threshold and sell_fraction > buy_fraction:
            signals[ticker] = "SELL"
    return signals

class PaperPortfolio:
    def __init__(self, capital, log_csv, log_json):
        self.initial_capital = float(capital)
        self.capital = float(capital)
        self.holdings = {}
        self.log_csv = log_csv
        self.log_json = log_json
        self.trade_id = 1
        self.all_trades = []
        self.realized_pnl = 0.0

        if Path(log_csv).exists():
            existing = pd.read_csv(log_csv)
            if not existing.empty: self.trade_id = int(existing["trade_id"].max()) + 1
        if Path(log_json).exists():
            with open(log_json) as f: self.all_trades = json.load(f)
        if not Path(log_csv).exists():
            pd.DataFrame(
                columns=[
                    "trade_id", "simulated_date", "logged_at", "action", "ticker", "side",
                    "quantity", "price", "value", "pnl", "capital_remaining", "notes"
                ]
            ).to_csv(log_csv, index=False)

    def _current_price(self, ticker, data):
        return float(data[ticker]["Close"].iloc[-1])

    def _unrealized_pnl(self, data):
        total = 0.0
        for ticker, pos in self.holdings.items():
            if ticker not in data:
                continue
            price = self._current_price(ticker, data)
            if pos["side"] == "BUY":
                total += (price - pos["entry_price"]) * pos["qty"]
            else:
                total += (pos["entry_price"] - price) * pos["qty"]
        return total

    def snapshot(self, data):
        reserved_margin = sum(pos["reserved_value"] for pos in self.holdings.values())
        unrealized_pnl = self._unrealized_pnl(data)
        return {
            "free_capital": round(self.capital, 2),
            "reserved_margin": round(reserved_margin, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "total_equity": round(self.capital + reserved_margin + unrealized_pnl, 2),
            "open_positions": len(self.holdings),
        }

    def execute(self, signals, data, simulated_date=None):
        simulated_label = (
            simulated_date.strftime("%Y-%m-%d")
            if simulated_date is not None
            else datetime.now().strftime("%Y-%m-%d")
        )
        logged_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        valid_signals = {t: s for t, s in signals.items() if t in data}

        to_close = [ticker for ticker, pos in self.holdings.items() if ticker not in valid_signals or valid_signals[ticker] != pos["side"]]
        for ticker in to_close:
            pos = self.holdings.pop(ticker)
            price = self._current_price(ticker, data)
            pnl = (price - pos["entry_price"]) * pos["qty"] if pos["side"] == "BUY" else (pos["entry_price"] - price) * pos["qty"]
            released_value = pos["reserved_value"] + pnl
            self.capital += released_value
            self.realized_pnl += pnl
            self._log(
                simulated_label, logged_at, "CLOSE", ticker, pos["side"], pos["qty"],
                price, released_value, pnl, f"Closed {pos['side']}"
            )

        open_tickers = set(self.holdings.keys())
        budget_per_trade = self.capital / max(1, len(valid_signals))

        for ticker, signal in valid_signals.items():
            if ticker in open_tickers: continue
            price = self._current_price(ticker, data)
            if price <= 0: continue
            qty = math.floor(budget_per_trade / price)
            if qty < 1: continue
            cost = qty * price
            if cost > self.capital: continue
            self.capital -= cost
            self.holdings[ticker] = {
                "qty": qty,
                "entry_price": price,
                "side": signal,
                "reserved_value": cost,
            }
            self._log(
                simulated_label, logged_at, signal, ticker, signal, qty, price,
                cost, 0.0, f"Opened {signal}"
            )

    def _log(self, simulated_date, logged_at, action, ticker, side, qty, price, value, pnl, notes):
        row = {
            "trade_id": self.trade_id,
            "simulated_date": simulated_date,
            "logged_at": logged_at,
            "action": action,
            "ticker": ticker,
            "side": side,
            "quantity": qty,
            "price": round(price, 2),
            "value": round(value, 2),
            "pnl": round(pnl, 2),
            "capital_remaining": round(self.capital, 2),
            "notes": notes,
        }
        pd.DataFrame([row]).to_csv(self.log_csv, mode="a", header=False, index=False)
        self.all_trades.append(row)
        with open(self.log_json, "w") as f: json.dump(self.all_trades, f, indent=2)
        self.trade_id += 1

    def summary(self):
        print("\n" + "=" * 60 + "\n  PORTFOLIO SUMMARY\n" + "=" * 60)
        print(f"  Free Capital   : Rs {self.capital:>14,.2f}")
        print(f"  Realized PnL   : Rs {self.realized_pnl:>14,.2f}")
        print(f"  Open Positions : {len(self.holdings)}")
        for t, p in self.holdings.items():
            print(f"    {t:20s}  {p['side']:4s}  qty={p['qty']}  entry=Rs {p['entry_price']:.2f}")
        print("=" * 60)

def run():
    print("\n" + "=" * 60 + f"\n  ALPHA TRADING ENGINE - PAPER MODE\n  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n" + "=" * 60 + "\n")
    alphas_df = load_alphas(ALPHA_FILE)
    data = fetch_data(NIFTY_50, LOOKBACK_DAYS)
    if not data: return print("[ERROR] No data fetched.")
    
    evaluator = AlphaEvaluator(data)
    print("\n[SIGNAL] Evaluating alphas ...")
    signals = generate_signals(alphas_df, evaluator, TOP_N, BOTTOM_N, VOTE_THRESHOLD)
    
    print("\n[TRADE] Executing paper trades ...")
    portfolio = PaperPortfolio(CAPITAL, TRADE_LOG_CSV, TRADES_JSON)
    portfolio.execute(signals, data)
    portfolio.summary()

if __name__ == "__main__":
    run()