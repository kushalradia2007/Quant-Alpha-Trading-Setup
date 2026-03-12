import json
import unittest
from pathlib import Path
from uuid import uuid4

import pandas as pd

from trading_engine import PaperPortfolio, generate_signals


class StubEvaluator:
    def __init__(self, scores_by_expression):
        self.tickers = ["AAA.NS", "BBB.NS", "CCC.NS"]
        self._scores_by_expression = scores_by_expression

    def evaluate(self, expression):
        return self._scores_by_expression[expression]


def make_price_frame(close_price):
    index = pd.date_range("2026-01-01", periods=25, freq="D")
    return pd.DataFrame(
        {
            "Open": [close_price] * len(index),
            "High": [close_price] * len(index),
            "Low": [close_price] * len(index),
            "Close": [close_price] * len(index),
            "Volume": [1000] * len(index),
        },
        index=index,
    )


class TradingEngineTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(".test_artifacts") / f"{self._testMethodName}_{uuid4().hex}"
        self.tmpdir.mkdir(parents=True, exist_ok=True)

    def test_generate_signals_uses_weighted_votes(self):
        alphas = pd.DataFrame(
            [
                {"expression": "alpha_1", "weight": 1.0},
                {"expression": "alpha_2", "weight": 1.0},
            ]
        )
        evaluator = StubEvaluator(
            {
                "alpha_1": pd.Series({"AAA.NS": 3.0, "BBB.NS": 2.0, "CCC.NS": 1.0}),
                "alpha_2": pd.Series({"AAA.NS": 2.0, "BBB.NS": 3.0, "CCC.NS": 1.0}),
            }
        )

        signals = generate_signals(alphas, evaluator, top_n=1, bottom_n=1, vote_threshold=0.5)

        self.assertEqual(signals["AAA.NS"], "BUY")
        self.assertEqual(signals["BBB.NS"], "BUY")
        self.assertEqual(signals["CCC.NS"], "SELL")

    def test_execute_long_trade_updates_realized_pnl_and_log_schema(self):
        csv_path = self.tmpdir / "trades.csv"
        json_path = self.tmpdir / "trades.json"
        portfolio = PaperPortfolio(1000, str(csv_path), str(json_path))

        open_data = {"AAA.NS": make_price_frame(10.0)}
        close_data = {"AAA.NS": make_price_frame(12.0)}

        portfolio.execute({"AAA.NS": "BUY"}, open_data, simulated_date=pd.Timestamp("2026-01-10"))
        portfolio.execute({}, close_data, simulated_date=pd.Timestamp("2026-01-11"))

        self.assertEqual(round(portfolio.capital, 2), 1200.00)
        self.assertEqual(round(portfolio.realized_pnl, 2), 200.00)

        log_df = pd.read_csv(csv_path)
        self.assertEqual(list(log_df["simulated_date"]), ["2026-01-10", "2026-01-11"])
        self.assertIn("logged_at", log_df.columns)
        self.assertEqual(list(log_df["action"]), ["BUY", "CLOSE"])

        with open(json_path, encoding="utf-8") as handle:
            trades = json.load(handle)
        self.assertEqual(len(trades), 2)

    def test_execute_short_trade_returns_margin_plus_pnl(self):
        csv_path = self.tmpdir / "trades.csv"
        json_path = self.tmpdir / "trades.json"
        portfolio = PaperPortfolio(1000, str(csv_path), str(json_path))

        open_data = {"AAA.NS": make_price_frame(10.0)}
        close_data = {"AAA.NS": make_price_frame(8.0)}

        portfolio.execute({"AAA.NS": "SELL"}, open_data, simulated_date=pd.Timestamp("2026-01-10"))
        portfolio.execute({}, close_data, simulated_date=pd.Timestamp("2026-01-11"))

        self.assertEqual(round(portfolio.capital, 2), 1200.00)
        self.assertEqual(round(portfolio.realized_pnl, 2), 200.00)


if __name__ == "__main__":
    unittest.main()
