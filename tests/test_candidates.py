"""features/candidates.py on a synthetic world: every candidate lands as a
ranked x_ column, invisible to feature_cols until admitted."""
import numpy as np
import pandas as pd

from stocks_ml.data.world import FUND_COLS
from stocks_ml.features.candidates import CANDIDATES, add_candidates, candidate_cols
from stocks_ml.features.panel import feature_cols

TICKERS = [f"T{i:02d}" for i in range(30)]


class _World:
    """A DataStore stand-in: 30 names + SPY, 3 years of prices, 12 quarters of
    SF1, a few 8-Ks and Form 4 sales."""

    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        days = pd.bdate_range("2019-01-02", "2021-12-31")
        px = 50 * np.exp(np.cumsum(rng.normal(0, 0.02, (len(days), len(TICKERS) + 1)), axis=0))
        self.prices = pd.concat([
            pd.DataFrame({"date": days, "ticker": t, "open": px[:, i], "close": px[:, i], "volume": 1e6})
            for i, t in enumerate([*TICKERS, "SPY"])], ignore_index=True)
        cols = [c for c in FUND_COLS if c not in ("ticker", "dimension", "calendardate", "date", "reportperiod")]
        rows = []
        for i, t in enumerate(TICKERS):
            for q, rp in enumerate(pd.date_range("2018-12-31", periods=12, freq="QE")):
                vals = {c: float(100 + 5 * q + i + 20 * rng.normal()) for c in cols}
                vals.update(dps=0.5 + 0.1 * abs(rng.normal()), sharesbas=1e6, fcf=100 * rng.normal(),
                            grossmargin=0.4 + 0.01 * rng.normal(), ebitdamargin=0.2 + 0.02 * rng.normal(),
                            bvps=20 + q + rng.normal())
                for dim in ("ARQ", "ART"):
                    rows.append({"ticker": t, "dimension": dim, "calendardate": rp,
                                 "date": rp + pd.Timedelta(days=40), "reportperiod": rp, **vals})
        self.fundamentals = pd.DataFrame(rows)
        self.sec8k = pd.DataFrame([
            *({"ticker": "T00", "filed": d, "accepted": d.tz_localize("UTC") + pd.Timedelta(hours=21),
               "items": "2.02,9.01", "is_amendment": False}
              for d in pd.to_datetime(["2019-08-01", "2019-11-01", "2020-02-03", "2020-05-01", "2020-08-03"])),
            {"ticker": "T01", "filed": pd.Timestamp("2020-06-02"), "accepted": pd.Timestamp("2020-06-02 21:00", tz="UTC"),
             "items": "2.04", "is_amendment": False},
            {"ticker": "T02", "filed": pd.Timestamp("2020-06-03"), "accepted": pd.NaT,
             "items": "5.02", "is_amendment": False},
            {"ticker": "T04", "filed": pd.Timestamp("2020-06-03"), "accepted": pd.Timestamp("2020-06-03 12:00", tz="UTC"),
             "items": "8.01", "is_amendment": False}])
        self.form4 = pd.DataFrame([{"ticker": "T03", "filed": pd.Timestamp("2020-06-04"), "code": "S", "value": 1e5},
                                   {"ticker": "T03", "filed": pd.Timestamp("2020-06-05"), "code": "P", "value": 1e5}])

    def read(self, name):
        return getattr(self, name).copy()


def _panel():
    weeks = pd.date_range("2020-01-03", "2021-12-24", freq="7D")   # Fridays
    return pd.DataFrame({"date": np.repeat(weeks, len(TICKERS)), "ticker": TICKERS * len(weeks),
                         "f_mom": 0.1, "label_4w": 0.0})


def test_add_candidates_appends_every_ranked_x_column():
    panel = _panel()
    out = add_candidates(panel, _World())
    assert candidate_cols(out) == list(CANDIDATES) and len(CANDIDATES) == 30
    assert out.shape == (len(panel), panel.shape[1] + 30)
    pd.testing.assert_frame_equal(out[panel.columns], panel)          # untouched, same order
    x = out[candidate_cols(out)]
    assert x.notna().all().all() and (x.abs() <= 1).all().all()        # ranked to [-1, 1], neutral-filled
    assert (x.groupby(out["date"]).nunique() > 1).any().all(), "every candidate ranks somewhere"
    assert feature_cols(out) == ["f_mom"]                             # invisible until admitted
    # rebuilding replaces the x_ columns rather than duplicating them
    again = add_candidates(out, _World())
    pd.testing.assert_frame_equal(again, out)


def test_candidates_are_point_in_time():
    """T01's distress 8-K (accepted 2020-06-02) is invisible on the Friday
    before, ranks T01 top on the next, and rolls off after 26 weeks."""
    out = add_candidates(_panel(), _World())
    x = out.set_index(["date", "ticker"])["x_8k_distress_26w"]
    tied = x.groupby("date").nunique() == 1                            # nobody flagged: all tied
    assert tied.loc["2020-05-29"] and not tied.loc["2020-06-05"]
    assert x.loc[("2020-06-05", "T01")] == 1.0
    assert tied.loc[tied.index > "2020-12-31"].all()
