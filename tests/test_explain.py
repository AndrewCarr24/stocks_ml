"""stocks-ml explain: the summary and the plot from per-year SHAP tables."""
import numpy as np
import pandas as pd

import stocks_ml.explain as ex


def test_render_ranks_features_by_mean_abs_shap_and_writes_plot_and_table(tmp_path):
    by = pd.DataFrame({2007: {"f_vol_12w": 0.30, "f_mom_4w": 0.10, "f_sf_roe": 0.05},
                       2008: {"f_vol_12w": 0.20, "f_mom_4w": 0.12, "f_sf_roe": 0.01}})
    sg = pd.DataFrame({2007: {"f_vol_12w": 0.4, "f_mom_4w": -0.3, "f_sf_roe": 0.1},
                       2008: {"f_vol_12w": 0.2, "f_mom_4w": -0.5, "f_sf_roe": 0.0}})
    res = {"by_year": by, "sign": sg, "recipe": {"label": "label_4w_sector_rank", "train_years": 8},
           "weeks": ["2007-01-05", "2008-01-04"]}
    summ = ex.render(res, png=tmp_path / "s.png", md=tmp_path / "s.md", top=2)
    assert list(summ.index) == ["f_vol_12w", "f_mom_4w", "f_sf_roe"]
    assert summ.loc["f_vol_12w", "mean_abs_shap"] == 0.25 and summ.loc["f_mom_4w", "sign_corr"] == -0.4
    assert abs(summ["share_of_total"].sum() - 1) < 1e-9
    assert (tmp_path / "s.png").exists() and (tmp_path / "s.png").stat().st_size > 1000
    md = (tmp_path / "s.md").read_text()
    assert "| 1 | 12-week volatility (`f_vol_12w`)" in md and "| 3 | return on equity (`f_sf_roe`)" in md


def test_year_table_gives_mean_abs_shap_and_a_sign_per_feature():
    rng = np.random.default_rng(0)
    n = 60
    rows = pd.DataFrame({"ticker": [f"T{i}" for i in range(n)], "a": rng.normal(size=n), "b": rng.normal(size=n)})
    shap = pd.DataFrame({"a": rows["a"].values * 0.5, "b": -rows["b"].values * 0.2}, index=rows["ticker"].values)
    tab = ex.year_table(shap, rows, ["a", "b"])
    assert tab.loc["a", "mean_abs_shap"] > tab.loc["b", "mean_abs_shap"]
    assert tab.loc["a", "sign_corr"] > 0.99 and tab.loc["b", "sign_corr"] < -0.99


def test_render_top3_averages_the_picks_contributions_and_appends_the_table(tmp_path):
    picks = [{"year": 2007, "week": "2007-01-05", "rank": r, "ticker": f"T{r}", "score": 1.0 - 0.1 * r,
              "shap": pd.Series({"f_vol_12w": 0.3 * r, "f_mom_4w": -0.1, "f_sf_roe": 0.0})} for r in (1, 2, 3)]
    res = {"picks": picks, "recipe": {"label": "label_4w_sector_rank", "train_years": 8}, "weeks": ["2007-01-05"]}
    md = tmp_path / "s.md"; md.write_text("# head\n")
    t3 = ex.render_top3(res, png=tmp_path / "t.png", md=md, top=3)
    assert t3.index[0] == "f_vol_12w" and t3.loc["f_vol_12w", "mean_signed_shap"] == 0.6
    assert t3.loc["f_mom_4w", "share_of_picks_pushed_up"] == 0.0
    assert (tmp_path / "t.png").exists()
    text = md.read_text()
    assert "## The top-3 picks" in text and "| 2007-01-05 | 1 | T1 |" in text and text.startswith("# head")
    assert ex.render_top3({"picks": [], "recipe": {}, "weeks": []}, png=tmp_path / "n.png", md=md).empty


def test_render_picks_draws_a_random_sample_of_decompositions(tmp_path):
    picks = [{"year": 2007 + i, "week": f"{2007 + i}-01-05", "rank": 1, "ticker": f"T{i}", "score": 0.5,
              "shap": pd.Series({f"f_{j}": (j - 5) * 0.01 * (i + 1) for j in range(15)})} for i in range(10)]
    res = {"picks": picks, "recipe": {}, "weeks": []}
    sample = ex.render_picks(res, png=tmp_path / "p.png", n=4, seed=1)
    assert len(sample) == 4 and len({p["ticker"] for p in sample}) == 4
    assert (tmp_path / "p.png").exists() and (tmp_path / "p.png").stat().st_size > 1000
    assert ex.render_picks(res, png=tmp_path / "q.png", n=4, seed=1) == sample          # seeded
