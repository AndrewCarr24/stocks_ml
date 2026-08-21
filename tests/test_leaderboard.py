from stocks_ml.backtest.leaderboard import rank_select


def _e(model, config, earnings, sr=0.5):
    return {"model": model, "config": config, "earnings": earnings, "sr": sr}


def test_rank_select_three_two_one():
    entries = [
        _e("A", "a1", 900), _e("A", "a2", 800), _e("A", "a3", 700), _e("A", "a4", 600),
        _e("B", "b1", 850), _e("B", "b2", 650), _e("B", "b3", 640),
        _e("C", "c1", 500), _e("C", "c2", 400),
        _e("D", "d1", 300),
    ]
    out = rank_select(entries)
    assert [(r["model"], r["config"]) for r in out] == [
        ("A", "a1"), ("A", "a2"), ("A", "a3"),
        ("B", "b1"), ("B", "b2"),
        ("C", "c1")]


def test_models_ranked_by_best_config_and_sr_tiebreak():
    entries = [
        _e("A", "a1", 700), _e("B", "b1", 700, sr=0.9), _e("C", "c1", 100),
    ]
    out = rank_select(entries)
    assert out[0]["model"] == "B"          # tie on earnings -> higher SR first
    assert [r["model"] for r in out] == ["B", "A", "C"]
