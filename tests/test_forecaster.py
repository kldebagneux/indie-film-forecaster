"""Run with:  python3 -m pytest"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import forecaster as fc

FILM = "Heretic"


@pytest.fixture(scope="module")
def df():
    return fc.load_films()


@pytest.fixture(scope="module")
def df_future_changed(tmp_path_factory):
    """Same data, but every film released after FILM earned 10x more."""
    raw = pd.read_csv(fc.DATA)
    date = raw.loc[raw.title == FILM, "release_date"].iloc[0]
    later = raw.release_date > date
    raw.loc[later, "revenue"] *= 10
    path = tmp_path_factory.mktemp("data") / "films.csv"
    raw.to_csv(path, index=False)
    return fc.load_films(path)


def test_backtest_does_not_see_the_future(df, df_future_changed):
    """A backtest forecast must not change when later films' results change."""
    a = fc.forecast_existing(df, FILM)
    b = fc.forecast_existing(df_future_changed, FILM)
    for key in ["forecast_central", "forecast_range_50pct", "forecast_range_80pct",
                "tier_probabilities", "main_drivers", "comparables"]:
        assert a[key] == b[key], key


def test_director_track_uses_only_earlier_films(df, df_future_changed):
    date = df.loc[df.title == FILM, "release_date"].iloc[0]
    cols = ["title", "director_track"]
    both = df[df.release_date <= date][cols].merge(df_future_changed[cols], on="title")
    assert len(both) == (df.release_date <= date).sum()
    assert np.allclose(both.director_track_x, both.director_track_y)


def test_rolling_backtest_trains_only_on_earlier_years(df):
    bt = fc.rolling_backtest(df, with_extras=False)
    assert not set(bt.year) & fc.SKIP_YEARS
    assert bt.year.min() == fc.FIRST_TEST_YEAR


def test_new_film_forecast_is_ordered(df):
    r = fc.forecast_new(df, {"title": "Test", "studio": "A24", "release_date": "2026-10-01",
                             "budget": 20e6, "genres": ["Horror"], "rating": "R", "runtime": 100,
                             "director": "Nobody", "is_sequel": False, "competitors": None})
    lo80, hi80 = r["forecast_range_80pct"]
    lo50, hi50 = r["forecast_range_50pct"]
    assert lo80 < lo50 < r["forecast_central"] < hi50 < hi80
    assert abs(sum(r["tier_probabilities"].values()) - 1) < 0.02
    assert len(r["comparables"]) == 5


def test_new_film_needs_all_details(df):
    with pytest.raises(ValueError, match="Missing details"):
        fc.forecast_new(df, {"title": "Incomplete", "budget": 1e6})


def test_inflation_puts_old_budgets_in_2025_dollars():
    assert fc.in_2025_dollars(1e6, 2025) == 1e6
    assert fc.in_2025_dollars(1e6, 2012) > 1.3e6
