"""Core of the forecaster: data prep, features, models and the forecast itself.

Every feature uses only information you would have before a film's release.
The other scripts (predict.py, evaluate.py, brief.py, app.py) all import from here.
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
DATA = ROOT / "films.csv"

# US consumer prices (CPI-U, yearly average, BLS). Used to put budgets and past
# box office in 2025 dollars, so a 2013 budget is comparable with a 2025 one.
# 2025 is approximate and 2026 is a projection (only used for unreleased films).
CPI = {2012: 229.6, 2013: 233.0, 2014: 236.7, 2015: 237.0, 2016: 240.0, 2017: 245.1,
       2018: 251.1, 2019: 255.7, 2020: 258.8, 2021: 271.0, 2022: 292.7, 2023: 304.7,
       2024: 313.7, 2025: 322.0, 2026: 330.0}
BASE_YEAR = 2025

GENRES = ["Horror", "Drama", "Comedy", "Thriller"]
FEATURES = ["log_budget", *GENRES, "is_R", "is_sequel", "runtime", "month",
            "competitors", "director_track", "has_track"]
COMP_FEATURES = ["log_budget", *GENRES, "is_R", "is_sequel"]
TIERS = ["underperform (<1x budget)", "solid (1-3x)", "breakout (>3x)"]

# Years scored in the headline evaluation. 2020 and 2021 are left out as test years
# because theaters were closed or half-empty; those films are still used for training.
SKIP_YEARS = {2020, 2021}
HEADLINE_YEARS = [2019, 2022, 2023, 2024, 2025]
FIRST_TEST_YEAR = 2015
NO_TRACK = np.log(2)   # director track value used when a director has no earlier film   # earliest year with enough earlier films (2012-14) to train on

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]


def money(x):
    return f"${x / 1e6:.0f}M" if x >= 1e6 else f"${x / 1e3:.0f}K"


def in_2025_dollars(amount, year):
    year = min(max(int(year), min(CPI)), max(CPI))
    return amount * CPI[BASE_YEAR] / CPI[year]


def _basic_features(f):
    """Features that only depend on the film itself (shared by past and new films)."""
    f["year"] = f.release_date.dt.year
    f["month"] = f.release_date.dt.month
    f["real_budget"] = [in_2025_dollars(b, y) for b, y in zip(f.budget, f.year)]
    f["log_budget"] = np.log(f.real_budget)
    f["is_R"] = (f.rating == "R").astype(int)
    f["is_sequel"] = f.is_sequel.astype(bool).astype(int)
    for g in GENRES:
        f[g] = f.genres.fillna("").str.contains(g).astype(int)
    return f


def load_films(path=DATA, last_year=2025):
    """Load films.csv and build every feature. Only films with final box office are kept."""
    df = pd.read_csv(path)
    df = df[(df.budget > 0) & (df.revenue >= 1e6)].copy()   # under $1M: mostly streaming-first
    df["release_date"] = pd.to_datetime(df.release_date)
    df = df[df.release_date.dt.year <= last_year]            # later films may still be in theaters
    df = df.sort_values("release_date").reset_index(drop=True)

    df = _basic_features(df)
    df["runtime"] = df.runtime.where(df.runtime > 0)          # TMDB uses 0 for "unknown"
    df["runtime"] = df.runtime.fillna(df.runtime.median())
    df["multiple"] = df.revenue / df.budget
    df["log_multiple"] = np.log(df.multiple)                  # what the main model predicts
    df["tier"] = pd.cut(df.multiple, [0, 1, 3, np.inf], labels=[0, 1, 2]).astype(int)

    # Director track record: how many times their budget the director's EARLIER films
    # earned, on average. Directors with no earlier film get a fixed neutral value
    # (about 2x, the typical multiple), so nothing from later films leaks in.
    tracks = [df[(df.release_date < row.release_date) & (df.director == row.director)].log_multiple.mean()
              for row in df.itertuples()]
    df["has_track"] = pd.notna(tracks).astype(int)
    df["director_track"] = pd.Series(tracks, index=df.index).fillna(NO_TRACK)
    return df


def make_models():
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10)),
        "gb": HistGradientBoostingRegressor(max_iter=100, learning_rate=0.05, max_depth=2,
                                            min_samples_leaf=10, random_state=0),
        "tiers": HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_depth=2,
                                                min_samples_leaf=10, random_state=0),
    }


def rolling_backtest(df, with_extras=True):
    """For each year, train only on films from earlier years and predict that year.

    Returns one row per test film. The errors from this are what the forecast
    range is built from, so the range reflects how far off forecasts are on
    films the model has never seen.
    """
    out = []
    for y in range(FIRST_TEST_YEAR, df.year.max() + 1):
        train, test = df[df.year < y], df[df.year == y]
        if y in SKIP_YEARS or test.empty:
            continue
        m = make_models()
        m["ridge"].fit(train[FEATURES], train.log_multiple)
        rows = test[["title", "studio", "year", "release_date", "budget", "revenue",
                     "multiple", "log_multiple", "tier"]].copy()
        rows["pred"] = m["ridge"].predict(test[FEATURES])
        if with_extras:
            m["gb"].fit(train[FEATURES], train.log_multiple)
            m["tiers"].fit(train[FEATURES], train.tier)
            rows["pred_gb"] = m["gb"].predict(test[FEATURES])
            rows["pred_base"] = np.log(np.median(train.multiple))
            rows["pred_tier"] = m["tiers"].predict(test[FEATURES])
            rows["base_tier"] = train.tier.mode()[0]
        out.append(rows)
    bt = pd.concat(out, ignore_index=True)
    bt["error"] = bt.log_multiple - bt.pred
    return bt


def ranges_from_errors(errors):
    """Middle-50% and middle-80% ranges (as log-multiple offsets) from past forecast errors."""
    p10, p25, p75, p90 = np.percentile(errors, [10, 25, 75, 90])
    return (p25, p75), (p10, p90)


def _describe(feature, value, film, past):
    """A plain-English statement of one input, for the drivers list."""
    if feature == "log_budget":
        return f"Budget of {money(film.budget.iloc[0])} (typical here: {money(past.budget.median())})"
    if feature in GENRES:
        return f"{'It is' if value else 'It is not'} a {feature.lower()} film"
    if feature == "is_R":
        return "Rated R" if value else f"Rated {film.rating.iloc[0]}, not R"
    if feature == "is_sequel":
        return "Part of a franchise" if value else "Not a sequel or franchise film"
    if feature == "runtime":
        return f"Runtime of {value:.0f} min (typical: {past.runtime.median():.0f} min)"
    if feature == "month":
        return f"Released in {MONTHS[int(value) - 1]}"
    if feature == "competitors":
        return f"{value:.0f} other films opening the same week (typical: {past.competitors.median():.0f})"
    if feature == "director_track":
        if film.has_track.iloc[0]:
            return f"Director's earlier films earned about {np.exp(value):.1f}x their budget"
        return "No earlier films by this director in the data"
    if feature == "has_track":
        return "Director has earlier films in the data" if value else "Director is new to the data"
    return feature


def forecast(film, past, errors, actual=None):
    """Forecast one film.

    film:   one-row table with the film's features
    past:   films released before it (the training data)
    errors: out-of-sample forecast errors from earlier years (for the range)
    """
    m = make_models()
    m["ridge"].fit(past[FEATURES], past.log_multiple)
    m["tiers"].fit(past[FEATURES], past.tier)

    pred = m["ridge"].predict(film[FEATURES])[0]
    budget = float(film.budget.iloc[0])
    (lo50, hi50), (lo80, hi80) = ranges_from_errors(errors)

    # What pushed the forecast up or down, compared with an average film in the data.
    # On the log scale effects add up, so exp(effect) is a multiplier on the forecast.
    scaler, ridge = m["ridge"].named_steps.values()
    effects = scaler.transform(film[FEATURES])[0] * ridge.coef_
    drivers = []
    for feat, eff in sorted(zip(FEATURES, effects), key=lambda x: -abs(x[1]))[:5]:
        drivers.append({
            "factor": _describe(feat, film[feat].iloc[0], film, past),
            "effect": "raises" if eff > 0 else "lowers",
            "multiplier": round(float(np.exp(eff)), 2),
        })

    # The 5 most similar past films (on budget, genre, rating and franchise)
    sc = StandardScaler().fit(past[COMP_FEATURES])
    nn = NearestNeighbors(n_neighbors=5).fit(sc.transform(past[COMP_FEATURES]))
    _, idx = nn.kneighbors(sc.transform(film[COMP_FEATURES]))
    comps = past.iloc[idx[0]]

    result = {
        "film": film.title.iloc[0],
        "studio": film.studio.iloc[0],
        "release_date": str(film.release_date.iloc[0].date()),
        "budget": int(budget),
        "genres": film.genres.iloc[0],
        "rating": film.rating.iloc[0],
        "director": film.director.iloc[0],
        "competing_releases_that_week": int(film.competitors.iloc[0]),
        "forecast_most_likely": round(budget * np.exp(pred)),
        "forecast_multiple": round(float(np.exp(pred)), 2),
        "forecast_range_50pct": [round(budget * np.exp(pred + lo50)), round(budget * np.exp(pred + hi50))],
        "forecast_range_80pct": [round(budget * np.exp(pred + lo80)), round(budget * np.exp(pred + hi80))],
        "range_based_on": f"forecast errors on {len(errors)} past films the model had not seen",
        "trained_on_films": len(past),
        "tier_probabilities": dict(zip(TIERS, np.round(m["tiers"].predict_proba(film[FEATURES])[0], 2).tolist())),
        "main_drivers": drivers,
        "comparables": [
            {"title": c.title, "studio": c.studio, "year": int(c.year), "budget": int(c.budget),
             "revenue": int(c.revenue), "multiple": round(c.revenue / c.budget, 1)}
            for c in comps.itertuples()
        ],
    }
    if actual is not None:
        result["actual_revenue_for_checking"] = int(actual)
    return result


def forecast_existing(df, title):
    """Backtest one released film, using only films (and errors) from before it."""
    film = df[df.title == title]
    if film.empty:
        raise ValueError(f"'{title}' isn't in the data. Check the spelling, or forecast it as a new film.")
    date, year = film.release_date.iloc[0], film.year.iloc[0]
    past = df[df.release_date < date]
    errors = rolling_backtest(df[df.year < year], with_extras=False).error
    if len(errors) < 20:
        raise ValueError(f"Too few earlier test films to build a range for a {year} film. Pick a film from 2017 on.")
    return forecast(film, past, errors, actual=film.revenue.iloc[0])


REQUIRED = ["title", "studio", "release_date", "budget", "genres", "rating", "runtime", "director", "is_sequel"]


def forecast_new(df, info, errors=None):
    """Forecast an unreleased film from a dict of its details (see films/*.json)."""
    missing = [k for k in REQUIRED if k not in info]
    if missing:
        raise ValueError(f"Missing details: {', '.join(missing)}")
    if info["budget"] <= 0:
        raise ValueError("Budget must be a positive number of dollars.")
    info = dict(info)
    if isinstance(info["genres"], list):
        info["genres"] = "|".join(info["genres"])

    f = pd.DataFrame([info])
    f["release_date"] = pd.to_datetime(f.release_date)
    f = _basic_features(f)
    # Director track record from the data (every film in the data is earlier)
    earlier = df[df.director == info["director"]]
    f["has_track"] = int(len(earlier) > 0)
    f["director_track"] = earlier.log_multiple.mean() if len(earlier) else NO_TRACK
    # Unknown competition: use the usual number for that month
    if info.get("competitors") is None:
        f["competitors"] = int(df[df.month == f.month.iloc[0]].competitors.median())
    if errors is None:
        errors = rolling_backtest(df, with_extras=False).error
    return forecast(f, df, errors)
