import json, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
warnings.filterwarnings("ignore")

# ---------- 1. Load and clean ----------
df = pd.read_csv("films.csv")
df = df[(df.budget > 0) & (df.revenue > 0)].copy()
df["release_date"] = pd.to_datetime(df["release_date"])
df["year"] = df.release_date.dt.year
df["month"] = df.release_date.dt.month
df = df[df.year <= 2025]       # 2026 films may still be in theaters: revenue not final
df = df[df.revenue >= 1e6]     # keep real theatrical releases (drops streaming-first films like White Noise)
df = df.sort_values("release_date").reset_index(drop=True)

# ---------- 2. Features (only things known BEFORE release) ----------
df["log_budget"] = np.log(df.budget)
df["log_revenue"] = np.log(df.revenue)
df["multiple"] = df.revenue / df.budget                   # revenue / budget
df["log_multiple"] = np.log(df.multiple)                  # what the regression predicts
df["tier"] = pd.cut(df.multiple, [0, 1, 3, np.inf], labels=[0, 1, 2]).astype(int)
TIERS = ["underperform (<1x budget)", "solid (1-3x)", "breakout (>3x)"]

df["is_R"] = (df.rating == "R").astype(int)
df["is_sequel"] = df.is_sequel.astype(int)
for g in ["Horror", "Drama", "Comedy", "Thriller"]:
    df[g] = df.genres.str.contains(g).astype(int)

def director_track(row):   # average log revenue of the director's EARLIER films
    past = df[(df.director == row.director) & (df.release_date < row.release_date)]
    return past.log_revenue.mean() if len(past) else np.nan
df["director_track"] = df.apply(director_track, axis=1)
df["has_track"] = df.director_track.notna().astype(int)
df["director_track"] = df.director_track.fillna(df.log_revenue.median())

FEATURES = ["log_budget", "Horror", "Drama", "Comedy", "Thriller", "is_R", "is_sequel",
            "runtime", "month", "competitors", "director_track", "has_track"]
COMP_FEATURES = ["log_budget", "Horror", "is_R", "is_sequel"]

def make_models():
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10)),
        "gb": HistGradientBoostingRegressor(max_iter=100, learning_rate=0.05, max_depth=2,
                                            min_samples_leaf=10, random_state=0),
        "tiers": HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_depth=2,
                                                min_samples_leaf=10, random_state=0),
    }

# ---------- 3. Honest test: for each year, train only on earlier films ----------
scores = {k: [] for k in ["base", "ridge", "gb", "tier_guess", "tier_model"]}
for y in [2019, 2022, 2023, 2024, 2025]:
    tr, te = df[df.year < y], df[df.year == y]
    m = make_models()
    m["ridge"].fit(tr[FEATURES], tr.log_multiple)
    m["gb"].fit(tr[FEATURES], tr.log_multiple)
    m["tiers"].fit(tr[FEATURES], tr.tier)
    off = lambda pred_log_mult: np.median(np.exp(np.abs(pred_log_mult - te.log_multiple)))
    scores["base"].append(off(np.log(np.median(tr.multiple))))
    scores["ridge"].append(off(m["ridge"].predict(te[FEATURES])))
    scores["gb"].append(off(m["gb"].predict(te[FEATURES])))
    scores["tier_guess"].append(np.mean(te.tier == tr.tier.mode()[0]))
    scores["tier_model"].append(np.mean(te.tier == m["tiers"].predict(te[FEATURES])))

print(f"{len(df)} films. Each year 2019-2025 tested on a model trained only on earlier films.\n")
print("Revenue forecast (typical error, lower is better):")
print(f"  Simple rule (budget x usual multiple): x{np.mean(scores['base']):.2f}")
print(f"  Linear model (ridge):                  x{np.mean(scores['ridge']):.2f}")
print(f"  Gradient boosting:                     x{np.mean(scores['gb']):.2f}")
print("Tier prediction (accuracy, higher is better):")
print(f"  Always guess the most common tier:     {np.mean(scores['tier_guess']):.0%}")
print(f"  Gradient boosting classifier:          {np.mean(scores['tier_model']):.0%}")

# ---------- 4. Forecast a film ----------
def forecast(film, past, actual=None):
    """film: one-row table with the film's details. past: films released before it."""
    m = make_models()
    m["ridge"].fit(past[FEATURES], past.log_multiple)
    m["tiers"].fit(past[FEATURES], past.tier)

    # forecast + range from the model's typical past errors
    resid = past.log_multiple - m["ridge"].predict(past[FEATURES])
    pred = m["ridge"].predict(film[FEATURES])[0]
    budget = film.budget.iloc[0]
    lo, hi = np.percentile(resid, [25, 75])

    # what pushed the forecast up or down
    scaler, ridge = m["ridge"].named_steps.values()
    contrib = scaler.transform(film[FEATURES])[0] * ridge.coef_
    drivers = sorted(zip(FEATURES, contrib), key=lambda x: -abs(x[1]))[:4]

    # the 5 most similar past films
    sc = StandardScaler().fit(past[COMP_FEATURES])
    _, idx = NearestNeighbors(n_neighbors=5).fit(sc.transform(past[COMP_FEATURES])) \
                                            .kneighbors(sc.transform(film[COMP_FEATURES]))
    comps = past.iloc[idx[0]]

    result = {
        "film": film.title.iloc[0],
        "studio": film.studio.iloc[0],
        "release_date": str(film.release_date.iloc[0].date()),
        "budget": int(budget),
        "genres": film.genres.iloc[0],
        "rating": film.rating.iloc[0],
        "competing_releases_that_week": int(film.competitors.iloc[0]),
        "forecast_most_likely": round(budget * np.exp(pred)),
        "forecast_range_50pct": [round(budget * np.exp(pred + lo)), round(budget * np.exp(pred + hi))],
        "tier_probabilities": dict(zip(TIERS, np.round(m["tiers"].predict_proba(film[FEATURES])[0], 2).tolist())),
        "main_drivers": [{"feature": f, "effect": "up" if c > 0 else "down"} for f, c in drivers],
        "comparables": comps[["title", "studio", "year", "budget", "revenue"]].to_dict("records"),
    }
    if actual:
        result["actual_revenue_for_checking"] = int(actual)
    return result

def forecast_existing(title):
    """Backtest: forecast a film already in the data, using only films released before it."""
    film = df[df.title == title]
    if film.empty:
        raise SystemExit(f"'{title}' isn't in the data. Check the spelling, or use NEW_FILM instead.")
    past = df[df.release_date < film.release_date.iloc[0]]
    return forecast(film, past, actual=film.revenue.iloc[0])

def forecast_new(info):
    """A new, unreleased film: turn the details you typed into the same features."""
    f = pd.DataFrame([info])
    f["release_date"] = pd.to_datetime(f.release_date)
    f["month"] = f.release_date.dt.month
    f["log_budget"] = np.log(f.budget)
    f["is_R"] = int(info["rating"] == "R")
    f["is_sequel"] = int(info["is_sequel"])
    for g in ["Horror", "Drama", "Comedy", "Thriller"]:
        f[g] = int(g in info["genres"])
    # director's track record from the data (all films in the data are earlier)
    past_films = df[df.director == info["director"]]
    f["has_track"] = int(len(past_films) > 0)
    f["director_track"] = past_films.log_revenue.mean() if len(past_films) else df.log_revenue.median()
    # if you don't know the competition, use the typical number for that month
    if info.get("competitors") is None:
        f["competitors"] = int(df[df.month == f.month.iloc[0]].competitors.median())
    return forecast(f, df)

# ---------- 5. Choose the film to forecast ----------
# Option A: test the model on a film already released (shows the real result too)
# result = forecast_existing("Heretic")

# Option B: a new, unreleased film. Fill in its details:
NEW_FILM = {
    "title": "Behemoth!",
    "studio": "Searchlight Pictures",
    "release_date": "2026-12-02",
    "budget": 36_000_000,
    "genres": "Music|Drama",          # TMDB genres, separated by |
    "rating": "R",                    # R, PG-13 or PG
    "runtime": 127,                   # minutes
    "director": "Tony Gilroy",
    "is_sequel": False,
    "competitors": None,              # other films opening that week; None = use the usual for that month
}
result = forecast_new(NEW_FILM)

print("\n" + json.dumps(result, indent=2, default=int))
with open("forecast.json", "w") as f:
    json.dump(result, f, indent=2, default=int)