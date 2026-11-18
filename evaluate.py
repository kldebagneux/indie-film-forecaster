"""Score the forecaster honestly: each year is predicted by a model trained only on earlier years.

    python3 evaluate.py

Writes results/metrics.json, results/backtest.csv (every test film) and
results/backtest_2025.png (every 2025 film: forecast range vs what it earned).
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import forecaster as fc

OUT = fc.ROOT / "results"


def typical_error(actual_log, pred_log):
    """Median 'times off': 2.0 means the forecast was typically off by a factor of 2."""
    return float(np.median(np.exp(np.abs(actual_log - pred_log))))


def add_ranges(bt):
    """Give each test film the 50% and 80% ranges it would have had at the time:
    built only from errors on films from earlier test years."""
    for col in ["lo50", "hi50", "lo80", "hi80"]:
        bt[col] = np.nan
    for y in bt.year.unique():
        earlier = bt[bt.year < y].error
        if len(earlier) < 20:
            continue
        (lo50, hi50), (lo80, hi80) = fc.ranges_from_errors(earlier)
        rows = bt.year == y
        for col, off in [("lo50", lo50), ("hi50", hi50), ("lo80", lo80), ("hi80", hi80)]:
            bt.loc[rows, col] = bt.loc[rows, "pred"] + off
        odds = np.array([list(fc.tier_probabilities(pr, earlier).values()) for pr in bt.loc[rows, "pred"]])
        bt.loc[rows, ["err_p0", "err_p1", "err_p2"]] = odds
    bt["in50"] = (bt.log_multiple >= bt.lo50) & (bt.log_multiple <= bt.hi50)
    bt["in80"] = (bt.log_multiple >= bt.lo80) & (bt.log_multiple <= bt.hi80)
    return bt


def chart(year_rows, path, year):
    rows = year_rows.sort_values("revenue").reset_index(drop=True)
    b = rows.budget
    lo50, hi50 = b * np.exp(rows.lo50), b * np.exp(rows.hi50)
    lo80, hi80 = b * np.exp(rows.lo80), b * np.exp(rows.hi80)
    mid = b * np.exp(rows.pred)

    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    band, inside, outside = "#2a78d6", "#2a78d6", "#eb6834"
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(rows) + 1.6), dpi=160)
    fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
    y = np.arange(len(rows))
    ax.hlines(y, lo80, hi80, color=band, alpha=0.18, lw=7, label="80% range")
    ax.hlines(y, lo50, hi50, color=band, alpha=0.45, lw=7, label="50% range")
    ax.scatter(mid, y, marker="|", s=110, color=ink, lw=2, zorder=3, label="Forecast")
    colors = np.where(rows.in50, inside, outside)
    ax.scatter(rows.revenue, y, s=46, color=colors, edgecolor="#fcfcfb", lw=1.5, zorder=4)
    ax.scatter([], [], s=46, color=inside, label="Actual, inside 50% range")
    ax.scatter([], [], s=46, color=outside, label="Actual, outside 50% range")

    ax.set_xscale("log")
    ax.set_yticks(y, rows.title, fontsize=8, color=ink)
    ticks = [1e6, 3e6, 1e7, 3e7, 1e8, 3e8, 1e9]
    lo, hi = min(lo80.min(), rows.revenue.min()) * 0.8, max(hi80.max(), rows.revenue.max()) * 1.25
    ticks = [t for t in ticks if lo <= t <= hi]
    ax.set_xticks(ticks, [fc.money(t) for t in ticks], fontsize=8, color=muted)
    ax.set_xlim(lo, hi)
    ax.grid(axis="x", color=grid, lw=0.8); ax.set_axisbelow(True)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.tick_params(length=0)
    n50 = int(rows.in50.sum())
    ax.set_title(f"Every {year} film, forecast using only earlier films\n"
                 f"{n50} of {len(rows)} landed inside the 50% range, {int(rows.in80.sum())} inside the 80% range",
                 loc="left", fontsize=10.5, color=ink, pad=12)
    ax.set_xlabel("Worldwide box office (log scale)", fontsize=8, color=muted)
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.045), ncol=5, fontsize=7.5,
              frameon=False, labelcolor=muted, handletextpad=0.4, columnspacing=1.2)
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def brier(rows, prefix):
    """Brier score for the outcome odds: 0 is perfect, about 0.67 is knowing nothing."""
    p = rows[[f"{prefix}_p0", f"{prefix}_p1", f"{prefix}_p2"]].values
    return float(np.mean(((p - np.eye(3)[rows.tier.values]) ** 2).sum(1)))


def main():
    df = fc.load_films()
    bt = add_ranges(fc.rolling_backtest(df))
    head = bt[bt.year.isin(fc.HEADLINE_YEARS)]
    per_year = head.groupby("year")

    def avg(fn):
        return float(np.mean([fn(g) for _, g in per_year]))

    metrics = {
        "films_in_data": len(df),
        "test_years": fc.HEADLINE_YEARS,
        "test_films": len(head),
        "typical_error_times_off": {
            "simple_rule_budget_x_usual_multiple": round(avg(lambda g: typical_error(g.log_multiple, g.pred_base)), 2),
            "linear_model_ridge": round(avg(lambda g: typical_error(g.log_multiple, g.pred)), 2),
            "gradient_boosting": round(avg(lambda g: typical_error(g.log_multiple, g.pred_gb)), 2),
        },
        "outcome_odds_score_brier_lower_is_better": {
            "base_rates_only": round(brier(head, "base"), 3),
            "separate_classifier_old": round(brier(head, "clf"), 3),
            "from_tested_error_spread": round(brier(head, "err"), 3),
        },
        "outcome_most_likely_correct": {
            "always_guess_most_common": round(avg(lambda g: np.mean(g.tier == g.base_tier)), 2),
            "separate_classifier_old": round(avg(lambda g: np.mean(g.tier == g.pred_tier)), 2),
            "from_tested_error_spread": round(avg(lambda g: np.mean(g.tier == g[["err_p0", "err_p1", "err_p2"]].values.argmax(1))), 2),
        },
        "range_coverage": {
            "50pct_range_contained_actual": round(float(head.in50.mean()), 2),
            "80pct_range_contained_actual": round(float(head.in80.mean()), 2),
        },
        "per_year": {int(y): {"films": len(g),
                              "typical_error": round(typical_error(g.log_multiple, g.pred), 2),
                              "in_50pct_range": round(float(g.in50.mean()), 2)}
                     for y, g in per_year},
    }

    OUT.mkdir(exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    cols = ["title", "studio", "year", "budget", "revenue", "multiple", "pred", "lo50", "hi50", "in50", "in80"]
    table = bt[bt.year.isin(fc.HEADLINE_YEARS)][cols].copy()
    for c in ["pred", "lo50", "hi50"]:
        table[{"pred": "forecast", "lo50": "range50_low", "hi50": "range50_high"}[c]] = \
            (table.budget * np.exp(table[c])).round(-3).astype(int)
    table.drop(columns=["pred", "lo50", "hi50"]).round({"multiple": 2}) \
        .to_csv(OUT / "backtest.csv", index=False)
    chart(bt[bt.year == 2025], OUT / "backtest_2025.png", 2025)

    e, c = metrics["typical_error_times_off"], metrics["range_coverage"]
    print(f"{len(df)} films. {len(head)} test films from {', '.join(map(str, fc.HEADLINE_YEARS))}, "
          "each predicted by a model trained only on earlier years.\n")
    print("Box office forecast (typical error, lower is better):")
    print(f"  Simple rule (budget x usual multiple): x{e['simple_rule_budget_x_usual_multiple']:.2f}")
    print(f"  Linear model (ridge):                  x{e['linear_model_ridge']:.2f}")
    print(f"  Gradient boosting:                     x{e['gradient_boosting']:.2f}")
    o = metrics["outcome_odds_score_brier_lower_is_better"]
    print("Outcome odds (Brier score, lower is better):")
    print(f"  Base rates only:                       {o['base_rates_only']:.3f}")
    print(f"  Separate classifier (old):             {o['separate_classifier_old']:.3f}")
    print(f"  From the tested error spread (used):   {o['from_tested_error_spread']:.3f}")
    print("Are the ranges honest?")
    print(f"  50% range contained the actual result: {c['50pct_range_contained_actual']:.0%} of films")
    print(f"  80% range contained the actual result: {c['80pct_range_contained_actual']:.0%} of films")
    print(f"\nSaved results/metrics.json, results/backtest.csv, results/backtest_2025.png")


if __name__ == "__main__":
    main()
