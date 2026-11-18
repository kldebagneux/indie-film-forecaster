"""Forecast a film from the command line.

    python3 predict.py films/behemoth.json      # a new, unreleased film
    python3 predict.py --backtest "Heretic"     # a released film, using only earlier data

The forecast is printed and saved to output/<film>.forecast.json, ready for brief.py.
"""
import argparse, json, re, sys
from pathlib import Path

import forecaster as fc


def slug(title):
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def save(result):
    path = fc.ROOT / "output" / f"{slug(result['film'])}.forecast.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("film_file", nargs="?", help="JSON file with the film's details (see films/)")
    p.add_argument("--backtest", metavar="TITLE", help="forecast a released film using only earlier films")
    a = p.parse_args()
    if bool(a.film_file) == bool(a.backtest):
        p.error("give either a film file or --backtest TITLE")

    df = fc.load_films()
    try:
        if a.backtest:
            result = fc.forecast_existing(df, a.backtest)
        else:
            result = fc.forecast_new(df, json.loads(Path(a.film_file).read_text()))
    except ValueError as e:
        sys.exit(f"Error: {e}")

    print(json.dumps(result, indent=2))
    print(f"\nSaved {save(result).relative_to(fc.ROOT)}")


if __name__ == "__main__":
    main()
