"""Download the film data from TMDB into films.csv.

    export TMDB_API_KEY="your key"
    python3 fetch_films.py              # uses cached responses where it has them
    python3 fetch_films.py --refresh    # ignore the cache and download everything again

Every TMDB response is cached in .cache/tmdb/, so re-running is fast and doesn't
hit the API around 1,000 times. films_meta.json records when the data was pulled,
because TMDB numbers change over time.
"""
import argparse, hashlib, json, os, time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

BASE = "https://api.themoviedb.org/3"
ROOT = Path(__file__).parent
CACHE = ROOT / ".cache" / "tmdb"

# Exact TMDB studio IDs (Searchlight was "Fox Searchlight" before 2020)
STUDIOS = {
    "A24": 41077,
    "Neon": 90733,
    "Focus Features": 10146,
    "Searchlight Pictures": 127929,
    "Fox Searchlight": 43,
    "Annapurna Pictures": 13184,
    "Bleecker Street": 167986,
    "Blumhouse": 3172,
}

REFRESH = False


def get(path, **params):
    key = hashlib.sha1(json.dumps([path, sorted(params.items())]).encode()).hexdigest()
    cached = CACHE / f"{key}.json"
    if cached.exists() and not REFRESH:
        return json.loads(cached.read_text())
    r = requests.get(f"{BASE}{path}", params={"api_key": os.environ["TMDB_API_KEY"], **params}, timeout=30)
    r.raise_for_status()
    time.sleep(0.05)
    cached.write_text(r.text)
    return r.json()


def us_rating(film):
    for country in film["release_dates"]["results"]:
        if country["iso_3166_1"] == "US":
            for r in country["release_dates"]:
                if r["certification"]:
                    return r["certification"]
    return None


def competitors(release_date):
    """Number of other films opening in the US within 3 days either side."""
    if not release_date:
        return None
    day = datetime.strptime(release_date, "%Y-%m-%d")
    data = get("/discover/movie", region="US", with_release_type=3,
               **{"release_date.gte": (day - timedelta(days=3)).strftime("%Y-%m-%d"),
                  "release_date.lte": (day + timedelta(days=3)).strftime("%Y-%m-%d")})
    return max(data["total_results"] - 1, 0)


def fetch(refresh=False):
    """Download everything and write films.csv + films_meta.json. Also used by the online app."""
    global REFRESH
    REFRESH = refresh
    if not os.environ.get("TMDB_API_KEY"):
        raise SystemExit("Set TMDB_API_KEY first (free at themoviedb.org/settings/api).")
    CACHE.mkdir(parents=True, exist_ok=True)

    # Check that each ID really is the studio we think it is
    for name, studio_id in STUDIOS.items():
        print(f"{name} -> TMDB says: {get(f'/company/{studio_id}')['name']}")

    rows = []
    for name, studio_id in STUDIOS.items():
        page, total = 1, 1
        while page <= total:
            data = get("/discover/movie", with_companies=studio_id,
                       **{"primary_release_date.gte": "2012-01-01"}, page=page)
            total = data["total_pages"]
            for m in data["results"]:
                f = get(f"/movie/{m['id']}", append_to_response="credits,keywords,release_dates")
                rows.append({
                    "title": f["title"],
                    "studio": name,
                    "release_date": f["release_date"],
                    "runtime": f["runtime"],
                    "genres": "|".join(g["name"] for g in f["genres"]),
                    "language": f["original_language"],
                    "director": next((c["name"] for c in f["credits"]["crew"] if c["job"] == "Director"), None),
                    "top_cast": "|".join(c["name"] for c in f["credits"]["cast"][:3]),
                    "keywords": "|".join(k["name"] for k in f["keywords"]["keywords"]),
                    "is_sequel": f["belongs_to_collection"] is not None,
                    "rating": us_rating(f),
                    "competitors": competitors(f["release_date"]),
                    "budget": f["budget"],
                    "revenue": f["revenue"],
                })
            page += 1
        print(f"{name}: done, {len(rows)} films so far")

    df = pd.DataFrame(rows).drop_duplicates("title")
    df.to_csv(ROOT / "films.csv", index=False)
    usable = df[(df.budget > 0) & (df.revenue > 0)]
    (ROOT / "films_meta.json").write_text(json.dumps(
        {"pulled_on": str(date.today()), "films": len(df), "with_budget_and_revenue": len(usable),
         "source": "TMDB API", "studios": STUDIOS}, indent=2))
    print(f"Total films: {len(df)}, usable: {len(usable)}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--refresh", action="store_true", help="ignore cached responses")
    fetch(p.parse_args().refresh)


if __name__ == "__main__":
    main()
