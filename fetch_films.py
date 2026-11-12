import os, requests, time, pandas as pd
from datetime import datetime, timedelta

KEY = os.environ["TMDB_API_KEY"]
BASE = "https://api.themoviedb.org/3"

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

def get(path, **params):
    r = requests.get(f"{BASE}{path}", params={"api_key": KEY, **params})
    r.raise_for_status()
    time.sleep(0.05)
    return r.json()

def us_rating(film):
    for country in film["release_dates"]["results"]:
        if country["iso_3166_1"] == "US":
            for r in country["release_dates"]:
                if r["certification"]:
                    return r["certification"]
    return None

def competitors(date):
    if not date:
        return None
    day = datetime.strptime(date, "%Y-%m-%d")
    data = get("/discover/movie", region="US", with_release_type=3,
               **{"release_date.gte": (day - timedelta(days=3)).strftime("%Y-%m-%d"),
                  "release_date.lte": (day + timedelta(days=3)).strftime("%Y-%m-%d")})
    return max(data["total_results"] - 1, 0)

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
df.to_csv("films.csv", index=False)
usable = df[(df.budget > 0) & (df.revenue > 0)]
print(f"Total films: {len(df)}, usable: {len(usable)}")