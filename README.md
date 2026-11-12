# Indie film box office forecaster

A small project to see how far you can get forecasting a film's box office with public data, and then have an AI agent turn the forecast into something a marketing team could actually read.

## What it does

You give it a film's details before release (budget, genre, rating, release date, director, and so on). It gives you a likely box office number, a range, the odds of the film underperforming, doing fine or breaking out, and the five most similar past films. Then Claude writes a one page brief from that, in plain English.

There are three scripts:

1. `fetch_films.py` pulls about 430 films from 2012 to 2026 from TMDB, from A24 and studios that make similar films: Neon, Focus Features, Searchlight, Annapurna, Bleecker Street and Blumhouse.
2. `model.py` cleans the data and makes the forecast. It predicts how many times its budget a film will earn, using only things you'd know before release.
3. `brief.py` sends the forecast to Claude and builds a one page report (`brief.html`). Claude only writes the text. All the numbers and charts come from Python, so it can't get a figure wrong.

## The model

I used three simple models, each for one job:

- A linear model (ridge regression) for the main number and the range
- A gradient boosting classifier for the odds of each outcome
- Nearest neighbours to find the five closest past films

To test it fairly, I went year by year from 2019 to 2025. Each year, the model only learned from films released before that year.

| Method | Typical error |
|---|---|
| Simple rule (budget times the usual multiple) | 2.51x |
| Linear model | 2.20x |
| Gradient boosting | 2.47x |

So the model beats the rule of thumb, but not by a huge margin. The outcome classifier was right 45% of the time, against 34% if you just always guess the most common outcome.

Two examples:

- **Heretic (A24, 2024).** Using only films released before it, the model said about $27M, probably between $13M and $58M. It made $52M.
- **Behemoth! (Searchlight, December 2026).** Not out yet. The model says about $35M, probably between $17M and $73M, with a 62% chance of a solid result. We'll see.

## Things I tried that didn't work

- Star power, meaning the past box office of the top three actors. It made no difference, and it wasn't significant (p = 0.45).
- Keywords like "supernatural horror" or "based on a true story". They looked significant on past films (p = 0.046) but made forecasts on new films worse. Classic overfitting, so I dropped them.
- Removing the weaker features. Most weren't significant on their own, but taking them out made forecasts worse, so I kept them.
- Gradient boosting for the main number. With only about 200 films, the simpler model held up better.

## Limits

- It's still usually off by around 2x. Public data just doesn't capture what makes a film take off.
- Box office isn't profit. Theaters keep a big share, and marketing costs aren't included.
- TMDB lists production companies, so some films a studio only distributed are missing.
- I left out films that made under $1M, since most of those went straight to streaming.

## What a studio's own data would add

This is where most of the accuracy would come from:

- **Presales.** Tickets sold a couple of weeks before release, compared with similar films at the same point.
- **Trailers and clips.** Views in the first week after each trailer or clip drops, how fast that grows, and how people react in the comments. It would have to be measured before release, otherwise the model is just learning which films ended up being hits.
- **Social media.** Follower growth, mentions and engagement in the weeks before release, on TikTok, Instagram, X and Letterboxd.
- **Screenwriter track record**, the same way I did it for directors. I'd keep the scripts themselves out of it. The point is to plan the release, not to judge the creative work.

The next step would be a simple form on top of it, so someone could get a forecast without touching any code.

## Running it

    pip3 install requests pandas scikit-learn anthropic
    export TMDB_API_KEY="your key"
    export ANTHROPIC_API_KEY="your key"
    python3 fetch_films.py
    python3 model.py     # change NEW_FILM at the bottom to forecast another film
    python3 brief.py

*This product uses the TMDB API but is not endorsed or certified by TMDB.*