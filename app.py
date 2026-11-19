"""A simple form on top of the forecaster, so anyone can use it without touching code.

    streamlit run app.py

Online (Streamlit Community Cloud), the keys live in the app's private settings
("Secrets"), never in the code:

    TMDB_API_KEY = "..."        # used once, to download the film data on first start
    ANTHROPIC_API_KEY = "..."   # optional: Claude's written read
    BRIEFS_PER_DAY = 30         # optional: cap on Claude briefs per day (default 30)
    APP_PASSWORD = "..."        # optional: a password on top of Streamlit's own access control
"""
import datetime as dt
import hmac
import json
import os
import threading

import streamlit as st

st.set_page_config(page_title="Indie Film Forecaster", layout="centered")


# ---------- Settings: copy private secrets into the environment the scripts already read ----------
def secret(name, default=None):
    try:
        return st.secrets.get(name, default)
    except Exception:          # no secrets file when running locally
        return default


for key in ["TMDB_API_KEY", "ANTHROPIC_API_KEY"]:
    if secret(key) and not os.environ.get(key):
        os.environ[key] = str(secret(key))

# ---------- Optional password ----------
password = secret("APP_PASSWORD")
if password and not st.session_state.get("signed_in"):
    st.title("Indie Film Forecaster")
    typed = st.text_input("Password", type="password")
    if typed and hmac.compare_digest(typed, str(password)):
        st.session_state.signed_in = True
        st.rerun()
    elif typed:
        st.error("Wrong password.")
    st.stop()

import brief                 # noqa: E402  (imported after the keys are set)
import forecaster as fc      # noqa: E402


# ---------- Data: download it on first start if it isn't there ----------
if not fc.DATA.exists():
    if not os.environ.get("TMDB_API_KEY"):
        st.error("No film data yet. Add TMDB_API_KEY to the app's secrets, then reboot the app.")
        st.stop()
    with st.spinner("First start: downloading the film data from TMDB (a few minutes, only once)..."):
        import fetch_films
        fetch_films.fetch()


@st.cache_data
def load():
    df = fc.load_films()
    errors = fc.rolling_backtest(df, with_extras=False).error
    genres = sorted({g for gs in df.genres.dropna() for g in gs.split("|")})
    return df, errors, genres


@st.cache_data
def backtest(title):
    return fc.forecast_existing(load()[0], title)


# ---------- Claude briefs, with a daily cap so nobody runs up a bill ----------
@st.cache_resource
def usage():
    return {"day": None, "count": 0, "lock": threading.Lock()}


def take_brief_credit():
    u, limit = usage(), int(secret("BRIEFS_PER_DAY", 30))
    with u["lock"]:
        today = dt.date.today()
        if u["day"] != today:
            u["day"], u["count"] = today, 0
        if u["count"] >= limit:
            return False
        u["count"] += 1
        return True


@st.cache_data(show_spinner="Claude is writing the brief...")
def words_for(result_json):
    forecast = {k: v for k, v in json.loads(result_json).items() if k != "actual_revenue_for_checking"}
    return brief.write_words(forecast)


# ---------- Page ----------
df, errors, all_genres = load()

st.title("Indie Film Forecaster")
st.caption(f"Trained on {len(df)} releases from A24 and similar studios (2012–2025). "
           "Typically off by about 2x, so read the range, not just the number.")

with st.expander("How to read this"):
    st.markdown(
        "- **Central forecast**: the single best estimate of worldwide box office.\n"
        "- **Ranges**: built from how far off the forecasts were on past films the model had not seen. "
        "In testing, the 50% range held the real result 44% of the time and the 80% range 78%.\n"
        "- **Odds**: from the same past misses, so they can lean lower than the central forecast "
        "(past misses were more often too high, mostly on small releases).\n"
        "- **Check the model on a past film** to see a forecast made using only films released before it, "
        "next to what the film actually earned.")

mode = st.radio("What do you want to do?", ["Forecast a new film", "Check the model on a past film"],
                horizontal=True, label_visibility="collapsed")

if mode == "Forecast a new film":
    studios = sorted(df.studio.unique())
    with st.form("film"):
        title = st.text_input("Title", "Behemoth!")
        c1, c2 = st.columns(2)
        studio = c1.selectbox("Studio", studios, index=studios.index("Searchlight Pictures"))
        release = c2.date_input("Release date", dt.date(2026, 12, 2))
        budget = c1.number_input("Budget ($M)", min_value=0.1, value=36.0, step=1.0)
        runtime = c2.number_input("Runtime (minutes)", min_value=60, max_value=240, value=127)
        genres = c1.multiselect("Genres", all_genres, default=["Music", "Drama"])
        rating = c2.selectbox("US rating", ["R", "PG-13", "PG", "G", "NR"])
        director = c1.text_input("Director", "Tony Gilroy")
        competitors = c2.number_input("Other films opening that week (0 = use the usual for that month)",
                                      min_value=0, value=0)
        is_sequel = st.checkbox("Sequel or part of a franchise")
        submitted = st.form_submit_button("Forecast", type="primary")
    if submitted:
        info = {"title": title, "studio": studio, "release_date": str(release), "budget": budget * 1e6,
                "genres": genres, "rating": rating, "runtime": runtime, "director": director.strip(),
                "is_sequel": is_sequel, "competitors": competitors or None}
        try:
            st.session_state.result = fc.forecast_new(df, info, errors)
        except ValueError as e:
            st.error(str(e))
else:
    choices = sorted(df[df.year >= 2017].title)
    title = st.selectbox("Pick a released film. The model only sees films released before it.",
                         choices, index=choices.index("Heretic"))
    st.session_state.result = backtest(title)

result = st.session_state.get("result")
if result:
    words = None
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        if st.toggle("Add Claude's written read for the marketing team"):
            key = json.dumps(result, sort_keys=True)
            if key in st.session_state.setdefault("written", set()) or take_brief_credit():
                try:
                    words = words_for(key)
                    st.session_state.written.add(key)
                except Exception as e:
                    st.warning(f"Couldn't get the written read: {e}")
            else:
                st.warning("Today's limit of written briefs is used up. The numbers below are still live.")
    else:
        st.info("Claude's written read is off (no Anthropic key set).")

    page = brief.render_html(result, words)
    st.download_button("Download this brief (HTML)", page, file_name=f"{result['film']} brief.html", mime="text/html")
    st.iframe(page)   # every film detail and Claude's text are HTML-escaped by render_html
