"""A simple form on top of the forecaster, so anyone can use it without touching code.

    streamlit run app.py
"""
import datetime as dt
import os

import streamlit as st
import streamlit.components.v1 as components

import brief
import forecaster as fc

st.set_page_config(page_title="Indie film forecaster", page_icon="🎬", layout="centered")


@st.cache_data
def load():
    df = fc.load_films()
    errors = fc.rolling_backtest(df, with_extras=False).error
    genres = sorted({g for gs in df.genres.dropna() for g in gs.split("|")})
    return df, errors, genres


@st.cache_data
def backtest(title):
    return fc.forecast_existing(load()[0], title)


@st.cache_data(show_spinner="Claude is writing the brief...")
def words_for(result_json):
    import json
    forecast = {k: v for k, v in json.loads(result_json).items() if k != "actual_revenue_for_checking"}
    return brief.write_words(forecast)


df, errors, all_genres = load()

st.title("Indie film box office forecaster")
st.caption(f"Trained on {len(df)} releases from A24 and similar studios (2012–2025). "
           "Typically off by about 2x, so read the range, not just the number.")

mode = st.radio("What do you want to do?", ["Forecast a new film", "Check the model on a past film"],
                horizontal=True, label_visibility="collapsed")

if mode == "Forecast a new film":
    with st.form("film"):
        title = st.text_input("Title", "Behemoth!")
        c1, c2 = st.columns(2)
        studio = c1.selectbox("Studio", sorted(df.studio.unique()), index=sorted(df.studio.unique()).index("Searchlight Pictures"))
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
    import json
    words = None
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    if has_key:
        if st.toggle("Add Claude's written read for the marketing team"):
            try:
                words = words_for(json.dumps(result, sort_keys=True))
            except Exception as e:
                st.warning(f"Couldn't get the written read: {e}")
    else:
        st.info("Set ANTHROPIC_API_KEY before starting the app to add Claude's written read.")

    page = brief.render_html(result, words)
    st.download_button("Download this brief (HTML)", page, file_name=f"{result['film']} brief.html", mime="text/html")
    components.html(page, height=2300 if words else 1700, scrolling=True)
