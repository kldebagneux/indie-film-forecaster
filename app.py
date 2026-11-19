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

st.set_page_config(page_title="Indie film forecaster", page_icon="🎬", layout="centered")


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
    st.title("Indie film box office forecaster")
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
has_claude = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def show(result, slot):
    """The answer: three key numbers first, then Claude's brief (optional), then the full page."""
    budget, central = result["budget"], result["forecast_central"]
    low, high = result["forecast_range_50pct"]
    odds = list(result["tier_probabilities"].values())

    st.subheader(f"Forecast for {result['film']}")
    def usd(x): return fc.money(x).replace("$", "\\$")   # a bare $ would turn into maths formatting

    c1, c2, c3 = st.columns(3)
    note = dict(delta_color="off", delta_arrow="off", border=True)
    c1.metric("Central forecast", usd(central), f"{central / budget:.1f}x its budget", **note)
    c2.metric("Likely range", f"{usd(low)}–{usd(high)}", "half of outcomes land here", **note)
    c3.metric("Chance of a breakout (>3x)", f"{odds[2]:.0%}", f"under budget: {odds[0]:.0%}", **note)

    actual = result.get("actual_revenue_for_checking")
    if actual:
        inside = low <= actual <= high
        msg = (f"**What actually happened:** {usd(actual)} ({actual / budget:.1f}x its budget), "
               f"{'inside' if inside else 'outside'} the likely range.")
        (st.success if inside else st.warning)(msg)

    key = json.dumps(result, sort_keys=True)
    written = st.session_state.setdefault("written", {})
    if has_claude and key not in written:
        if st.button("✍️  Write the marketing brief with Claude", key=f"claude_{slot}", width="stretch"):
            if take_brief_credit():
                try:
                    written[key] = words_for(key)
                    st.rerun()   # redraw without the button, with Claude's text in the brief
                except Exception as e:
                    st.warning(f"Couldn't get the written read: {e}")
            else:
                st.warning("Today's limit of written briefs is used up. The numbers are still live.")
    words = written.get(key)

    page = brief.render_html(result, words, no_words_note="")
    with st.expander("📄  Full one-page brief", expanded=True):
        st.download_button("Download this brief", page, file_name=f"{result['film']} brief.html",
                           mime="text/html", key=f"download_{slot}")
        st.iframe(page)   # every film detail and Claude's text are HTML-escaped by render_html


st.title("🎬 Indie film forecaster")
st.markdown("Enter a film's details **before release**. You get a box office forecast, an honest range, "
            "and the odds of each outcome. Claude can then write a one-page brief for the marketing team.")

with st.expander("How accurate is it?"):
    st.markdown(
        f"- Trained on **{len(df)} releases** from A24 and similar studios (2012–2025), public data only.\n"
        "- Tested on 101 films it had never seen: typically off by about **2x**. Read the range, not just the number.\n"
        "- The likely range held the real result **44%** of the time (it aims for 50%).\n"
        "- The odds can lean lower than the central forecast: past misses were more often too high, "
        "mostly on small releases.\n"
        "- Use the **Test it on a past film** tab to see a forecast next to what really happened.")

tab_new, tab_past = st.tabs(["Forecast a new film", "Test it on a past film"])

with tab_new:
    with st.form("film"):
        title = st.text_input("Film title", "Behemoth!")
        c1, c2 = st.columns(2)
        budget = c1.number_input("Budget ($ millions)", min_value=0.1, value=36.0, step=1.0, format="%.1f")
        release = c2.date_input("Release date", dt.date(2026, 12, 2))
        genres = c1.multiselect("Genres", all_genres, default=["Music", "Drama"])
        rating = c2.selectbox("US rating", ["R", "PG-13", "PG", "G", "NR"])
        is_sequel = st.checkbox("It's a sequel or part of a franchise")
        with st.expander("More details (optional)"):
            studios = sorted(df.studio.unique())
            d1, d2 = st.columns(2)
            director = d1.text_input("Director", "Tony Gilroy",
                                     help="Used only if they have earlier films in the data.")
            studio = d2.selectbox("Studio", studios, index=studios.index("Searchlight Pictures"))
            runtime = d1.number_input("Runtime (minutes)", min_value=60, max_value=240, value=127)
            competitors = d2.number_input("Films opening the same week", min_value=0, value=0,
                                          help="Leave at 0 if you don't know: the usual number for that month is used.")
        submitted = st.form_submit_button("Get the forecast", type="primary", width="stretch")
    if submitted:
        if not genres:
            st.error("Pick at least one genre.")
        else:
            info = {"title": title.strip() or "Untitled", "studio": studio, "release_date": str(release),
                    "budget": budget * 1e6, "genres": genres, "rating": rating, "runtime": runtime,
                    "director": director.strip(), "is_sequel": is_sequel, "competitors": competitors or None}
            try:
                st.session_state.new_result = fc.forecast_new(df, info, errors)
            except ValueError as e:
                st.error(str(e))
    if st.session_state.get("new_result"):
        show(st.session_state.new_result, "new")

with tab_past:
    st.markdown("Pick a released film. The forecast uses **only films released before it**, "
                "so you can see how it would have done.")
    choices = sorted(df[df.year >= 2017].title)
    past_title = st.selectbox("Film", choices, index=choices.index("Heretic"))
    show(backtest(past_title), "past")
