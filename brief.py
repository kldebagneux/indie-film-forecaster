"""Turn a forecast into a one-page brief for the marketing team.

    python3 brief.py output/behemoth.forecast.json
    python3 brief.py output/heretic.forecast.json --no-open

Claude writes only the words, returned as structured data that is checked
against a fixed shape. Every number and chart on the page comes from Python,
so the text can't get a figure wrong. Claude's words are saved next to the
forecast (<film>.words.json), so the page can be rebuilt without calling the API again.
"""
import argparse, html, json, math, os, sys, webbrowser
from pathlib import Path

from pydantic import BaseModel

import forecaster as fc

MODEL = "claude-sonnet-5-5"

SYSTEM = """You are an analyst on A24's data team. You turn model outputs into short,
clear briefs for the marketing team, who are not technical.

Rules:
- Use ONLY the information in the data you are given. Never invent figures, films or facts.
- No jargon: no "log", "ridge", "coefficient", "feature" or "model". Plain English.
- Box office is not profit. Never say a film "makes" or "loses money" or "covers its budget".
  Talk in multiples of budget instead (e.g. "about 3x its budget").
- The ranges come from how far off forecasts were on past films the system had not seen.
  Never call outcomes "simulated".
- "main_drivers" compares this film with a typical film in the data: "multiplier" is how much
  that factor moves the forecast (0.8 = about 20% lower). Use them for "helping" and "holding_back".
- Be honest about uncertainty, but be useful: marketing needs a clear read.
- Keep every list item to one short sentence. 2-4 items per list."""


class Brief(BaseModel):
    headline: str          # one short sentence verdict, max 12 words
    bottom_line: str       # 2 sentences
    helping: list[str]
    holding_back: list[str]
    comparables_read: str  # 1-2 sentences on what the comparable films tell us
    risks: list[str]
    actions: list[str]     # concrete suggestions for marketing


def write_words(forecast):
    """Ask Claude for the text of the brief. Returns a dict shaped like Brief."""
    import anthropic
    client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from the environment
    response = client.beta.messages.parse(
        model=MODEL,
        max_tokens=8000,
        system=SYSTEM,
        messages=[{"role": "user", "content": "Pre-release forecast data:\n\n" + json.dumps(forecast, indent=2)}],
        output_format=Brief,
        # If a safety check ever declines the request, retry it on a fallback model automatically
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason == "refusal" or response.parsed_output is None:
        raise RuntimeError(f"Claude didn't return a brief (stop reason: {response.stop_reason}).")
    return response.parsed_output.model_dump()


CSS = """
:root{--ink:#f2f0ea;--muted:#8a877f;--line:#2a2a2a;--bg:#0b0b0b;--card:#121212;--accent:#f2f0ea;
--good:#7fbf9a;--mid:#d8c37a;--bad:#d97a6a;--band:#3a3833;--band80:#211f1c}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 "Helvetica Neue",Helvetica,Arial,sans-serif;-webkit-font-smoothing:antialiased}
.wrap{max-width:880px;margin:0 auto;padding:56px 24px 72px}
.eyebrow{font-size:11px;letter-spacing:.28em;text-transform:uppercase;color:var(--muted)}
h1{font-size:72px;line-height:.95;margin:14px 0 14px;letter-spacing:-.04em;font-weight:700;text-transform:uppercase;overflow-wrap:anywhere}
.meta{color:var(--muted);font-size:13px;letter-spacing:.04em;padding-bottom:22px;border-bottom:1px solid var(--line)}
.headline{font-size:24px;margin:30px 0 8px;font-weight:600;letter-spacing:-.01em}
.lede{color:#c9c6be;margin:0 0 30px;max-width:680px}
.kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:var(--line);border:1px solid var(--line);margin-bottom:22px}
.kpi{background:var(--bg);padding:20px}
.kpi b{display:block;font-size:34px;letter-spacing:-.03em;font-weight:700}.kpi span{color:var(--muted);font-size:11px;letter-spacing:.18em;text-transform:uppercase}
.card{background:var(--card);border:1px solid var(--line);padding:22px;margin-bottom:16px}
h2{font-size:11px;letter-spacing:.28em;text-transform:uppercase;color:var(--muted);margin:0 0 16px;font-weight:600}
.range{position:relative;height:74px;margin:30px 6px 6px}
.axis{position:absolute;top:34px;left:0;right:0;height:1px;background:#3a3a3a}
.band80{position:absolute;top:28px;height:14px;background:var(--band80)}
.band{position:absolute;top:24px;height:22px;background:var(--band)}
.midmark{position:absolute;top:18px;width:3px;height:34px;background:var(--accent);transform:translateX(-50%)}
.midmark span{position:absolute;top:-22px;left:50%;transform:translateX(-50%);white-space:nowrap;font-weight:700;font-size:13px}
.dot{position:absolute;top:30px;width:9px;height:9px;border-radius:50%;background:#77746c;transform:translateX(-50%);border:2px solid var(--card)}
.mark{position:absolute;top:14px;height:42px;border-left:1px dashed;transform:translateX(-1px)}
.mark span{position:absolute;top:44px;left:0;transform:translateX(-50%);white-space:nowrap;font-size:12px}
.mark.budget{border-color:#77746c;color:var(--muted)}.mark.actual{border-color:var(--good);color:var(--good);font-weight:600}
.legend{font-size:11px;color:var(--muted);margin-top:10px;letter-spacing:.06em}
.legend i{display:inline-block;width:8px;height:8px;border-radius:50%;background:#77746c;margin:0 4px 0 0}
.legend em{display:inline-block;width:18px;height:8px;background:var(--band);margin:0 4px 0 12px;vertical-align:-1px}
.legend em.e80{background:var(--band80);border:1px solid #2e2c28}
.tier{display:grid;grid-template-columns:150px 1fr 48px;gap:14px;align-items:center;margin:10px 0}
.tname small{display:block;color:var(--muted);font-size:12px}
.tbar{height:6px;background:#242424;overflow:hidden}.tbar div{height:100%}
.tpct{text-align:right;font-weight:700}
.drv{display:grid;grid-template-columns:1fr 70px;gap:12px;padding:8px 0;border-bottom:1px solid #1d1d1d;font-size:14px}
.drv:last-child{border-bottom:0}.drv b{text-align:right;font-variant-numeric:tabular-nums}
.drv .up{color:var(--good)}.drv .down{color:var(--bad)}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:16px}
ul{margin:0;padding-left:18px}li{margin:6px 0;color:#d6d3cb}
.up h2{color:var(--good)}.down h2{color:var(--bad)}
.table{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:14px}
th{text-align:left;font-size:10px;letter-spacing:.2em;text-transform:uppercase;color:var(--muted);font-weight:600;border-bottom:1px solid var(--line);padding:8px 4px}
td{padding:10px 4px;border-bottom:1px solid #1d1d1d}.num{text-align:right;font-variant-numeric:tabular-nums}
.read{margin:14px 0 0;color:#c9c6be}
ol{margin:0;padding-left:20px}ol li{margin:8px 0}
.check{padding:14px 16px;font-size:14px;margin-bottom:16px;border:1px solid}
.check.ok{border-color:#2f5a44;color:var(--good);background:#0f1a14}.check.miss{border-color:#5a2f28;color:var(--bad);background:#1a110f}
footer{color:#66635c;font-size:11px;margin-top:34px;line-height:1.6;border-top:1px solid var(--line);padding-top:16px}
@media(max-width:640px){.kpis,.cols{grid-template-columns:1fr}h1{font-size:44px}.tier{grid-template-columns:110px 1fr 40px}}
@media print{*{-webkit-print-color-adjust:exact;print-color-adjust:exact}.wrap{padding:24px}}
"""


def render_html(forecast, words=None):
    """Build the one-page brief. Without words (no API key), the text sections are left out."""
    e, money = html.escape, fc.money
    actual = forecast.get("actual_revenue_for_checking")
    budget, mid = forecast["budget"], forecast["forecast_most_likely"]
    low, high = forecast["forecast_range_50pct"]
    low80, high80 = forecast["forecast_range_80pct"]
    comps = forecast["comparables"]

    # Range chart on a log scale (box office spans small to huge)
    values = [low80, high80, mid, budget] + [c["revenue"] for c in comps] + ([actual] if actual else [])
    lo_axis, hi_axis = math.log(min(values) * 0.8), math.log(max(values) * 1.25)
    def pos(x): return 100 * (math.log(x) - lo_axis) / (hi_axis - lo_axis)

    marks = f'<div class="mark budget" style="left:{pos(budget):.1f}%"><span>Budget {money(budget)}</span></div>'
    for c in comps:
        marks += f'<div class="dot" style="left:{pos(c["revenue"]):.1f}%" title="{e(c["title"])}: {money(c["revenue"])}"></div>'
    if actual:
        marks += f'<div class="mark actual" style="left:{pos(actual):.1f}%"><span>Actual {money(actual)}</span></div>'

    tier_colors = ["var(--bad)", "var(--mid)", "var(--good)"]
    tier_rows = "".join(
        f'<div class="tier"><span class="tname">{e(name.split(" (")[0].capitalize())}'
        f'<small>{e("(" + name.split(" (")[1]) if " (" in name else ""}</small></span>'
        f'<div class="tbar"><div style="width:{p * 100:.0f}%;background:{tier_colors[i]}"></div></div>'
        f'<span class="tpct">{p * 100:.0f}%</span></div>'
        for i, (name, p) in enumerate(forecast["tier_probabilities"].items()))

    driver_rows = "".join(
        f'<div class="drv"><span>{e(d["factor"])}</span>'
        f'<b class="{"up" if d["effect"] == "raises" else "down"}">×{d["multiplier"]:.2f}</b></div>'
        for d in forecast["main_drivers"])

    comp_rows = "".join(
        f'<tr><td>{e(c["title"])}</td><td>{e(c["studio"])}</td><td>{c["year"]}</td>'
        f'<td class="num">{money(c["budget"])}</td><td class="num">{money(c["revenue"])}</td>'
        f'<td class="num">{c["revenue"] / c["budget"]:.1f}x</td></tr>' for c in comps)

    def items(lst): return "".join(f"<li>{e(x)}</li>" for x in lst)

    check = ""
    if actual:
        inside = low <= actual <= high
        check = (f'<div class="check {"ok" if inside else "miss"}"><b>Backtest:</b> this forecast used only films released '
                 f'before {e(forecast["film"])}. It actually earned {money(actual)} ({actual / budget:.1f}x budget), '
                 f'{"inside" if inside else "outside"} the 50% range.</div>')

    if words:
        intro = f'<div class="headline">{e(words["headline"])}</div><p class="lede">{e(words["bottom_line"])}</p>'
        helping = f"""<div class="cols">
  <div class="card up"><h2>Helping</h2><ul>{items(words["helping"])}</ul></div>
  <div class="card down"><h2>Holding it back</h2><ul>{items(words["holding_back"])}</ul></div>
</div>"""
        comp_read = f'<p class="read">{e(words["comparables_read"])}</p>'
        closing = f"""<div class="cols">
  <div class="card"><h2>Risks</h2><ul>{items(words["risks"])}</ul></div>
  <div class="card"><h2>Suggested actions</h2><ol>{items(words["actions"])}</ol></div>
</div>"""
    else:
        intro = '<p class="lede" style="margin-top:30px">Numbers only: add an Anthropic API key to get the written read.</p>'
        helping = comp_read = closing = ""

    metrics_path = fc.ROOT / "results" / "metrics.json"
    accuracy = ""
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        accuracy = (f'Tested on {m["test_films"]} past releases it had not seen: typically off by about '
                    f'{m["typical_error_times_off"]["linear_model_ridge"]:.1f}x, and the 50% range held the real result '
                    f'{m["range_coverage"]["50pct_range_contained_actual"]:.0%} of the time. ')

    return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{e(forecast["film"])} – Pre-release brief</title>
<style>{CSS}</style></head><body><div class="wrap">

<div class="eyebrow">Pre-release brief · Data team</div>
<h1>{e(forecast["film"])}</h1>
<div class="meta">{e(forecast["studio"])} · Release {e(forecast["release_date"])} · {e(forecast["genres"].replace("|", ", "))} · Rated {e(str(forecast["rating"]))} · {money(budget)} budget</div>

{intro}

<div class="kpis">
  <div class="kpi"><b>{money(mid)}</b><span>Most likely box office</span></div>
  <div class="kpi"><b>{mid / budget:.1f}x</b><span>Budget multiple</span></div>
  <div class="kpi"><b>{money(low)}–{money(high)}</b><span>Likely range (middle 50%)</span></div>
</div>

{check}

<div class="card"><h2>Where it could land</h2>
  <div class="range"><div class="axis"></div>
    <div class="band80" style="left:{pos(low80):.1f}%;width:{pos(high80) - pos(low80):.1f}%"></div>
    <div class="band" style="left:{pos(low):.1f}%;width:{pos(high) - pos(low):.1f}%"></div>
    {marks}
    <div class="midmark" style="left:{pos(mid):.1f}%"><span>{money(mid)}</span></div>
  </div>
  <div class="legend"><i></i>comparable films <em></em>50% range <em class="e80"></em>80% range ({money(low80)}–{money(high80)}) · log scale</div>
</div>

<div class="card"><h2>Chance of each outcome</h2>{tier_rows}</div>

<div class="card"><h2>What moves the forecast (vs a typical film)</h2>{driver_rows}</div>

{helping}

<div class="card"><h2>Comparable films</h2>
  <div class="table"><table><tr><th>Film</th><th>Studio</th><th>Year</th><th class="num">Budget</th><th class="num">Box office</th><th class="num">Multiple</th></tr>{comp_rows}</table></div>
  {comp_read}
</div>

{closing}

<footer>Forecast from a model trained on public data for {forecast["trained_on_films"]} indie releases. {accuracy}Treat figures as odds, not promises.
Box office is worldwide gross, not profit. Data: TMDB. This product uses the TMDB API but is not endorsed or certified by TMDB.</footer>
</div></body></html>"""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("forecast_file", help="a forecast made by predict.py, e.g. output/behemoth.forecast.json")
    p.add_argument("--rewrite", action="store_true", help="ask Claude again even if saved words exist")
    p.add_argument("--no-open", action="store_true", help="don't open the page in a browser")
    a = p.parse_args()

    path = Path(a.forecast_file)
    forecast = json.loads(path.read_text())
    words_path = path.with_name(path.name.replace(".forecast.json", ".words.json"))
    out_path = path.with_name(path.name.replace(".forecast.json", ".html"))

    if words_path.exists() and not a.rewrite:
        words = json.loads(words_path.read_text())
        print(f"Using saved words from {words_path.name} (pass --rewrite to ask Claude again)")
    else:
        # The real result is hidden from Claude so a backtest brief is written blind
        for_claude = {k: v for k, v in forecast.items() if k != "actual_revenue_for_checking"}
        try:
            words = write_words(for_claude)
        except Exception as err:   # no key, network, refusal: still build the numbers-only page
            print(f"Couldn't get the written read from Claude ({err}). Building a numbers-only page.", file=sys.stderr)
            words = None
        if words:
            words_path.write_text(json.dumps(words, indent=2))

    out_path.write_text(render_html(forecast, words))
    print(f"Saved {out_path}")
    if not a.no_open:
        webbrowser.open("file://" + os.path.abspath(out_path))


if __name__ == "__main__":
    main()
