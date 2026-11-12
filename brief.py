import json, html, webbrowser, os, math
import anthropic

# ---------- 1. Load the model's output ----------
with open("forecast.json") as f:
    forecast = json.load(f)
actual = forecast.pop("actual_revenue_for_checking", None)   # hidden from the agent

# ---------- 2. Ask Claude for the WORDS only (numbers are drawn by Python) ----------
SYSTEM = """You are an analyst on A24's data team. You turn model outputs into short,
clear briefs for the marketing team, who are not technical.

Rules:
- Use ONLY the information in the data you are given. Never invent figures, films or facts.
- No jargon: no "log", "ridge", "coefficient", "feature" or "model". Plain English.
- Box office is not profit. Never say a film "makes" or "loses money" or "covers its budget".
  Talk in multiples of budget instead (e.g. "about 3x its budget").
- The range comes from how far off forecasts were on past films. Never call outcomes "simulated".
- Be honest about uncertainty, but be useful: marketing needs a clear read.

Reply with ONLY a JSON object, no other text, in exactly this shape:
{
  "headline": "one short sentence verdict (max 12 words)",
  "bottom_line": "2 sentences",
  "helping": ["short point", "..."],
  "holding_back": ["short point", "..."],
  "comparables_read": "1-2 sentences on what the comparable films tell us",
  "risks": ["short point", "...", "..."],
  "actions": ["concrete suggestion for marketing", "...", "..."]
}"""

client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from the terminal
response = client.messages.create(
    model="claude-sonnet-5-5",
    max_tokens=2000,
    system=SYSTEM,
    messages=[{"role": "user", "content": "Pre-release forecast data:\n\n" + json.dumps(forecast, indent=2)}],
)
text = "".join(b.text for b in response.content if b.type == "text").strip()
text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
words = json.loads(text)

# ---------- 3. Build the one-page report ----------
e = html.escape
def money(x): return f"${x / 1e6:.0f}M" if x >= 1e6 else f"${x / 1e3:.0f}K"
budget = forecast["budget"]
mid = forecast["forecast_most_likely"]
low, high = forecast["forecast_range_50pct"]
tiers = forecast["tier_probabilities"]
comps = forecast["comparables"]

# Range chart on a log scale (box office spans small to huge)
values = [low, mid, high, budget] + [c["revenue"] for c in comps] + ([actual] if actual else [])
lo_axis, hi_axis = math.log(min(values) * 0.7), math.log(max(values) * 1.4)
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
    for i, (name, p) in enumerate(tiers.items()))

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
             f'{"inside" if inside else "outside"} the forecast range.</div>')

page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{e(forecast["film"])} – Pre-release brief</title>
<style>
:root{{--ink:#f2f0ea;--muted:#8a877f;--line:#2a2a2a;--bg:#0b0b0b;--card:#121212;--accent:#f2f0ea;
--good:#7fbf9a;--mid:#d8c37a;--bad:#d97a6a;--band:#2b2a27}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 "Helvetica Neue",Helvetica,Arial,sans-serif;-webkit-font-smoothing:antialiased}}
.wrap{{max-width:880px;margin:0 auto;padding:56px 24px 72px}}
.eyebrow{{font-size:11px;letter-spacing:.28em;text-transform:uppercase;color:var(--muted)}}
h1{{font-size:72px;line-height:.95;margin:14px 0 14px;letter-spacing:-.04em;font-weight:700;text-transform:uppercase}}
.meta{{color:var(--muted);font-size:13px;letter-spacing:.04em;padding-bottom:22px;border-bottom:1px solid var(--line)}}
.headline{{font-size:24px;margin:30px 0 8px;font-weight:600;letter-spacing:-.01em}}
.lede{{color:#c9c6be;margin:0 0 30px;max-width:680px}}
.kpis{{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:var(--line);border:1px solid var(--line);margin-bottom:22px}}
.kpi{{background:var(--bg);padding:20px}}
.kpi b{{display:block;font-size:34px;letter-spacing:-.03em;font-weight:700}}.kpi span{{color:var(--muted);font-size:11px;letter-spacing:.18em;text-transform:uppercase}}
.card{{background:var(--card);border:1px solid var(--line);padding:22px;margin-bottom:16px}}
h2{{font-size:11px;letter-spacing:.28em;text-transform:uppercase;color:var(--muted);margin:0 0 16px;font-weight:600}}
.range{{position:relative;height:74px;margin:30px 6px 6px}}
.axis{{position:absolute;top:34px;left:0;right:0;height:1px;background:#3a3a3a}}
.band{{position:absolute;top:24px;height:22px;background:var(--band)}}
.midmark{{position:absolute;top:18px;width:3px;height:34px;background:var(--accent);transform:translateX(-50%)}}
.midmark span{{position:absolute;top:-22px;left:50%;transform:translateX(-50%);white-space:nowrap;font-weight:700;font-size:13px}}
.dot{{position:absolute;top:30px;width:9px;height:9px;border-radius:50%;background:#77746c;transform:translateX(-50%);border:2px solid var(--card)}}
.mark{{position:absolute;top:14px;height:42px;border-left:1px dashed;transform:translateX(-1px)}}
.mark span{{position:absolute;top:44px;left:0;transform:translateX(-50%);white-space:nowrap;font-size:12px}}
.mark.budget{{border-color:#77746c;color:var(--muted)}}.mark.actual{{border-color:var(--good);color:var(--good);font-weight:600}}
.legend{{font-size:11px;color:var(--muted);margin-top:10px;letter-spacing:.06em}}
.legend i{{display:inline-block;width:8px;height:8px;border-radius:50%;background:#77746c;margin:0 4px 0 0}}
.legend em{{display:inline-block;width:18px;height:8px;background:var(--band);margin:0 4px 0 12px;vertical-align:-1px}}
.tier{{display:grid;grid-template-columns:150px 1fr 48px;gap:14px;align-items:center;margin:10px 0}}
.tname small{{display:block;color:var(--muted);font-size:12px}}
.tbar{{height:6px;background:#242424;overflow:hidden}}.tbar div{{height:100%}}
.tpct{{text-align:right;font-weight:700}}
.cols{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
ul{{margin:0;padding-left:18px}}li{{margin:6px 0;color:#d6d3cb}}
.up h2{{color:var(--good)}}.down h2{{color:var(--bad)}}
table{{width:100%;border-collapse:collapse;font-size:14px}}
th{{text-align:left;font-size:10px;letter-spacing:.2em;text-transform:uppercase;color:var(--muted);font-weight:600;border-bottom:1px solid var(--line);padding:8px 4px}}
td{{padding:10px 4px;border-bottom:1px solid #1d1d1d}}.num{{text-align:right;font-variant-numeric:tabular-nums}}
.read{{margin:14px 0 0;color:#c9c6be}}
ol{{margin:0;padding-left:20px}}ol li{{margin:8px 0}}
.check{{padding:14px 16px;font-size:14px;margin-bottom:16px;border:1px solid}}
.check.ok{{border-color:#2f5a44;color:var(--good);background:#0f1a14}}.check.miss{{border-color:#5a2f28;color:var(--bad);background:#1a110f}}
footer{{color:#66635c;font-size:11px;margin-top:34px;line-height:1.6;border-top:1px solid var(--line);padding-top:16px}}
@media(max-width:640px){{.kpis,.cols{{grid-template-columns:1fr}}h1{{font-size:44px}}.tier{{grid-template-columns:110px 1fr 40px}}}}
@media print{{*{{-webkit-print-color-adjust:exact;print-color-adjust:exact}}.wrap{{padding:24px}}}}
</style></head><body><div class="wrap">

<div class="eyebrow">Pre-release brief · Data team</div>
<h1>{e(forecast["film"])}</h1>
<div class="meta">{e(forecast["studio"])} · Release {e(forecast["release_date"])} · {e(forecast["genres"].replace("|", ", "))} · Rated {e(forecast["rating"])} · {money(budget)} budget</div>

<div class="headline">{e(words["headline"])}</div>
<p class="lede">{e(words["bottom_line"])}</p>

<div class="kpis">
  <div class="kpi"><b>{money(mid)}</b><span>Most likely box office</span></div>
  <div class="kpi"><b>{mid / budget:.1f}x</b><span>Budget multiple</span></div>
  <div class="kpi"><b>{money(low)}–{money(high)}</b><span>Likely range (middle 50%)</span></div>
</div>

{check}

<div class="card"><h2>Where it could land</h2>
  <div class="range"><div class="axis"></div>
    <div class="band" style="left:{pos(low):.1f}%;width:{pos(high) - pos(low):.1f}%"></div>
    {marks}
    <div class="midmark" style="left:{pos(mid):.1f}%"><span>{money(mid)}</span></div>
  </div>
  <div class="legend"><i></i>comparable films <em></em>likely range · log scale</div>
</div>

<div class="card"><h2>Chance of each outcome</h2>{tier_rows}</div>

<div class="cols">
  <div class="card up"><h2>Helping</h2><ul>{items(words["helping"])}</ul></div>
  <div class="card down"><h2>Holding it back</h2><ul>{items(words["holding_back"])}</ul></div>
</div>

<div class="card"><h2>Comparable films</h2>
  <table><tr><th>Film</th><th>Studio</th><th>Year</th><th class="num">Budget</th><th class="num">Box office</th><th class="num">Multiple</th></tr>{comp_rows}</table>
  <p class="read">{e(words["comparables_read"])}</p>
</div>

<div class="cols">
  <div class="card"><h2>Risks</h2><ul>{items(words["risks"])}</ul></div>
  <div class="card"><h2>Suggested actions</h2><ol>{items(words["actions"])}</ol></div>
</div>

<footer>Forecast from a model trained on public data for about 200 indie releases (2012–2025); typical error is about 2x, so treat figures as odds, not promises.
Box office is worldwide gross, not profit. Data: TMDB. This product uses the TMDB API but is not endorsed or certified by TMDB.</footer>
</div></body></html>"""

with open("brief.html", "w") as f:
    f.write(page)
print("Saved brief.html, opening it in your browser...")
webbrowser.open("file://" + os.path.abspath("brief.html"))