"""Low Water Crossing Risk - demo app.

Run from the repo folder:  streamlit run app.py
Needs data/processed/crossings.csv and vlm_config.py. Reads generated briefs.csv when present;
interactive model responses are saved separately in interactive_briefs.csv.
"""
from pathlib import Path

import pandas as pd
import streamlit as st
from openai import OpenAI

from vlm_config import API_KEY, BASE_URL, MODEL

DATA = Path(__file__).resolve().parent / "data" / "processed"
BRIEFS = DATA / "briefs.csv"
INTERACTIVE_BRIEFS = DATA / "interactive_briefs.csv"
TIERS = [("Top 10: check first", 10, "#A32D2D"), ("Next 30: check soon", 40, "#EF9F27"),
         ("Lower priority", 999, "#888780")]

st.set_page_config(page_title="Low Water Crossing Risk", layout="wide")


# ---------- data ----------
@st.cache_data
def load_crossings():
    df = pd.read_csv(DATA / "crossings.csv").sort_values("risk_rank").reset_index(drop=True)
    df["tier"] = df["risk_rank"].map(lambda r: next(t for t, m, _ in TIERS if r <= m))
    return df


def load_briefs():
    briefs = {}
    if BRIEFS.exists():
        b = pd.read_csv(BRIEFS, keep_default_na=False)

        # Batch-generated briefs have one row per crossing and fixed language columns.
        if {"CartID", "warning_en", "warning_es", "maintenance_suggestion"}.issubset(b.columns):
            for r in b.itertuples(index=False):
                en = str(r.warning_en).strip()
                es = str(r.warning_es).strip()
                maintenance = str(r.maintenance_suggestion).strip()
                if en:
                    briefs[(r.CartID, "Residents", "English")] = en
                if es:
                    briefs[(r.CartID, "Residents", "Spanish")] = es
                if en or es:
                    both = [en] if en else []
                    if es:
                        both.append(f"Español\n{es}")
                    briefs[(r.CartID, "Residents", "Both")] = "\n\n".join(both)
                if maintenance:
                    briefs[(r.CartID, "Public Works crew", "English")] = maintenance

        # Read the older long format for compatibility with previously saved briefs.
        if {"CartID", "audience", "language", "brief"}.issubset(b.columns):
            for r in b.itertuples(index=False):
                if r.brief:
                    briefs[(r.CartID, r.audience, r.language)] = r.brief

    # Keep app-generated prompts separate so briefs.csv retains its batch schema.
    if INTERACTIVE_BRIEFS.exists():
        saved = pd.read_csv(INTERACTIVE_BRIEFS, keep_default_na=False)
        if {"CartID", "audience", "language", "brief"}.issubset(saved.columns):
            for r in saved.itertuples(index=False):
                if r.brief:
                    briefs[(r.CartID, r.audience, r.language)] = r.brief
    return briefs


USGS_URL = ("https://waterservices.usgs.gov/nwis/iv/?format=json"
            "&countyCd=48029&parameterCd=00065&siteStatus=active")   # Bexar County, gage height (ft)


def parse_usgs(payload):
    rows = []
    for t in payload["value"]["timeSeries"]:
        vals = t["values"][0]["value"]
        if not vals:
            continue
        geo = t["sourceInfo"]["geoLocation"]["geogLocation"]
        height = float(vals[-1]["value"])
        if height <= -999999:
            height = float("nan")
        rows.append({"gauge_name": t["sourceInfo"]["siteName"], "g_lat": float(geo["latitude"]),
                     "g_lon": float(geo["longitude"]), "gage_height_ft": height,
                     "gage_time": vals[-1]["dateTime"][:16].replace("T", " ")})
    return pd.DataFrame(rows)


@st.cache_data(ttl=300, show_spinner="Reading live USGS gauges...")
def fetch_gauges():
    """Latest reading from every active USGS gauge in Bexar County. Cached for 5 minutes."""
    import requests
    try:
        return parse_usgs(requests.get(USGS_URL, timeout=30).json()), None
    except Exception as e:
        return pd.DataFrame(), str(e)


def refresh_gauge_readings(df, gauges):
    """Refresh saved nearest-gauge readings by name; crossings.csv already has the spatial join."""
    if gauges.empty or "nearest_gauge" not in df.columns:
        return df

    live = gauges.copy()
    live["gage_height_ft"] = pd.to_numeric(live["gage_height_ft"], errors="coerce")
    live.loc[live["gage_height_ft"] <= -999999, "gage_height_ft"] = float("nan")
    live = live.dropna(subset=["gage_height_ft"]).drop_duplicates("gauge_name", keep="last")
    if live.empty:
        return df

    latest_height = df["nearest_gauge"].map(live.set_index("gauge_name")["gage_height_ft"])
    latest_time = df["nearest_gauge"].map(live.set_index("gauge_name")["gage_time"])
    refreshed = df.copy()

    if "gauge_height_ft" not in refreshed:
        refreshed["gauge_height_ft"] = float("nan")
    refreshed["gauge_height_ft"] = pd.to_numeric(refreshed["gauge_height_ft"], errors="coerce")
    refreshed.loc[refreshed["gauge_height_ft"] <= -999999, "gauge_height_ft"] = float("nan")
    refreshed["gauge_height_ft"] = latest_height.combine_first(refreshed["gauge_height_ft"])

    if "gauge_reading_time" not in refreshed:
        refreshed["gauge_reading_time"] = ""
    refreshed["gauge_reading_time"] = latest_time.combine_first(refreshed["gauge_reading_time"])
    return refreshed


def save_brief(cart_id, audience, language, text):
    row = pd.DataFrame([{"CartID": cart_id, "audience": audience, "language": language, "brief": text}])
    if INTERACTIVE_BRIEFS.exists():
        old = pd.read_csv(INTERACTIVE_BRIEFS)
        old = old[~((old.CartID == cart_id) & (old.audience == audience) & (old.language == language))]
        row = pd.concat([old, row])
    row.to_csv(INTERACTIVE_BRIEFS, index=False, encoding="utf-8")


def facts(r):
    """Plain facts sent to the model. The model may only use these."""
    lines = [
        f"Crossing: {r.Name} (ID {r.CartID})",
        f"Risk rank: {r.risk_rank} of 156 ({r.tier}); risk score {r.risk_score}",
        f"Active storm inlets within 1,500 ft: {r.inlets_active}",
        f"Inlets rated Poor or Failed nearby: {r.inlets_poor_failed}",
        f"Inlets never inspected nearby: {r.inlets_uninspected}",
        f"Distance to nearest inlet: {int(r.dist_nearest_inlet_ft):,} ft",
        f"City drainage projects planned within half a mile (2026-2030): {r.imp_projects_nearby}",
    ]
    if pd.notna(r.creek):
        lines.append(f"Creek: {r.creek}; sewer overflows into this creek since 2021: {r.sso_same_creek}")
    if "nearest_gauge" in r._fields and pd.notna(r.nearest_gauge):
        lines.append(f"Nearest public USGS stream gauge: {r.nearest_gauge}, {r.gauge_dist_mi} miles away")
        if ("gauge_height_ft" in r._fields and pd.notna(r.gauge_height_ft)
                and r.gauge_height_ft > -999999):
            lines.append(f"Latest reading at that gauge: {r.gauge_height_ft} ft gage height at {r.gauge_reading_time} "
                         "(local gauge measure, not water depth on the road)")
    return "\n".join(lines)


# ---------- model ----------
SYSTEM = ("You write short, plain-language flood safety text for San Antonio. Use ONLY the facts given. "
          "Never invent water levels, dates, closures or numbers. If a fact is missing, do not mention it. "
          "The risk rank shows warning signs in city data, not a flood forecast.")

AUDIENCE = {
    "Residents": "Write a 2-3 sentence warning for drivers and residents near this crossing, with one practical "
                 "safety tip (e.g. turn around, don't drown; check BexarFlood.org before driving in rain).",
    "Public Works crew": "Write a 3-bullet maintenance brief for a city crew: what to inspect first and why, "
                         "based only on the facts.",
}


def ask_model(messages, max_tokens=350):
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=90, max_retries=0)
    resp = client.chat.completions.create(
        model=MODEL, messages=messages, temperature=0.2, max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    return (resp.choices[0].message.content or "").strip()


def write_brief(r, audience, language):
    lang = {"English": "Write in English.", "Spanish": "Write in Spanish.",
            "Both": "Write it in English, then the same text in Spanish under a 'Español' heading."}[language]
    return ask_model([{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": f"{AUDIENCE[audience]} {lang}\n\nFacts:\n{facts(r)}"}])


# ---------- layout ----------
df = load_crossings()
gauges, gauge_error = fetch_gauges()
df = refresh_gauge_readings(df, gauges)
st.session_state.setdefault("briefs", load_briefs())
st.session_state.setdefault("chat", {})

st.title("Low water crossing risk, San Antonio")
st.caption("Ranks 156 low water crossings by warning signs in city drainage data. "
           f"AI text is written locally by {MODEL} on a Dell Pro Max GB10. Review before sharing.")

with st.sidebar:
    st.header("Filter")
    tiers = st.multiselect("Priority", [t for t, _, _ in TIERS], default=[t for t, _, _ in TIERS])
    search = st.text_input("Search by road or creek")
    view = df[df["tier"].isin(tiers)]
    if search:
        view = view[view["Name"].str.contains(search, case=False) | view["creek"].fillna("").str.contains(search, case=False)]
    st.metric("Crossings shown", len(view))
    if "gauge_dist_mi" in df:
        st.metric("With a public gauge within 1 mi", int((df["gauge_dist_mi"] <= 1).sum()))
    st.divider()
    st.subheader("Live USGS gauges")
    if gauges.empty:
        st.warning("Live gauges unavailable; showing saved data." + (f" ({gauge_error[:80]})" if gauge_error else ""))
    else:
        st.success(f"{len(gauges)} gauges · latest {gauges['gage_time'].max()}")
    if st.button("Refresh live data"):
        fetch_gauges.clear()
        st.rerun()
    st.divider()
    st.caption(f"Model server: {BASE_URL}")

if view.empty:
    st.warning("No crossings match these filters.")
    st.stop()

left, right = st.columns([3, 2], gap="large")

# map
with left:
    labels = [f"#{r.risk_rank} {r.Name}" for r in view.itertuples()]
    if "selected" not in st.session_state or st.session_state.selected not in labels:
        st.session_state.selected = labels[0]
    try:
        import folium
        from streamlit_folium import st_folium
        m = folium.Map(location=[df["lat"].mean(), df["lon"].mean()], zoom_start=10)
        for r in view.sort_values("risk_rank", ascending=False).itertuples():
            color = next(c for t, _, c in TIERS if t == r.tier)
            folium.CircleMarker([r.lat, r.lon], radius=9 if r.risk_rank <= 10 else 6 if r.risk_rank <= 40 else 4,
                                color="white", weight=1, fill=True, fill_color=color, fill_opacity=0.9,
                                tooltip=f"#{r.risk_rank} {r.Name}").add_to(m)
        for g in gauges.itertuples():
            reading = f"{g.gage_height_ft:.2f} ft" if pd.notna(g.gage_height_ft) else "height unavailable"
            folium.RegularPolygonMarker([g.g_lat, g.g_lon], number_of_sides=4, radius=6, color="#185FA5",
                                        fill=True, fill_opacity=0.9,
                                        tooltip=f"USGS: {g.gauge_name} · {reading} at {g.gage_time}").add_to(m)
        clicked = st_folium(m, height=480, use_container_width=True, returned_objects=["last_object_clicked_tooltip"])
        tip = (clicked or {}).get("last_object_clicked_tooltip")
        if tip in labels:
            st.session_state.selected = tip
    except ImportError:
        st.map(view.rename(columns={"lon": "longitude", "lat": "latitude"}), size=60)
    st.caption("Red: top 10 · Orange: next 30 · Gray: lower priority · Blue squares: live USGS gauges. Click a dot to select it.")

    st.subheader("Top 10 to check first")
    st.dataframe(df.head(10)[["risk_rank", "Name", "inlets_active", "inlets_poor_failed", "imp_projects_nearby",
                              "sso_same_creek"]].rename(columns={
        "risk_rank": "Rank", "inlets_active": "Active inlets", "inlets_poor_failed": "Poor/failed",
        "imp_projects_nearby": "Drainage projects", "sso_same_creek": "Creek overflows"}),
        hide_index=True, use_container_width=True)

# detail + AI
with right:
    choice = st.selectbox("Crossing", labels, index=labels.index(st.session_state.selected))
    st.session_state.selected = choice
    r = next(x for x in view.itertuples() if f"#{x.risk_rank} {x.Name}" == choice)

    color = next(c for t, _, c in TIERS if t == r.tier)
    st.markdown(f"### {r.Name}")
    st.markdown(f'<span style="background:{color};color:white;padding:2px 10px;border-radius:10px;'
                f'font-size:13px">#{r.risk_rank} · {r.tier}</span>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Active inlets", r.inlets_active)
    c2.metric("Poor/failed", r.inlets_poor_failed)
    c3.metric("Uninspected", r.inlets_uninspected)
    c1.metric("Drainage projects", r.imp_projects_nearby)
    c2.metric("Creek overflows", r.sso_same_creek if pd.notna(r.creek) else "n/a")
    if "gauge_dist_mi" in r._fields and pd.notna(r.gauge_dist_mi):
        c3.metric("Nearest gauge", f"{r.gauge_dist_mi} mi")
        if "gauge_height_ft" in r._fields and pd.notna(r.gauge_height_ft):
            st.caption(f"Live: {r.nearest_gauge} reads {r.gauge_height_ft} ft at {r.gauge_reading_time} "
                       "(USGS gage height, not water depth on the road)")

    st.divider()
    st.markdown("#### AI brief")
    a1, a2 = st.columns(2)
    audience = a1.radio("For", list(AUDIENCE), horizontal=True)
    language = a2.radio("Language", ["English", "Spanish", "Both"], horizontal=True)
    key = (r.CartID, audience, language)
    if st.button("Write brief on the GB10", type="primary", use_container_width=True):
        with st.spinner("Asking the local model..."):
            try:
                text = write_brief(r, audience, language)
                st.session_state.briefs[key] = text
                save_brief(*key, text)
            except Exception as e:
                st.error(f"Could not reach the model at {BASE_URL}: {e}")
    if key in st.session_state.briefs:
        st.info(st.session_state.briefs[key])
    with st.expander("Facts sent to the model"):
        st.code(facts(r), language=None)

    st.divider()
    st.markdown("#### Ask about this crossing")
    history = st.session_state.chat.setdefault(r.CartID, [])
    for msg in history:
        st.chat_message(msg["role"]).write(msg["content"])
    q = st.chat_input("e.g. Why is this crossing ranked high?")
    if q:
        history.append({"role": "user", "content": q})
        msgs = [{"role": "system", "content": SYSTEM + " Answer in 2-4 sentences.\n\nFacts:\n" + facts(r)}] + history
        try:
            history.append({"role": "assistant", "content": ask_model(msgs, 250)})
        except Exception as e:
            history.pop()
            st.error(f"Could not reach the model: {e}")
        st.rerun()

with st.expander("Batch: write resident briefs for the top 10"):
    if st.button("Run for top 10 (English and Spanish)"):
        bar = st.progress(0.0)
        for i, row in enumerate(df.head(10).itertuples()):
            try:
                text = write_brief(row, "Residents", "Both")
                st.session_state.briefs[(row.CartID, "Residents", "Both")] = text
                save_brief(row.CartID, "Residents", "Both", text)
            except Exception as e:
                st.error(f"{row.Name}: {e}")
                break
            bar.progress((i + 1) / 10)
        st.success("Saved to data/processed/interactive_briefs.csv")
