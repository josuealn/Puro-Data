from pathlib import Path
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="San Antonio Low Water Crossing Risk", layout="wide")
st.title("San Antonio Low Water Crossing Risk & AI Safety Briefs")

df = pd.read_csv(Path("data/processed/briefs.csv")).sort_values("risk_rank", ascending=True)

cluster_order = (
    df.groupby("cluster")["risk_score"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)
tier_meta = [
    ("Tier 1 · Highest Infrastructure Risk", "#dc2626"),
    ("Tier 2 · Elevated Drainage / Overflow Risk", "#ea580c"),
    ("Tier 3 · Moderate Drainage Stress", "#2563eb"),
    ("Tier 4 · Lower Risk / Better Drainage Coverage", "#16a34a")
]
cluster_lookup = {
    cid: tier_meta[idx] if idx < len(tier_meta) else (f"Cluster {cid}", "#7c3aed")
    for idx, cid in enumerate(cluster_order)
}

lo, hi = df["risk_score"].min(), df["risk_score"].max()

col_map, col_side = st.columns([3, 2])

with col_map:
    m = folium.Map(location=[df["lat"].mean(), df["lon"].mean()], zoom_start=10, tiles="OpenStreetMap")
    for _, row in df.iterrows():
        label, color = cluster_lookup[row["cluster"]]
        s = (row["risk_score"] - lo) / (hi - lo) if hi > lo else 0.5
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=5 + 8 * s,
            color=color,
            fill=True,
            fill_opacity=0.82,
            weight=1,
            tooltip=f"#{int(row['risk_rank'])} | {row['Name']} (Score: {row['risk_score']:.2f})"
        ).add_to(m)
    map_data = st_folium(m, width=780, height=540)

with col_side:
    st.subheader("Top 10 Highest-Risk Crossings")
    top10 = df[[
        "risk_rank", "Name", "risk_score", "inlets_active",
        "inlets_poor_failed", "dist_nearest_inlet_ft", "sso_same_creek"
    ]].head(10)
    st.dataframe(top10, use_container_width=True, hide_index=True)

selected = df.iloc[0]
if map_data and map_data.get("last_object_clicked_tooltip"):
    clicked_part = map_data["last_object_clicked_tooltip"].split(" | ", 1)[-1]
    clicked_name = clicked_part.split(" (Score:")[0]
    match = df[df["Name"] == clicked_name]
    if not match.empty:
        selected = match.iloc[0]

cluster_label, _ = cluster_lookup[selected["cluster"]]
has_creek = pd.notna(selected["creek"]) and str(selected["creek"]).strip() != ""
creek_display = selected["creek"] if has_creek else "No named creek (SSO count unknown)"
sso_display = f"{int(selected['sso_rain_related'])} rain / {int(selected['sso_same_creek'])} total" if has_creek else "Unknown (no creek)"
imp_fy = f"FY {int(selected['imp_next_fiscal_year'])}" if pd.notna(selected["imp_next_fiscal_year"]) else "None within 0.5 mi"

st.divider()
st.subheader(f"Rank #{int(selected['risk_rank'])}: {selected['Name']} ({selected['CartID']})")
st.caption(f"Creek: {creek_display}  |  {cluster_label} (Cluster {int(selected['cluster'])})")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Risk Score (Z-Index)", f"{selected['risk_score']:.2f}")
m2.metric("Active / Poor / TBD Inlets", f"{int(selected['inlets_active'])} / {int(selected['inlets_poor_failed'])} / {int(selected['inlets_uninspected'])}")
m3.metric("Nearest Inlet (ft)", f"{selected['dist_nearest_inlet_ft']:,.0f}")
m4.metric("Same-Creek SSOs", sso_display)
m5.metric("Planned IMP (0.5 mi)", f"{int(selected['imp_projects_nearby'])} ({imp_fy})")

c1, c2, c3 = st.columns(3)
c1.info(f"**Driver Warning (EN)**\n\n{selected['warning_en']}")
c2.warning(f"**Alerta para Conductores (ES)**\n\n{selected['warning_es']}")
c3.success(f"**Public Works Action**\n\n{selected['maintenance_suggestion']}")