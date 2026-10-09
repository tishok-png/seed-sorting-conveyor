"""
app.py - Operator Monitoring & Control Dashboard
Automated Seed Sorting System

Professional dashboard layout:
  - Top bar with brand, connection mode and live connection status
  - KPI cards (status, diverter, good, bad, total) with line icons
  - Dedicated camera monitor panel (LIVE badge, framed viewport, timestamp)
  - Control panel (start / stop / belt speed / calibrate / servo test)
  - Analytics cards (trend line, good-vs-bad bar, good rate)
  - History tab with per-session graphs and CSV downloads

Run with:
    streamlit run app.py
"""

import base64
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

from serial_bridge import SerialBridge, SimulationBridge
from cloud_bridge import CloudBridge

# --------------------------------------------------------------------------
# Saved settings - edit here once, never type it again
# --------------------------------------------------------------------------
DEFAULT_FIREBASE_URL = "https://maize-sorting-default-rtdb.firebaseio.com/seed_sorting"

# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Seed Sorting Control Center",
    page_icon="🌽",
    layout="wide",
)

MODE_OPTIONS = [
    "Cloud (WiFi - check from anywhere)",
    "Live (Raspberry Pi over USB)",
    "Simulation (no hardware)",
]

if "bridge" not in st.session_state:
    st.session_state.bridge = None
if "mode" not in st.session_state:
    st.session_state.mode = MODE_OPTIONS[0]
if "live_series" not in st.session_state:
    st.session_state.live_series = []
if "sessions" not in st.session_state:
    st.session_state.sessions = []
if "active_session" not in st.session_state:
    st.session_state.active_session = None
if "last_run_series" not in st.session_state:
    st.session_state.last_run_series = []

# ---- palette ----
NAVY = "#0F172A"
NAVY_2 = "#1E293B"
INDIGO = "#4F46E5"
INDIGO_LIGHT = "#EEF2FF"
GREEN = "#10B981"
GREEN_LIGHT = "#D1FAE5"
RED = "#EF4444"
RED_LIGHT = "#FEE2E2"
BLUE = "#3B82F6"
BLUE_LIGHT = "#DBEAFE"
AMBER = "#F59E0B"
AMBER_LIGHT = "#FEF3C7"
GRAY_LIGHT = "#F1F5F9"
TEXT_DARK = "#0F172A"
TEXT_MUTED = "#64748B"
BORDER = "#E2E8F0"

SPEED_LEVELS = [(1, 3.75), (2, 6.2), (3, 7.4)]

GOOD_COLOR, BAD_COLOR, TOTAL_COLOR = GREEN, RED, INDIGO

# --------------------------------------------------------------------------
# Line icons (Lucide-style SVG paths)
# --------------------------------------------------------------------------
ICONS = {
    "power": '<path d="M18.36 6.64a9 9 0 1 1-12.73 0"/><line x1="12" y1="2" x2="12" y2="12"/>',
    "shuffle": '<polyline points="16 3 21 3 21 8"/><line x1="4" y1="20" x2="21" y2="3"/><polyline points="21 16 21 21 16 21"/><line x1="15" y1="15" x2="21" y2="21"/><line x1="4" y1="4" x2="9" y2="9"/>',
    "check": '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>',
    "x": '<circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/>',
    "layers": '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>',
    "percent": '<line x1="19" y1="5" x2="5" y2="19"/><circle cx="6.5" cy="6.5" r="2.5"/><circle cx="17.5" cy="17.5" r="2.5"/>',
    "camera": '<path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/>',
    "leaf": '<path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"/><path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/>',
    "sliders": '<line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/>',
    "trend": '<polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/>',
    "wifi": '<path d="M5 12.55a11 11 0 0 1 14.08 0"/><path d="M1.42 9a16 16 0 0 1 21.16 0"/><path d="M8.53 16.11a6 6 0 0 1 6.95 0"/><line x1="12" y1="20" x2="12.01" y2="20"/>',
    "history": '<path d="M3 3v5h5"/><path d="M3.05 13A9 9 0 1 0 6 5.3L3 8"/><polyline points="12 7 12 12 15 14"/>',
}


def svg(name: str, size: int = 20, color: str = "currentColor") -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round">{ICONS[name]}</svg>'
    )


# status -> (icon, background, accent)
STATUS_STYLE = {
    "Idle": ("power", GRAY_LIGHT, TEXT_MUTED),
    "Running": ("power", GREEN_LIGHT, GREEN),
    "Calibration": ("sliders", BLUE_LIGHT, BLUE),
    "Fault": ("power", RED_LIGHT, RED),
}

# --------------------------------------------------------------------------
# Theme / styling
# --------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: %GRAYBG%; }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

.block-container {
    max-width: 1500px;
    padding: 1.4rem 2rem 3rem 2rem !important;
}

h1, h2, h3 { color: %TEXT_DARK% !important; font-weight: 800 !important; letter-spacing: -0.02em; }
p, label { color: %TEXT_DARK%; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] { background: %NAVY%; border-right: none; }
[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #FFFFFF !important; font-size: 1.05rem !important;
}
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: %NAVY_2% !important; color: #F8FAFC !important;
    border: 1px solid #334155 !important; border-radius: 10px !important;
}
[data-testid="stSidebar"] div.stButton > button {
    background: %NAVY_2%; color: #F8FAFC !important; border: 1px solid #334155 !important;
}
[data-testid="stSidebar"] div.stButton > button:hover { background: %INDIGO%; border-color: %INDIGO% !important; }
[data-testid="stSidebar"] div.stButton > button[kind="primary"] {
    background: %INDIGO% !important; border-color: %INDIGO% !important;
}
[data-testid="stSidebar"] hr { border-color: #1E293B !important; }
[data-testid="stSidebar"] [data-testid="stExpander"] { border: 1px solid #334155; border-radius: 10px; }

/* ---------- Top bar ---------- */
.topbar {
    display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-wrap: wrap;
    background: linear-gradient(120deg, %NAVY% 0%, #1E1B4B 100%);
    border-radius: 18px; padding: 18px 24px; margin-bottom: 18px;
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.18);
}
.brand { display: flex; align-items: center; gap: 14px; }
.brand-logo {
    width: 46px; height: 46px; border-radius: 12px; background: %GREEN%;
    display: flex; align-items: center; justify-content: center; color: white;
}
.brand-title { color: #FFFFFF; font-size: 1.35rem; font-weight: 800; letter-spacing: -0.01em; }
.brand-sub { color: #94A3B8; font-size: 0.82rem; font-weight: 500; margin-top: 2px; }
.pills { display: flex; gap: 10px; flex-wrap: wrap; }
.pill {
    display: inline-flex; align-items: center; gap: 8px; padding: 7px 14px; border-radius: 999px;
    font-size: 0.8rem; font-weight: 600; background: rgba(255,255,255,0.08); color: #E2E8F0;
    border: 1px solid rgba(255,255,255,0.12);
}
.pill .pdot { width: 8px; height: 8px; border-radius: 50%; background: #94A3B8; }
.pill.ok .pdot { background: %GREEN%; box-shadow: 0 0 0 3px rgba(16,185,129,0.25); }
.pill.bad .pdot { background: %RED%; box-shadow: 0 0 0 3px rgba(239,68,68,0.25); }

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid %BORDER%; }
.stTabs [data-baseweb="tab"] {
    background: transparent; border-radius: 10px 10px 0 0; padding: 10px 18px;
    color: %TEXT_MUTED% !important; font-weight: 700;
}
.stTabs [aria-selected="true"] { color: %INDIGO% !important; background: %INDIGO_LIGHT%; }
.stTabs [aria-selected="true"] p { color: %INDIGO% !important; }

/* ---------- Buttons ---------- */
div.stButton > button {
    font-size: 0.88rem !important; padding: 0.55rem 0.9rem !important; height: auto !important;
    border-radius: 12px !important; border: 1px solid %BORDER% !important;
    background: #ffffff; color: %TEXT_DARK% !important; font-weight: 700 !important;
    transition: transform 0.12s ease, box-shadow 0.12s ease;
}
div.stButton > button:hover:not(:disabled) {
    transform: translateY(-1px); box-shadow: 0 8px 20px rgba(15,23,42,0.10);
    border-color: %INDIGO% !important;
}
div.stButton > button:disabled { opacity: 0.45; }

.st-key-start_btn div.stButton > button {
    background: %GREEN% !important; color: white !important; border: none !important; padding: 0.8rem 0.9rem !important; font-size: 0.95rem !important;
}
.st-key-stop_btn div.stButton > button {
    background: %RED% !important; color: white !important; border: none !important; padding: 0.8rem 0.9rem !important; font-size: 0.95rem !important;
}
.st-key-cal_btn div.stButton > button {
    background: %BLUE% !important; color: white !important; border: none !important;
}

/* ---------- KPI cards ---------- */
.kpi {
    background: #ffffff; border: 1px solid %BORDER%; border-radius: 16px; padding: 16px 18px;
    display: flex; align-items: center; gap: 14px; height: 98px; box-sizing: border-box;
    box-shadow: 0 1px 3px rgba(15,23,42,0.05); position: relative; overflow: hidden;
}
.kpi::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px; background: var(--accent); }
.kpi-icon {
    width: 46px; height: 46px; border-radius: 12px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
}
.kpi-label { font-size: 0.74rem; color: %TEXT_MUTED%; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; }
.kpi-value { font-size: 1.7rem; font-weight: 800; line-height: 1.15; margin-top: 2px; }

/* ---------- Panel cards ---------- */
.panel-title {
    display: flex; align-items: center; gap: 8px; font-size: 0.95rem; font-weight: 700; color: %TEXT_DARK%;
}
.panel-sub { font-size: 0.8rem; color: %TEXT_MUTED%; margin-top: 2px; }
.st-key-ctrl_card, .st-key-chart_card, .st-key-side_card {
    background: #ffffff; border: 1px solid %BORDER%; border-radius: 16px; padding: 18px 20px;
    box-shadow: 0 1px 3px rgba(15,23,42,0.05);
}
.section-label {
    font-size: 0.72rem; color: %TEXT_MUTED%; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.07em; margin: 14px 0 6px 0;
}

/* ---------- Camera monitor ---------- */
.cam-panel {
    background: %NAVY%; border-radius: 16px; padding: 14px 16px 12px 16px;
    box-shadow: 0 10px 30px rgba(15,23,42,0.20);
}
.cam-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; }
.cam-title { display: flex; align-items: center; gap: 8px; color: #F8FAFC; font-weight: 700; font-size: 0.95rem; }
.cam-badge {
    display: inline-flex; align-items: center; gap: 6px; font-size: 0.7rem; font-weight: 800;
    letter-spacing: 0.1em; padding: 4px 10px; border-radius: 999px;
}
.cam-badge .bdot { width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
.cam-badge.live { background: rgba(239,68,68,0.18); color: #F87171; }
.cam-badge.live .bdot { animation: blink 1.2s infinite; }
.cam-badge.off { background: rgba(148,163,184,0.18); color: #94A3B8; }
@keyframes blink { 0%,100% { opacity: 1; } 50% { opacity: 0.25; } }

.cam-view {
    position: relative; width: 100%; aspect-ratio: 16 / 9; background: #020617; border-radius: 12px;
    overflow: hidden; display: flex; align-items: center; justify-content: center;
    border: 1px solid #1E293B;
}
.cam-view img { width: 100%; height: 100%; object-fit: contain; display: block; }
.cam-empty { text-align: center; color: #64748B; font-size: 0.85rem; padding: 0 20px; }
.cam-empty svg { opacity: 0.6; margin-bottom: 8px; }
.cam-empty b { color: #94A3B8; display: block; font-size: 0.95rem; margin-bottom: 2px; }
.corner { position: absolute; width: 22px; height: 22px; border-color: rgba(255,255,255,0.55); border-style: solid; }
.c-tl { top: 10px; left: 10px; border-width: 2px 0 0 2px; border-radius: 4px 0 0 0; }
.c-tr { top: 10px; right: 10px; border-width: 2px 2px 0 0; border-radius: 0 4px 0 0; }
.c-bl { bottom: 10px; left: 10px; border-width: 0 0 2px 2px; border-radius: 0 0 0 4px; }
.c-br { bottom: 10px; right: 10px; border-width: 0 2px 2px 0; border-radius: 0 0 4px 0; }
.cam-ts {
    position: absolute; bottom: 12px; left: 40px; font-size: 0.72rem; font-weight: 600; color: #E2E8F0;
    background: rgba(2,6,23,0.65); padding: 3px 8px; border-radius: 6px; font-variant-numeric: tabular-nums;
}
.cam-foot { display: flex; gap: 14px; align-items: center; margin-top: 10px; color: #94A3B8; font-size: 0.76rem; font-weight: 500; }
.cam-foot .tag { display: inline-flex; align-items: center; gap: 6px; }
.cam-foot .sq { width: 9px; height: 9px; border-radius: 3px; }

/* ---------- Legend ---------- */
.legend-row { text-align: right; font-weight: 600; color: %TEXT_DARK%; font-size: 0.82rem; }
.legend-dot { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin: 0 4px 0 12px; vertical-align: middle; }

/* ---------- Responsive ---------- */
@media (max-width: 768px) {
    [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; }
    [data-testid="column"] { min-width: 100% !important; margin-bottom: 12px; }
    .block-container { padding: 1rem !important; }
    .topbar { padding: 14px 16px; }
}
</style>
"""
_tokens = {
    "%GRAYBG%": "#F4F6FB", "%TEXT_DARK%": TEXT_DARK, "%TEXT_MUTED%": TEXT_MUTED,
    "%NAVY_2%": NAVY_2, "%NAVY%": NAVY, "%INDIGO_LIGHT%": INDIGO_LIGHT, "%INDIGO%": INDIGO,
    "%GREEN%": GREEN, "%RED%": RED, "%BLUE%": BLUE, "%BORDER%": BORDER,
}
for _k, _v in _tokens.items():
    CSS = CSS.replace(_k, _v)
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def kpi_card(icon: str, icon_bg: str, accent: str, label: str, value, value_color=None):
    vc = value_color or TEXT_DARK
    st.markdown(
        f"<div class='kpi' style='--accent:{accent};'>"
        f"<div class='kpi-icon' style='background:{icon_bg};'>{svg(icon, 22, accent)}</div>"
        f"<div><div class='kpi-label'>{label}</div>"
        f"<div class='kpi-value' style='color:{vc};'>{value}</div></div></div>",
        unsafe_allow_html=True,
    )


def panel_title(icon: str, title: str, sub: str = "", color: str = INDIGO):
    sub_html = f"<div class='panel-sub'>{sub}</div>" if sub else ""
    st.markdown(
        f"<div class='panel-title'>{svg(icon, 18, color)} {title}</div>{sub_html}",
        unsafe_allow_html=True,
    )


def frame_to_data_uri(cam_frame):
    """Turn whatever the Pi published into a data URI (or None if undecodable)."""
    if not cam_frame:
        return None
    try:
        if isinstance(cam_frame, str) and cam_frame.startswith("data:"):
            return cam_frame
        raw = base64.b64decode(cam_frame)
        mime = "image/png" if raw[:4] == b"\x89PNG" else "image/jpeg"
        return f"data:{mime};base64,{base64.b64encode(raw).decode()}"
    except Exception:
        return None


def camera_panel(cam_frame, connected: bool):
    uri = frame_to_data_uri(cam_frame)
    live = uri is not None
    badge = (
        "<span class='cam-badge live'><span class='bdot'></span>LIVE</span>"
        if live else
        "<span class='cam-badge off'><span class='bdot'></span>NO SIGNAL</span>"
    )
    if live:
        inner = f"<img src='{uri}' alt='camera feed'/>"
        ts = f"<div class='cam-ts'>{datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}</div>"
    else:
        msg = ("Waiting for the Pi / vision script to publish a frame."
               if connected else "Connect to the machine from the sidebar to start the feed.")
        inner = (
            f"<div class='cam-empty'>{svg('camera', 42, '#64748B')}"
            f"<b>No camera signal</b>{msg}</div>"
        )
        ts = ""
    st.markdown(
        "<div class='cam-panel'>"
        f"<div class='cam-head'><div class='cam-title'>{svg('camera', 18, '#A5B4FC')} Live Camera Feed</div>{badge}</div>"
        f"<div class='cam-view'>{inner}"
        "<div class='corner c-tl'></div><div class='corner c-tr'></div>"
        f"<div class='corner c-bl'></div><div class='corner c-br'></div>{ts}</div>"
        "<div class='cam-foot'>"
        f"<span class='tag'><span class='sq' style='background:{GREEN};'></span>1 = Good</span>"
        f"<span class='tag'><span class='sq' style='background:{RED};'></span>0 = Bad</span>"
        "<span style='margin-left:auto;'>Labelled frames from the Pi</span></div>"
        "</div>",
        unsafe_allow_html=True,
    )


def _chart_theme(chart):
    return (
        chart.configure_view(strokeWidth=0)
        .configure_axis(gridColor="#F1F5F9", domainColor=BORDER, tickColor=BORDER,
                        labelColor=TEXT_MUTED, titleColor=TEXT_MUTED, labelFont="Inter", titleFont="Inter")
        .configure_legend(labelColor=TEXT_DARK, titleColor=TEXT_DARK, orient="top")
    )


def trend_chart(df: pd.DataFrame, height=280):
    long_df = df.melt(id_vars=["time"], value_vars=["good", "bad", "total"],
                      var_name="type", value_name="count")
    chart = (
        alt.Chart(long_df)
        .mark_line(point=alt.OverlayMarkDef(size=45), strokeWidth=2.5, interpolate="monotone")
        .encode(
            x=alt.X("time:N", title="Time", axis=alt.Axis(labelAngle=-40)),
            y=alt.Y("count:Q", title="Seeds counted"),
            color=alt.Color(
                "type:N", title="",
                scale=alt.Scale(domain=["good", "bad", "total"],
                                range=[GOOD_COLOR, BAD_COLOR, TOTAL_COLOR]),
            ),
            tooltip=["time", "type", "count"],
        )
        .properties(height=height, background="white")
        .interactive()
    )
    return _chart_theme(chart)


def good_bad_bar(good: int, bad: int, height=240):
    df = pd.DataFrame({"Category": ["Good", "Bad"], "Count": [good, bad]})
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8, size=56)
        .encode(
            x=alt.X("Category:N", title="", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Count:Q", title="Seeds"),
            color=alt.Color(
                "Category:N", legend=None,
                scale=alt.Scale(domain=["Good", "Bad"], range=[GOOD_COLOR, BAD_COLOR]),
            ),
            tooltip=["Category", "Count"],
        )
        .properties(height=height, background="white")
    )
    return _chart_theme(chart)


def start_new_session(good: int, bad: int):
    st.session_state.active_session = {
        "start_dt": datetime.now(), "start_good": good, "start_bad": bad,
    }
    st.session_state.live_series = []
    st.session_state.last_run_series = []


def end_active_session(good: int, bad: int):
    active = st.session_state.active_session
    if not active:
        return
    session_good = good - active["start_good"]
    session_bad = bad - active["start_bad"]
    session_total = session_good + session_bad
    st.session_state.sessions.append({
        "id": len(st.session_state.sessions) + 1,
        "start_dt": active["start_dt"],
        "end_dt": datetime.now(),
        "good": session_good, "bad": session_bad, "total": session_total,
        "good_pct": (session_good / session_total * 100) if session_total else 0.0,
        "series": list(st.session_state.live_series),
    })
    st.session_state.last_run_series = list(st.session_state.live_series)
    st.session_state.active_session = None
    st.session_state.live_series = []


# --------------------------------------------------------------------------
# Sidebar - connection settings
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Connection")

    mode = st.radio(
        "Mode", MODE_OPTIONS, index=MODE_OPTIONS.index(st.session_state.mode),
        help="Cloud: Pi + dashboard talk over the internet via Firebase.\n"
             "Live: wired by USB, same computer.\nSimulation: demo with fake data.",
    )
    st.session_state.mode = mode

    port, baud = None, None
    db_url = DEFAULT_FIREBASE_URL
    if mode.startswith("Live"):
        port = st.text_input("Serial port", value="COM3",
                             help="e.g. COM3 on Windows, /dev/ttyACM0 on Linux/Pi.")
        baud = st.number_input("Baud rate", value=9600, step=1)
    elif mode.startswith("Cloud"):
        # Pre-filled and saved in the code — only open this to point at another database.
        with st.expander("Firebase database", expanded=False):
            db_url = st.text_input("Database URL", value=DEFAULT_FIREBASE_URL,
                                   help="Saved default. Edit DEFAULT_FIREBASE_URL in app.py to change it permanently.")
        st.caption("Using saved Firebase database")

    conn_container = st.container(key="conn_btns")
    col_a, col_b = conn_container.columns(2)
    with col_a:
        if st.button("Connect", use_container_width=True, type="primary", icon=":material/power:"):
            if st.session_state.bridge:
                st.session_state.bridge.disconnect()
            bridge = None
            if mode.startswith("Live"):
                bridge = SerialBridge(port, int(baud))
            elif mode.startswith("Cloud"):
                bridge = CloudBridge(db_url.strip()) if db_url and db_url.strip() else None
                if bridge is None:
                    st.error("Firebase URL is empty.")
            else:
                bridge = SimulationBridge()
            if bridge is not None:
                bridge.connect()
                st.session_state.bridge = bridge
    with col_b:
        if st.button("Disconnect", use_container_width=True, icon=":material/power_off:"):
            if st.session_state.bridge:
                st.session_state.bridge.disconnect()
            st.session_state.bridge = None

    st.divider()
    bridge = st.session_state.bridge
    if bridge is None:
        st.warning("Not connected. Click **Connect** above.")
    else:
        snap = bridge.get_snapshot()
        if snap["connected"]:
            st.success("Connected")
        else:
            st.error(f"Connection failed: {snap.get('error') or 'unknown error'}")

    st.divider()
    show_camera = st.checkbox("Show camera feed", value=True)

    st.divider()
    n_sessions = len(st.session_state.sessions)
    st.caption(f"**{n_sessions}** past session(s) logged — see the History tab.")
    if n_sessions:
        last = st.session_state.sessions[-1]
        st.caption(f"Last run: {last['good']} good / {last['bad']} bad "
                   f"({last['good_pct']:.1f}% good)")

    with st.expander("Protocol reference"):
        st.caption(
            "**USB protocol** — in: `0`/`1` (bad/good, per-seed) · `GOOD`/`BAD` · "
            "`COUNTS:<g>,<b>` · `STATUS:IDLE/RUNNING/CALIBRATION/FAULT` · "
            "`DIVERTER:READY/DIVERTING` — out: `START`/`STOP`/`CALIBRATE`/"
            "`SPD1`/`SPD2`/`SPD3`/`SERVO50`/`SERVO100`/`SERVO150`/`SERVO180`\n\n"
            "**Cloud protocol** — Firebase keys: `good`, `bad` (running totals), "
            "`status` (Idle/Running/Calibration/Fault), `diverter` (Ready/Diverting), "
            "`camera_frame`, `command`.\n\n"
            "**Speed levels** — 3.75 / 6.2 / 7.4 cm/s (level 1 is measured; "
            "2 and 3 are still computed estimates)."
        )

# --------------------------------------------------------------------------
# Top bar
# --------------------------------------------------------------------------
_b = st.session_state.bridge
if _b is None:
    conn_cls, conn_txt = "", "Disconnected"
else:
    _s = _b.get_snapshot()
    conn_cls, conn_txt = ("ok", "Connected") if _s["connected"] else ("bad", "Connection failed")
mode_short = mode.split(" (")[0]

st.markdown(
    "<div class='topbar'>"
    f"<div class='brand'><div class='brand-logo'>{svg('leaf', 26, 'white')}</div>"
    "<div><div class='brand-title'>Seed Sorting Control Center</div>"
    "<div class='brand-sub'>Real-time monitoring and control for the automated maize sorting system</div></div></div>"
    f"<div class='pills'><span class='pill'>{svg('wifi', 14, '#CBD5E1')} {mode_short} mode</span>"
    f"<span class='pill {conn_cls}'><span class='pdot'></span>{conn_txt}</span></div>"
    "</div>",
    unsafe_allow_html=True,
)

tab_live, tab_history = st.tabs(["Live Dashboard", "History"])

# --------------------------------------------------------------------------
# LIVE TAB
# --------------------------------------------------------------------------
with tab_live:

    @st.fragment(run_every=1)
    def live_dashboard():
        bridge = st.session_state.bridge
        if bridge is None:
            good, bad, status, diverter, cam_frame, connected = 0, 0, "Idle", "Ready", None, False
        else:
            snap = bridge.get_snapshot()
            good, bad, status = snap["good"], snap["bad"], snap["status"]
            diverter = snap.get("diverter", "Ready")
            cam_frame = snap.get("camera_frame")
            connected = bool(snap.get("connected"))

        total = good + bad
        good_pct = (good / total * 100) if total else 0.0

        series = st.session_state.live_series
        if st.session_state.active_session and (
            not series or series[-1]["good"] != good or series[-1]["bad"] != bad
        ):
            series.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "good": good, "bad": bad, "total": total,
            })

        # ---------------- KPI row ----------------
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            s_icon, s_bg, s_color = STATUS_STYLE.get(status, STATUS_STYLE["Idle"])
            kpi_card(s_icon, s_bg, s_color, "Machine Status",
                     status if status in STATUS_STYLE else "Idle", value_color=s_color)
        with k2:
            if diverter == "Diverting":
                d_bg, d_color = AMBER_LIGHT, AMBER
            else:
                d_bg, d_color = GRAY_LIGHT, TEXT_MUTED
            kpi_card("shuffle", d_bg, d_color, "Diverter", diverter, value_color=d_color)
        with k3:
            kpi_card("check", GREEN_LIGHT, GREEN, "Good Seeds", f"{good:,}")
        with k4:
            kpi_card("x", RED_LIGHT, RED, "Bad / Defective", f"{bad:,}")
        with k5:
            kpi_card("layers", INDIGO_LIGHT, INDIGO, "Total Counted", f"{total:,}")

        st.write("")

        # ---------------- Camera + Controls ----------------
        if show_camera:
            cam_col, ctrl_col = st.columns([1.7, 1], gap="medium")
        else:
            cam_col, ctrl_col = None, st.container()

        if cam_col is not None:
            with cam_col:
                camera_panel(cam_frame, connected)

        with ctrl_col:
            with st.container(key="ctrl_card"):
                panel_title("sliders", "Machine Controls", "Start, stop and tune the sorter")

                st.markdown("<div class='section-label'>Run</div>", unsafe_allow_html=True)
                b1, b2 = st.columns(2)
                with b1:
                    with st.container(key="start_btn"):
                        if st.button("START", use_container_width=True, icon=":material/play_arrow:"):
                            bridge = st.session_state.bridge
                            if bridge is None:
                                st.error("Connect to the machine first (see sidebar).")
                            else:
                                snap = bridge.get_snapshot()
                                if bridge.send_command("START"):
                                    if not st.session_state.active_session:
                                        start_new_session(snap["good"], snap["bad"])
                                    st.toast("START sent — new session logging began", icon="✅")
                                else:
                                    st.error("Failed to send START command.")
                with b2:
                    with st.container(key="stop_btn"):
                        if st.button("STOP", use_container_width=True, icon=":material/stop:"):
                            bridge = st.session_state.bridge
                            if bridge is None:
                                st.error("Connect to the machine first (see sidebar).")
                            else:
                                snap = bridge.get_snapshot()
                                if bridge.send_command("STOP"):
                                    end_active_session(snap["good"], snap["bad"])
                                    st.toast("STOP sent — session saved to History", icon="🛑")
                                else:
                                    st.error("Failed to send STOP command.")

                if status == "Fault":
                    st.error("Machine is in FAULT — press STOP to clear it before starting again.")

                st.markdown("<div class='section-label'>Belt speed</div>", unsafe_allow_html=True)
                sp_cols = st.columns(3)
                for col, (level, cms) in zip(sp_cols, SPEED_LEVELS):
                    with col:
                        if st.button(f"{cms:.2f} cm/s", use_container_width=True,
                                     key=f"spd_{level}", icon=":material/speed:"):
                            bridge = st.session_state.bridge
                            if bridge is None:
                                st.error("Connect to the machine first (see sidebar).")
                            elif bridge.send_command(f"SPD{level}"):
                                st.toast(f"Speed set to {cms:.2f} cm/s", icon="⚙️")
                            else:
                                st.error("Failed to send speed command.")

                st.markdown("<div class='section-label'>Calibration</div>", unsafe_allow_html=True)
                with st.container(key="cal_btn"):
                    if st.button("CALIBRATE", use_container_width=True, icon=":material/tune:"):
                        bridge = st.session_state.bridge
                        if bridge is None:
                            st.error("Connect to the machine first (see sidebar).")
                        elif bridge.send_command("CALIBRATE"):
                            st.toast("Calibration started — belt + singulator run at the "
                                     "lowest speed until STOP is pressed", icon="🛠️")
                        else:
                            st.error("Failed to send CALIBRATE command.")

                st.markdown("<div class='section-label'>Servo bring-up test (Idle only)</div>",
                            unsafe_allow_html=True)
                sv_cols = st.columns(4)
                for col, angle in zip(sv_cols, (50, 100, 150, 180)):
                    with col:
                        if st.button(f"{angle}°", use_container_width=True,
                                     key=f"servo_{angle}", disabled=(status != "Idle")):
                            bridge = st.session_state.bridge
                            if bridge is None:
                                st.error("Connect to the machine first (see sidebar).")
                            elif bridge.send_command(f"SERVO{angle}"):
                                st.toast(f"Servo moved to {angle}°", icon="🔧")
                            else:
                                st.error("Failed to send servo test command.")
                st.caption("Moves the gate directly, bypassing the FIFO.")

        st.write("")

        # ---------------- Analytics ----------------
        if series:
            df = pd.DataFrame(series)
        elif st.session_state.last_run_series:
            df = pd.DataFrame(st.session_state.last_run_series)
        else:
            df = pd.DataFrame([{
                "time": datetime.now().strftime("%H:%M:%S"),
                "good": good, "bad": bad, "total": total,
            }])

        left, right = st.columns([2, 1], gap="medium")
        with left:
            with st.container(key="chart_card"):
                h1, h2 = st.columns([2, 1])
                with h1:
                    panel_title("trend", "Current Run", f"{total:,} seeds sorted")
                with h2:
                    st.markdown(
                        f"<div class='legend-row'>"
                        f"<span class='legend-dot' style='background:{GREEN};'></span>Good"
                        f"<span class='legend-dot' style='background:{RED};'></span>Bad"
                        f"<span class='legend-dot' style='background:{INDIGO};'></span>Total</div>",
                        unsafe_allow_html=True,
                    )
                st.altair_chart(trend_chart(df, height=270), use_container_width=True)
                if not series:
                    st.caption("Press START to begin logging this run — it is saved to History when you press STOP.")
        with right:
            with st.container(key="side_card"):
                panel_title("percent", "Sort Quality", "Good vs bad split")
                st.altair_chart(good_bad_bar(good, bad, height=190), use_container_width=True)
                st.progress(min(good_pct / 100, 1.0), text=f"Good seed rate: {good_pct:.1f}%")

    live_dashboard()

# --------------------------------------------------------------------------
# HISTORY TAB
# --------------------------------------------------------------------------
with tab_history:
    st.write("")
    panel_title("history", "Past Sorting Sessions", "Every START → STOP run is saved here")
    st.write("")

    sessions = st.session_state.sessions
    if not sessions:
        st.info("No completed sessions yet. Run the machine (START → STOP) on the "
                "Live Dashboard tab, and each run will show up here afterward.")
    else:
        summary_rows = [{
            "Session": s["id"],
            "Start": s["start_dt"].strftime("%Y-%m-%d %H:%M:%S"),
            "End": s["end_dt"].strftime("%Y-%m-%d %H:%M:%S"),
            "Good": s["good"], "Bad": s["bad"], "Total": s["total"],
            "Good %": round(s["good_pct"], 1),
        } for s in sessions]
        summary_df = pd.DataFrame(summary_rows)

        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        all_csv = summary_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download all sessions (CSV)", data=all_csv,
            file_name=f"seed_sorting_sessions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv", icon=":material/download:",
        )

        st.divider()
        options = [f"Session {s['id']} — {s['start_dt'].strftime('%Y-%m-%d %H:%M')}" for s in sessions]
        pick = st.selectbox("View details for a specific session:", options[::-1])
        chosen = sessions[options.index(pick)]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi_card("check", GREEN_LIGHT, GREEN, "Good", f"{chosen['good']:,}")
        with c2:
            kpi_card("x", RED_LIGHT, RED, "Bad", f"{chosen['bad']:,}")
        with c3:
            kpi_card("layers", INDIGO_LIGHT, INDIGO, "Total", f"{chosen['total']:,}")
        with c4:
            kpi_card("percent", BLUE_LIGHT, BLUE, "Good rate", f"{chosen['good_pct']:.1f}%")

        st.write("")
        if chosen["series"]:
            sdf = pd.DataFrame(chosen["series"])
            gcol, bcol = st.columns([2, 1], gap="medium")
            with gcol:
                st.altair_chart(trend_chart(sdf), use_container_width=True)
            with bcol:
                st.altair_chart(good_bad_bar(chosen["good"], chosen["bad"]), use_container_width=True)

            session_csv = sdf.to_csv(index=False).encode("utf-8")
            st.download_button(
                f"Download Session {chosen['id']} detail (CSV)", data=session_csv,
                file_name=f"seed_sorting_session_{chosen['id']}.csv", mime="text/csv",
                icon=":material/download:",
            )
        else:
            st.caption("No detailed readings were logged during this session.")
