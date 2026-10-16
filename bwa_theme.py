"""Visual theme for the Streamlit frontend: neutral surfaces, soft cards, one blue accent."""
from __future__ import annotations

import html

import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --ink: #16161a;
  --muted: #6b6b73;
  --surface: #f5f4f2;
  --sidebar: #efeeec;
  --tile: #ffffff;
  --accent: #3b5bdb;
  --accent-soft: #e7ebfc;
  --positive: #1b7a46;
  --positive-soft: #e3f4ea;
  --radius: 20px;
}

html, body, [class*="css"], .stApp { font-family: 'Inter', system-ui, sans-serif; color: var(--ink); }
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { padding: 2.4rem 3rem 4rem; max-width: 1280px; }

/* Sidebar */
section[data-testid="stSidebar"] { background: var(--sidebar); border-right: none; }
section[data-testid="stSidebar"] .block-container { padding: 1.6rem 1.1rem; }
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
  font-size: 0.8rem; font-weight: 600; letter-spacing: .04em; text-transform: uppercase; color: var(--muted);
}
section[data-testid="stSidebar"] textarea, section[data-testid="stSidebar"] input {
  background: var(--tile) !important; border: 1px solid transparent !important; border-radius: 14px !important;
}
section[data-testid="stSidebar"] [data-baseweb="textarea"], section[data-testid="stSidebar"] [data-baseweb="input"] {
  background: var(--tile); border-radius: 14px; border: none;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label {
  background: transparent; border-radius: 12px; padding: .45rem .6rem; width: 100%;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
  background: var(--tile); box-shadow: 0 1px 2px rgba(0,0,0,.05);
}
section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child { display: none; }

/* Buttons */
.stButton > button, .stDownloadButton > button {
  border-radius: 14px; font-weight: 500; padding: .55rem 1.1rem;
  border: 1px solid transparent; background: var(--tile); color: var(--ink);
}
.stButton > button:hover, .stDownloadButton > button:hover { border-color: var(--ink); color: var(--ink); }
.stButton > button[kind="primary"] { background: var(--ink); color: #fff; width: 100%; }
.stButton > button[kind="primary"]:hover { background: #2c2c33; color: #fff; border-color: transparent; }

/* Tabs as pills */
.stTabs [data-baseweb="tab-list"] { gap: .35rem; border-bottom: none; background: var(--surface); padding: .35rem; border-radius: 16px; width: fit-content; }
.stTabs [data-baseweb="tab"] { height: auto; padding: .5rem 1rem; border-radius: 12px; color: var(--muted); font-weight: 500; }
.stTabs [aria-selected="true"] { background: var(--tile); color: var(--ink); box-shadow: 0 1px 2px rgba(0,0,0,.06); }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.6rem; }

/* Page header */
.page-title { font-size: 2rem; font-weight: 600; letter-spacing: -.02em; margin: 0; }
.page-sub { color: var(--muted); margin: .25rem 0 1.6rem; font-size: .98rem; }
.brand { font-size: 1.25rem; font-weight: 700; letter-spacing: -.02em; margin-bottom: 1.4rem; }
.section-label { font-size: 1.05rem; font-weight: 600; margin: 1.8rem 0 .8rem; }

/* Cards */
.card { background: var(--surface); border-radius: var(--radius); padding: 1.4rem 1.5rem; }
.stat-label { color: var(--muted); font-size: .9rem; }
.stat-value { font-size: 2.1rem; font-weight: 600; letter-spacing: -.03em; line-height: 1.15; margin: .15rem 0; }
.stat-note { color: var(--muted); font-size: .82rem; }
.stat-note.pos { color: var(--positive); }
.tile-row { display: flex; gap: .6rem; margin-top: 1rem; flex-wrap: wrap; }
.tile { background: var(--tile); border-radius: 14px; padding: .65rem .9rem; flex: 1; min-width: 110px; }
.tile b { display: block; font-weight: 500; }
.tile span { color: var(--muted); font-size: .8rem; }
.badge { display: inline-block; background: var(--accent-soft); color: #1d3a9e; border-radius: 999px; padding: .15rem .65rem; font-size: .78rem; font-weight: 500; }
.badge.pos { background: var(--positive-soft); color: var(--positive); }
.badge.plain { background: var(--tile); color: var(--ink); }
.bar { display: flex; gap: 4px; margin-top: 1rem; }
.bar i { display: block; height: 4px; border-radius: 4px; background: var(--accent); }
.task { display: flex; justify-content: space-between; align-items: center; gap: 1rem; background: var(--tile); border-radius: 14px; padding: .85rem 1rem; margin-bottom: .5rem; }
.task .t { font-weight: 500; }
.task .m { color: var(--muted); font-size: .82rem; }

/* Native widgets */
div[data-testid="stDataFrame"] { border-radius: 16px; overflow: hidden; }
div[data-testid="stStatusWidget"], details[data-testid="stExpander"], div[data-testid="stStatus"] { border-radius: 16px; border-color: transparent; background: var(--surface); }
div[data-testid="stAlert"] { border-radius: 14px; }
</style>
"""


def apply_theme() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def esc(x) -> str:
    return html.escape(str(x if x is not None else ""))


def header(title: str, subtitle: str) -> None:
    st.markdown(f'<h1 class="page-title">{esc(title)}</h1><div class="page-sub">{esc(subtitle)}</div>', unsafe_allow_html=True)


def section_label(text: str) -> None:
    st.markdown(f'<div class="section-label">{esc(text)}</div>', unsafe_allow_html=True)


def stat_card(label: str, value, note: str = "", positive: bool = False) -> str:
    cls = "stat-note pos" if positive else "stat-note"
    return (
        f'<div class="card"><div class="stat-label">{esc(label)}</div>'
        f'<div class="stat-value">{esc(value)}</div><div class="{cls}">{esc(note)}</div></div>'
    )


def stat_row(items: list[tuple]) -> None:
    cols = st.columns(len(items))
    for col, it in zip(cols, items):
        col.markdown(stat_card(*it), unsafe_allow_html=True)


def tiles(pairs: list[tuple]) -> str:
    inner = "".join(f'<div class="tile"><b>{esc(v)}</b><span>{esc(k)}</span></div>' for k, v in pairs)
    return f'<div class="tile-row">{inner}</div>'
