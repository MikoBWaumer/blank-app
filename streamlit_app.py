# app.py
import io
import os
import hashlib
from typing import Dict, List, Union, Literal
import numpy as np
import pandas as pd
import streamlit as st
from pandas.api.types import is_numeric_dtype

# Defensive Plotly import (avoid hard crash on missing install)
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

# ──────────────────────────────────────────────────────────────────────────────
# Streamlit page setup
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Multi-CSV Visualizer", layout="wide")
st.title("Pressure / Sensors CSV Visualizer — Multi-file")

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────
def pretty_name(source: str) -> str:
    """Shorten legend labels: strip path + .csv."""
    base = source.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    return base[:-4] if base.lower().endswith(".csv") else base

def sha1(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

def try_import_pyarrow() -> bool:
    try:
        import pyarrow as pa  # noqa: F401
        import pyarrow.parquet as pq  # noqa: F401
        return True
    except Exception:
        return False

def parquet_cache_paths() -> str:
    cache_dir = os.path.join(os.getcwd(), "st_parquet_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

def apply_deadband(
    series: pd.Series,
    band_abs: float = 0.0,
    band_pct: float = 0.0,
    pct_ref: Union[Literal["previous_output", "range"], float] = "previous_output",
) -> pd.Series:
    """Hysteresis deadband. Output updates only when |x - last_y| > band."""
    s = series.astype(float).values
    n = len(s)
    if n == 0:
        return series
    out = np.empty_like(s)
    y = s[0]; out[0] = y
    rng = np.nanmax(s) - np.nanmin(s) if pct_ref == "range" else None
    for i in range(1, n):
        x = s[i]
        if isinstance(pct_ref, (int, float)):
            ref = float(pct_ref)
        elif pct_ref == "range":
            ref = rng if (rng and rng > 0) else (abs(y) if y != 0 else 1.0)
        else:
            ref = abs(y) if y != 0 else 1.0
        band = max(band_abs, band_pct * ref)
        if abs(x - y) > band:
            y = x
        out[i] = y
    return pd.Series(out, index=series.index, name=series.name)

def numeric_coercion_audit(df_before: pd.DataFrame, cols: List[str]) -> Dict[str, int]:
    """Estimate how many values were coerced to NaN when forcing numeric."""
    report: Dict[str, int] = {}
    for c in cols:
        if c in df_before.columns and not is_numeric_dtype(df_before[c]):
            s = df_before[c]
            na0 = s.isna().sum()
            s_num = pd.to_numeric(s, errors="coerce")
            na1 = s_num.isna().sum()
            report[c] = int(max(0, na1 - na0))
    return report

# ──────────────────────────────────────────────────────────────────────────────
# Presets via URL query params
# ──────────────────────────────────────────────────────────────────────────────
def get_qp() -> Dict[str, List[str]]:
    if hasattr(st, "query_params"):  # Streamlit ≥ 1.30
        return dict(st.query_params)
    return st.experimental_get_query_params()

def set_qp(**kwargs):
    if hasattr(st, "query_params"):
        st.query_params.clear()
        for k, v in kwargs.items():
            st.query_params[k] = v
    else:
        st.experimental_set_query_params(**kwargs)

qp = get_qp()

def qp_get_str(key: str, default: str) -> str:
    v = qp.get(key, [default]); return v[0] if isinstance(v, list) else v

def qp_get_float(key: str, default: float) -> float:
    v = qp_get_str(key, str(default))
    try: return float(v)
    except Exception: return default

def qp_get_bool(key: str, default: bool) -> bool:
    v = qp_get_str(key, "1" if default else "0")
    return v in ("1", "true", "True", "yes")

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar controls
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Upload & Options")
    files = st.file_uploader("Upload one or more CSV files", type=["csv"], accept_multiple_files=True)

    display_mode = st.radio(
        "Display mode",
        ["Overlay (compare on same axes)", "Stacked (one panel per file)"],
        index=0 if qp_get_str("mode", "overlay").lower() == "overlay" else 1
    )
    x_mode = st.radio(
        "X-axis",
        ["Timestamp", "Relative time (seconds from file start)"],
        index=0 if qp_get_str("x", "ts") == "ts" else 1
    )

    resample_rule = st.selectbox(
        "Resample (time aggregation)",
        ["None", "0.5s", "1s", "5s", "10s", "30s", "1min", "5min"],
        index=["None", "0.5s", "1s", "5s", "10s", "30s", "1min", "5min"].index(qp_get_str("rs", "5s"))
    )
    aggregator = st.selectbox(
        "Aggregation", ["mean", "median", "min", "max"],
        index=["mean", "median", "min", "max"].index(qp_get_str("agg", "mean"))
    )
    show_relay = st.checkbox("Overlay relay_on (stepped)", value=qp_get_bool("relay", False))
    st.divider()

    st.subheader("Deadband filter")
    enable_deadband = st.checkbox("Enable deadband", value=qp_get_bool("db", True))
    deadband_target = st.multiselect(
        "Apply to columns",
        ["pressure_bar", "voltage_v", "current_mA", "moisture2_v", "moisture3_v"],
        default=["pressure_bar"]
    )
    band_abs = st.number_input("Absolute band (same units as signal)", value=qp_get_float("db_abs", 0.02), step=0.01, format="%.5f")
    band_pct = st.number_input("Percentage band (%)", value=qp_get_float("db_pct", 0.0), step=0.1, format="%.3f") / 100.0
    pct_ref = st.selectbox(
        "Percent band reference",
        ["previous_output", "range", "constant"],
        index=["previous_output", "range", "constant"].index(qp_get_str("db_ref", "previous_output"))
    )
    pct_ref_const = (st.number_input("Constant reference magnitude", value=qp_get_float("db_c", 1.0), step=0.1)
                     if pct_ref == "constant" else None)
    st.divider()

    st.subheader("Appearance")
    overlay_opacity = st.slider("Overlay line opacity", 0.1, 1.0, float(qp_get_float("op", 0.6)), 0.05, help="Used in Overlay mode")
    legend_location = st.radio(
        "Legend location", ["Below", "Right", "Top"],
        index={"below": 0, "right": 1, "top": 2}.get(qp_get_str("leg", "below"), 0),
        help="Place legend under the plot to maximize chart area"
    )
    st.divider()

    st.subheader("Pressure units")
    pressure_units = st.radio("Units for pressure_bar", ["bar", "kPa"], index=0 if qp_get_str("u", "bar") == "bar" else 1)
    st.divider()

    st.subheader("Computed channels")
    computed_channels = st.multiselect(
        "Add computed channels",
        ["pressure rolling mean", "pressure derivative (bar/s)"],
        default=[],
        help="Rolling mean uses point window; derivative uses timestamp spacing."
    )
    roll_window = st.number_input("Rolling mean window (points)", min_value=2, value=30, step=1)
    st.divider()

    st.subheader("Min/Max markers & shading")
    show_extrema = st.checkbox("Show min/max markers", value=False)
    shade_relay = st.checkbox("Shade periods with relay_on == 1", value=False)
    shade_color = st.color_picker("Shade color", value="#1f77b4")
    shade_opacity = st.slider("Shade opacity", 0.0, 0.5, 0.15, 0.01)
    st.divider()

    st.subheader("Presets / Sharing")
    if st.button("Copy current settings to URL"):
        set_qp(
            mode="overlay" if display_mode.startswith("Overlay") else "stacked",
            x="ts" if x_mode == "Timestamp" else "rel",
            rs=resample_rule, agg=aggregator,
            relay="1" if show_relay else "0",
            db="1" if enable_deadband else "0",
            db_abs=str(band_abs), db_pct=str(band_pct * 100.0),
            db_ref=pct_ref, db_c=str(pct_ref_const or 1.0),
            op=str(overlay_opacity), leg=legend_location.lower(),
            u=pressure_units.lower(),
        )
        st.success("URL updated with current settings. Share this page URL.")

    st.divider()
    show_preview = st.checkbox("Show head() preview per file", value=False)
    show_debug = st.checkbox("Debug info", value=False)

if not files:
    st.info("Upload at least one CSV to begin.")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Data loading (cached) + optional Parquet on-disk cache
# ──────────────────────────────────────────────────────────────────────────────
PARQUET_OK = try_import_pyarrow()
PARQ_DIR = parquet_cache_paths() if PARQUET_OK else None

@st.cache_data(show_spinner=True)
def load_one(file_bytes: bytes, name: str, usecols: List[str]) -> pd.DataFrame:
    """
    Load CSV with just the requested columns; if pyarrow is present, persist
    a small parquet cache keyed by file content hash, and project columns on read.
    """
    df = None
    # enforce exactly the columns we want (without "source"; added later)
    usecols = [c for c in usecols if c != "source"]
    parse_dates = ["timestamp_iso"] if "timestamp_iso" in usecols else None

    if PARQUET_OK and PARQ_DIR:
        import pyarrow as pa
        import pyarrow.parquet as pq
        h = sha1(file_bytes)
        parq_path = os.path.join(PARQ_DIR, f"{h}.parquet")
        if os.path.exists(parq_path):
            try:
                # Column projection from Parquet
                table = pq.read_table(parq_path, columns=usecols)
                df = table.to_pandas()
            except Exception:
                df = None
        if df is None:
            # CSV -> DataFrame (just needed columns)
            df = pd.read_csv(
                io.BytesIO(file_bytes),
                usecols=usecols,
                parse_dates=parse_dates,
                low_memory=False,
                dtype={"pressure_bar": "float64"} if "pressure_bar" in usecols else None,
            )
            # Write Parquet with only these columns
            try:
                table = pa.Table.from_pandas(df)
                pq.write_table(table, parq_path)
            except Exception:
                pass
    else:
        df = pd.read_csv(
            io.BytesIO(file_bytes),
            usecols=usecols,
            parse_dates=parse_dates,
            low_memory=False,
            dtype={"pressure_bar": "float64"} if "pressure_bar" in usecols else None,
        )

    df["source"] = name
    return df

# Only read the columns you care about
required = ["timestamp_iso", "pressure_bar", "source"]
loaded: Dict[str, pd.DataFrame] = {f.name: load_one(f.getvalue(), f.name, required) for f in files}

# numeric intersection across files
numeric_by_file = {k: v.select_dtypes(include="number").columns.tolist() for k, v in loaded.items()}
numeric_intersection = set(numeric_by_file[next(iter(loaded))])
for _, cols in numeric_by_file.items():
    numeric_intersection &= set(cols)
numeric_intersection = sorted(numeric_intersection)

ordered_numeric = [c for c in ["pressure_bar", "voltage_v", "current_mA", "moisture2_v", "moisture3_v", "relay_on"]
                   if c in numeric_intersection]
for c in numeric_intersection:
    if c not in ordered_numeric:
        ordered_numeric.append(c)

y_cols = st.multiselect(
    "Y-axis columns (numeric, present in all files)",
    options=[c for c in ordered_numeric if c != "relay_on"],
    default=[c for c in ["pressure_bar", "voltage_v"] if c in ordered_numeric][:2],
    max_selections=6,
)
if not y_cols:
    st.warning("Pick at least one numeric column to plot.")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Time-range controls
# ──────────────────────────────────────────────────────────────────────────────
min_ts = min(v["timestamp_iso"].min() for v in loaded.values())
max_ts = max(v["timestamp_iso"].max() for v in loaded.values())

if x_mode == "Timestamp":
    t0, t1 = st.slider(
        "Time range (timestamp)",
        min_value=min_ts.to_pydatetime(),
        max_value=max_ts.to_pydatetime(),
        value=(min_ts.to_pydatetime(), max_ts.to_pydatetime()),
        format="YYYY-MM-DD HH:mm:ss"
    )
    time_window = (pd.Timestamp(t0), pd.Timestamp(t1))
else:
    max_rel = 0.0
    for _, df0 in loaded.items():
        rel = (df0["timestamp_iso"] - df0["timestamp_iso"].min()).dt.total_seconds()
        max_rel = max(max_rel, float(rel.max() if len(rel) else 0.0))
    r0, r1 = st.slider(
        "Time range (relative seconds from each file start)",
        min_value=0.0, max_value=float(max_rel),
        value=(0.0, float(max_rel)), step=0.1
    )
    time_window = (r0, r1)

# ──────────────────────────────────────────────────────────────────────────────
# Per-file processing
# ──────────────────────────────────────────────────────────────────────────────
coercion_audit_global: Dict[str, Dict[str, int]] = {}

def process_one(df: pd.DataFrame, name: str) -> pd.DataFrame:
    d = df.copy()
    # Filter by selected time window
    if x_mode == "Timestamp":
        t0, t1 = time_window
        d = d.loc[(d["timestamp_iso"] >= t0) & (d["timestamp_iso"] <= t1)]
    else:
        start = d["timestamp_iso"].min()
        d["_rel_s"] = (d["timestamp_iso"] - start).dt.total_seconds()
        r0, r1 = time_window
        d = d.loc[(d["_rel_s"] >= r0) & (d["_rel_s"] <= r1)]
        # Relative seconds to pseudo-datetime for ms tick formatting
        d["_rel_dt"] = pd.to_datetime(0, unit="ms") + pd.to_timedelta(d["_rel_s"], unit="s")

    # Columns we need
    need_cols = set(y_cols)
    if "pressure_bar" in d.columns:
        need_cols.add("pressure_bar")
    if show_relay and "relay_on" in d.columns:
        need_cols.add("relay_on")

    use_cols = ["timestamp_iso"] + sorted(list(need_cols)) + ["source"]
    if x_mode.startswith("Relative"):
        use_cols += ["_rel_s", "_rel_dt"]
    use_cols = [c for c in use_cols if c in d.columns]
    d = d[use_cols].copy()

    # Coerce numeric
    audit = numeric_coercion_audit(d, [c for c in d.columns if c not in ("timestamp_iso", "source", "_rel_dt")])
    coercion_audit_global[name] = audit
    for c in d.columns:
        if c not in ("timestamp_iso", "source", "_rel_dt"):
            if not is_numeric_dtype(d[c]):
                d[c] = pd.to_numeric(d[c], errors="coerce")

    # Resample (numeric-only)
    if resample_rule != "None":
        try:
            numeric_cols = [c for c in d.columns if c not in ("timestamp_iso", "source", "_rel_dt", "_rel_s")
                            and is_numeric_dtype(d[c])]
            if len(numeric_cols) > 0:
                d = (d[["timestamp_iso"] + numeric_cols]
                     .set_index("timestamp_iso")
                     .resample(resample_rule).agg(aggregator)
                     .reset_index())
                d["source"] = name
                if x_mode.startswith("Relative"):
                    # rebuild relative helpers after resample
                    start = d["timestamp_iso"].min()
                    d["_rel_s"] = (d["timestamp_iso"] - start).dt.total_seconds()
                    d["_rel_dt"] = pd.to_datetime(0, unit="ms") + pd.to_timedelta(d["_rel_s"], unit="s")
            else:
                st.warning(f"[{name}] No numeric columns to resample; plotting raw data.")
        except Exception as e:
            st.warning(f"[{name}] Resampling failed ({e}); plotting raw data.")

    # Deadband
    if enable_deadband:
        targets = [c for c in deadband_target if c in d.columns]
        for col in targets:
            if pct_ref == "constant":
                d[col] = apply_deadband(d[col], band_abs=band_abs, band_pct=band_pct,
                                        pct_ref=float(pct_ref_const or 1.0))
            else:
                d[col] = apply_deadband(d[col], band_abs=band_abs, band_pct=band_pct, pct_ref=pct_ref)

    # Ensure relative helpers if needed
    if x_mode.startswith("Relative") and "_rel_dt" not in d.columns:
        start = d["timestamp_iso"].min()
        d["_rel_s"] = (d["timestamp_iso"] - start).dt.total_seconds()
        d["_rel_dt"] = pd.to_datetime(0, unit="ms") + pd.to_timedelta(d["_rel_s"], unit="s")

    return d

processed = []
for name, df0 in loaded.items():
    d = process_one(df0, name)
    if len(d): processed.append(d)

if not processed:
    st.warning("No data left after filters.")
    st.stop()

data = pd.concat(processed, ignore_index=True)

# ──────────────────────────────────────────────────────────────────────────────
# Computed channels
# ──────────────────────────────────────────────────────────────────────────────
if "pressure rolling mean" in computed_channels and "pressure_bar" in data.columns:
    data["pressure_roll_mean"] = data.groupby("source", sort=False)["pressure_bar"] \
        .transform(lambda s: s.rolling(window=int(roll_window), min_periods=1).mean())
    if "pressure_roll_mean" not in y_cols:
        y_cols.append("pressure_roll_mean")

if "pressure derivative (bar/s)" in computed_channels and {"timestamp_iso", "pressure_bar"}.issubset(data.columns):
    def derivative(grp: pd.DataFrame):
        s = grp["pressure_bar"].astype(float)
        dt = grp["timestamp_iso"].diff().dt.total_seconds().replace(0, np.nan)
        return (s.diff() / dt).fillna(0.0)
    data["pressure_dbar_s"] = data.groupby("source", sort=False).apply(derivative).reset_index(level=0, drop=True)
    if "pressure_dbar_s" not in y_cols:
        y_cols.append("pressure_dbar_s")

# ──────────────────────────────────────────────────────────────────────────────
# Pressure unit conversion (bar ⇄ kPa)
# ──────────────────────────────────────────────────────────────────────────────
display_cols_map: Dict[str, str] = {}
pressure_scale = 100.0 if pressure_units == "kPa" else 1.0
pressure_label = "kPa" if pressure_units == "kPa" else "bar"

for col in ["pressure_bar", "pressure_roll_mean"]:
    if col in data.columns:
        if pressure_scale != 1.0:
            disp = col + ("_kpa" if pressure_units == "kPa" else "")
            data[disp] = data[col] * pressure_scale
            display_cols_map[col] = disp
        else:
            display_cols_map[col] = col

y_cols_display: List[str] = [display_cols_map.get(c, c) for c in y_cols]

# ──────────────────────────────────────────────────────────────────────────────
# Downsampling (fair across files)
# ──────────────────────────────────────────────────────────────────────────────
MAX_POINTS = 140_000
if len(data) > MAX_POINTS:
    out = []
    n_files = max(1, len(loaded))
    quota = max(1, MAX_POINTS // n_files)
    for name, grp in data.groupby("source", sort=False):
        step = max(1, len(grp) // quota)
        out.append(grp.iloc[::step, :])
    data = pd.concat(out, ignore_index=True)

# ──────────────────────────────────────────────────────────────────────────────
# Plotly helpers (legend positioning + axis formatting)
# ──────────────────────────────────────────────────────────────────────────────
def apply_legend_position(fig: "go.Figure", where: str):
    """Place legend and extend bottom margin when legend is below."""
    if where == "Below":
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.35,          # moved slightly lower to clear the rangeslider
                xanchor="left",
                x=0.0,
                itemsizing="trace",
                itemwidth=60,
                font=dict(size=11)
            )
        )
        # bottom margin large enough for rangeslider + below-legend
        try:
            m_obj = fig.layout.margin
            if m_obj is not None and hasattr(m_obj, "to_plotly_json"):
                m = m_obj.to_plotly_json() or {}
                l = m.get("l", 10); r = m.get("r", 10); t = m.get("t", 40); b = m.get("b", 80)
            else:
                l = getattr(m_obj, "l", None) or 10
                r = getattr(m_obj, "r", None) or 10
                t = getattr(m_obj, "t", None) or 40
                b = getattr(m_obj, "b", None) or 80
        except Exception:
            l, r, t, b = 10, 10, 40, 80
        fig.update_layout(margin=dict(l=l, r=r, t=t, b=max(120, b)))
    elif where == "Top":
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0))
    else:  # Right
        fig.update_layout(legend=dict(orientation="v", x=1.02, xanchor="left", y=1.0, yanchor="top"))

def xaxis_time_layout():
    """Common x-axis config: HH:MM:SS.mmm labels + zoom-aware grid."""
    # Tick format controls (switch cleanly as you zoom)
    # dtickrange numbers are in milliseconds for 'date' axes
    tfs = [
        dict(dtickrange=[None, 1000], value="%H:%M:%S.%L"),   # < 1s → show ms
        dict(dtickrange=[1000, 60000], value="%H:%M:%S.%L"),  # 1s..60s → keep ms
        dict(dtickrange=[60000, 3600000], value="%H:%M:%S"),  # 1m..1h → seconds
        dict(dtickrange=[3600000, None], value="%H:%M"),      # >1h → minutes
    ]
    return dict(
        type="date",
        showgrid=True,
        gridcolor="rgba(128,128,128,0.25)",
        gridwidth=1,
        tickformat="%H:%M:%S.%L",
        tickformatstops=tfs,
        hoverformat="%H:%M:%S.%L",
        # explicit space for rangeslider so traces won't overlap it
        rangeslider=dict(
            visible=True,
            thickness=0.10,                       # reserve 10% of plotting height
            bgcolor="rgba(120,120,120,0.08)",
            bordercolor="rgba(0,0,0,0)",
        ),
        title="time (HH:MM:SS.mmm)",
    )

# ──────────────────────────────────────────────────────────────────────────────
# Figure builders (overlay & stacked)
# ──────────────────────────────────────────────────────────────────────────────
def add_extrema_markers(fig: "go.Figure", g: pd.DataFrame, xkey: str, y: str, source: str, opacity=0.8):
    if len(g) == 0 or y not in g.columns: return
    s = g[y]; i_min = s.idxmin() if len(s) else None; i_max = s.idxmax() if len(s) else None
    for lab, idx in [("min", i_min), ("max", i_max)]:
        if idx is None or idx not in g.index: continue
        fig.add_trace(go.Scatter(x=[g.loc[idx, xkey]], y=[g.loc[idx, y]], mode="markers+text",
                                 text=[f"{pretty_name(source)} {y} {lab}"], textposition="top center",
                                 marker=dict(size=7, symbol="x"), showlegend=False, opacity=opacity))

def add_relay_shading(fig: "go.Figure", df: pd.DataFrame, xkey: str, color: str, opacity: float):
    if "relay_on" not in df.columns: return
    for _, g in df.groupby("source", sort=False):
        if len(g) == 0: continue
        gg = g[[xkey, "relay_on"]].copy().reset_index(drop=True)
        on = gg["relay_on"].astype(int)
        edges = on.diff().fillna(on.iloc[0]).astype(int)
        starts = list(gg.index[edges == 1])
        if on.iloc[0] == 1: starts = [0] + starts
        ends = list(gg.index[edges == -1])
        if on.iloc[-1] == 1: ends = ends + [len(gg) - 1]
        for s_idx, e_idx in zip(starts, ends):
            fig.add_vrect(x0=gg.loc[s_idx, xkey], x1=gg.loc[e_idx, xkey],
                          fillcolor=color, opacity=opacity, line_width=0)

def x_key(df: pd.DataFrame) -> str:
    """Return the column to use on the X axis (always 'date' type for ms formatting)."""
    return "_rel_dt" if x_mode.startswith("Relative") else "timestamp_iso"

def make_overlay_figure(df: pd.DataFrame) -> "go.Figure":
    fig = go.Figure()
    xk = x_key(df)
    for y in y_cols_display:
        for source, g in df.groupby("source", sort=False):
            fig.add_trace(go.Scatter(x=g[xk], y=g[y], mode="lines",
                                     name=f"{pretty_name(source)} · {y}",
                                     legendgroup=source, line=dict(width=1),
                                     opacity=overlay_opacity))
            if show_extrema: add_extrema_markers(fig, g, xk, y, source)
    if show_relay and "relay_on" in df.columns:
        try:
            y_min = float(min(df[y].min() for y in y_cols_display))
            y_max = float(max(df[y].max() for y in y_cols_display))
            span = max(1e-9, y_max - y_min)
            for source, g in df.groupby("source", sort=False):
                relay_scaled = y_min + g["relay_on"].astype(float) * span * 0.12
                fig.add_trace(go.Scatter(x=g[xk], y=relay_scaled, mode="lines",
                                         name=f"{pretty_name(source)} · relay_on",
                                         legendgroup=source,
                                         line=dict(width=1, shape="hv", dash="dot"),
                                         opacity=min(overlay_opacity, 0.7)))
        except Exception:
            pass
    if shade_relay:
        add_relay_shading(fig, df, xk, color=shade_color, opacity=shade_opacity)

    y_title = ", ".join([
        "pressure ({})".format(pressure_label) if c.startswith("pressure_") else c
        for c in y_cols_display
    ])
    fig.update_layout(
        height=550,
        margin=dict(l=10, r=10, t=40, b=80),
        hovermode="x unified",
        template="plotly" if st.get_option("theme.base") == "light" else "plotly_dark",
        yaxis=dict(title=y_title),
        xaxis=xaxis_time_layout(),  # ms labels + zoom-aware grid + rangeslider
    )
    apply_legend_position(fig, legend_location)
    return fig


def make_stacked_figure(df: pd.DataFrame) -> "go.Figure":
    sources = list(df["source"].unique())
    rows = len(sources)
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.02,
        subplot_titles=[pretty_name(s) for s in sources]
    )
    xk = x_key(df)
    for i, source in enumerate(sources, start=1):
        g = df[df["source"] == source]
        for y in y_cols_display:
            fig.add_trace(go.Scatter(
                x=g[xk], y=g[y], mode="lines",
                name=f"{pretty_name(source)} · {y}",
                line=dict(width=1), opacity=1.0
            ), row=i, col=1)
            if show_extrema:
                add_extrema_markers(fig, g, xk, y, source)
        if show_relay and "relay_on" in g.columns:
            try:
                y_min = float(min(g[y].min() for y in y_cols_display))
                y_max = float(max(g[y].max() for y in y_cols_display))
                span = max(1e-9, y_max - y_min)
                relay_scaled = y_min + g["relay_on"].astype(float) * span * 0.12
                fig.add_trace(go.Scatter(
                    x=g[xk], y=relay_scaled, mode="lines",
                    name=f"{pretty_name(source)} · relay_on",
                    line=dict(width=1, shape="hv", dash="dot"),
                    opacity=0.7
                ), row=i, col=1)
            except Exception:
                pass
        if shade_relay:
            add_relay_shading(fig, g, xk, color=shade_color, opacity=shade_opacity)

    # Base layout for stacked
    fig.update_layout(
        height=max(300 * rows, 300),
        margin=dict(l=10, r=10, t=40, b=90),
        hovermode="x unified",
        showlegend=True,
        template="plotly" if st.get_option("theme.base") == "light" else "plotly_dark",
    )

    # Build two axis configs:
    # 1) Full config WITH rangeslider (bottom subplot)
    xcfg_bottom = xaxis_time_layout()

    # 2) Same config but slider hidden (apply to all upper subplots)
    xcfg_no_slider = xaxis_time_layout()
    xcfg_no_slider["rangeslider"] = dict(visible=False)  # keep vertical grid, hide slider

    # Apply x-axis settings to ALL rows so every subplot gets vertical gridlines
    # Upper rows: grid ON, rangeslider OFF
    for r in range(1, rows):
        fig.update_xaxes(row=r, col=1, **xcfg_no_slider)

    # Bottom row: grid ON, rangeslider ON (owned by bottom axis)
    fig.update_xaxes(row=rows, col=1, **xcfg_bottom)

    apply_legend_position(fig, legend_location)

    # Y axis title on right-most subplot
    fig.update_yaxes(title_text=", ".join([
        "pressure ({})".format(pressure_label) if c.startswith("pressure_") else c
        for c in y_cols_display
    ]), col=1)
    return fig

# ──────────────────────────────────────────────────────────────────────────────
# Draw chart
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("Shart")
if not HAS_PLOTLY:
    st.error("Plotly is not installed. Add `plotly` to requirements.txt and redeploy.")
    st.stop()

fig = make_overlay_figure(data) if (len(loaded) == 1 or display_mode.startswith("Overlay")) else make_stacked_figure(data)
st.plotly_chart(fig, width="stretch", config={"displaylogo": False})

# ──────────────────────────────────────────────────────────────────────────────
# Export (data & image)
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("Export")
st.caption("Export the **currently visible, processed** data, or enter a **custom range** to slice & export.")
st.download_button("Download current window (CSV)", data=data.to_csv(index=False).encode("utf-8"),
                   file_name="current_window.csv", mime="text/csv")

with st.expander("Custom time range export"):
    if x_mode == "Timestamp":
        c_t0 = st.text_input("Start (YYYY-MM-DD HH:MM:SS)", value=str(time_window[0]))
        c_t1 = st.text_input("End (YYYY-MM-DD HH:MM:SS)", value=str(time_window[1]))
        if st.button("Download custom range (CSV)"):
            try:
                t0 = pd.to_datetime(c_t0); t1 = pd.to_datetime(c_t1)
                mask = (data["timestamp_iso"] >= t0) & (data["timestamp_iso"] <= t1)
                sliced = data.loc[mask]
                st.download_button("Click to save (CSV)", data=sliced.to_csv(index=False).encode("utf-8"),
                                   file_name="custom_range.csv", mime="text/csv", key="dl_custom_ts")
            except Exception as e:
                st.error(f"Invalid timestamps: {e}")
    else:
        c_r0 = st.number_input("Start (relative seconds)", value=float(time_window[0]))
        c_r1 = st.number_input("End (relative seconds)", value=float(time_window[1]))
        if st.button("Download custom range (CSV)"):
            try:
                mask = (data["_rel_s"] >= float(c_r0)) & (data["_rel_s"] <= float(c_r1))
                sliced = data.loc[mask]
                st.download_button("Click to save (CSV)", data=sliced.to_csv(index=False).encode("utf-8"),
                                   file_name="custom_range_relative.csv", mime="text/csv", key="dl_custom_rel")
            except Exception as e:
                st.error(f"Invalid relative seconds: {e}")

with st.expander("Download chart as image (PNG / SVG)"):
    # --- Controls ---
    c1, c2, c3 = st.columns([1.2, 1.2, 1])
    with c1:
        png_w = st.number_input("PNG width (px)", min_value=200, max_value=12000, value=1920, step=10)
    with c2:
        png_h = st.number_input("PNG height (px)", min_value=200, max_value=12000, value=1080, step=10)
    with c3:
        png_scale = st.slider("Scale (×)", 0.5, 4.0, 1.5, 0.1, help="Multiplies width & height for extra sharpness")

    c4, c5 = st.columns([1, 1])
    with c4:
        st.caption("Aspect ratio presets")
        pr1, pr2, pr3 = st.columns(3)
        if pr1.button("16:9"):
            st.session_state["_png_ratio"] = (16, 9)
            st.experimental_rerun()
        if pr2.button("4:3"):
            st.session_state["_png_ratio"] = (4, 3)
            st.experimental_rerun()
        if pr3.button("1:1"):
            st.session_state["_png_ratio"] = (1, 1)
            st.experimental_rerun()
    with c5:
        transparent = st.checkbox("Transparent background", value=False)
        bg_color = st.color_picker("Background (PNG)", value="#ffffff", help="Ignored if Transparent is checked")

    # Apply session ratio if set (keeps current width, adjusts height)
    ratio = st.session_state.get("_png_ratio")
    if ratio:
        rw, rh = ratio
        new_h = int(round(png_w * (rh / rw)))
        if new_h != png_h:
            st.session_state["_force_png_h"] = new_h
            st.markdown(f"<script>window.location.reload();</script>", unsafe_allow_html=True)

    # Background handling with a cloned fig (so on-screen fig remains unchanged)
    fig_export = fig.to_dict()
    fig_export = go.Figure(fig_export)
    if transparent:
        fig_export.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    else:
        fig_export.update_layout(paper_bgcolor=bg_color, plot_bgcolor=bg_color)

    # --- PNG / SVG buttons ---
    col_png, col_svg = st.columns(2)
    try:
        png_bytes = fig_export.to_image(format="png", width=png_w, height=png_h, scale=png_scale)
        col_png.download_button("Download PNG", data=png_bytes, file_name="chart.png", mime="image/png")
    except Exception as e:
        col_png.info(f"PNG export unavailable (install/pin kaleido): {e}")

    try:
        svg_bytes = fig_export.to_image(format="svg")  # SVG is resolution-independent
        col_svg.download_button("Download SVG", data=svg_bytes, file_name="chart.svg", mime="image/svg+xml")
    except Exception as e:
        col_svg.info(f"SVG export unavailable (install/pin kaleido): {e}")

    st.caption(f"Output resolution (after scale): **{int(png_w * png_scale)} × {int(png_h * png_scale)} px**")

st.download_button("Download combined filtered data (CSV)",
                   data=data.to_csv(index=False).encode("utf-8"),
                   file_name="combined_filtered.csv", mime="text/csv")

# Optional previews & debug
show_preview = st.session_state.get("_show_preview", False) or False
show_debug = st.session_state.get("_show_debug", False) or False