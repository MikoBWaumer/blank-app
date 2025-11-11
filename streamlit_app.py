# app.py
import io
import os
import hashlib
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from pandas.api.types import is_numeric_dtype

import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
        import pyarrow as pa  # noqa
        import pyarrow.parquet as pq  # noqa
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
    pct_ref: str | float = "previous_output",
) -> pd.Series:
    """
    Hysteresis deadband. Output changes only when |input - last_output| > band,
    where band = max(band_abs, band_pct * reference).

    pct_ref:
      - 'previous_output': % relative to last output magnitude
      - 'range': % relative to (max(series) - min(series))
      - float: constant magnitude used as percent reference
    """
    s = series.astype(float).values
    n = len(s)
    if n == 0:
        return series

    out = np.empty_like(s)
    y = s[0]
    out[0] = y

    rng = np.nanmax(s) - np.nanmin(s) if pct_ref == "range" else None

    for i in range(1, n):
        x = s[i]
        if isinstance(pct_ref, (int, float)):
            ref = float(pct_ref)
        elif pct_ref == "range":
            ref = rng if (rng and rng > 0) else (abs(y) if y != 0 else 1.0)
        else:  # previous_output
            ref = abs(y) if y != 0 else 1.0

        band = max(band_abs, band_pct * ref)
        if abs(x - y) > band:
            y = x  # cross the band → accept & re-center
        out[i] = y

    return pd.Series(out, index=series.index, name=series.name)


def numeric_coercion_audit(df_before: pd.DataFrame, cols: List[str]) -> Dict[str, int]:
    """
    For each column that isn't numeric, estimate how many values were coerced to NaN.
    """
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
# Presets via URL query params (works on recent and older Streamlit)
# ──────────────────────────────────────────────────────────────────────────────
def get_qp() -> Dict[str, List[str]]:
    if hasattr(st, "query_params"):  # Streamlit ≥ 1.30
        return dict(st.query_params)
    return st.experimental_get_query_params()


def set_qp(**kwargs):
    if hasattr(st, "query_params"):  # Streamlit ≥ 1.30
        st.query_params.clear()
        for k, v in kwargs.items():
            st.query_params[k] = v
    else:
        st.experimental_set_query_params(**kwargs)


qp = get_qp()


def qp_get_str(key: str, default: str) -> str:
    v = qp.get(key, [default])
    return v[0] if isinstance(v, list) else v


def qp_get_float(key: str, default: float) -> float:
    v = qp_get_str(key, str(default))
    try:
        return float(v)
    except Exception:
        return default


def qp_get_bool(key: str, default: bool) -> bool:
    v = qp_get_str(key, "1" if default else "0")
    return v in ("1", "true", "True", "yes")


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar controls
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Upload & Options")
    files = st.file_uploader(
        "Upload one or more CSV files",
        type=["csv"],
        accept_multiple_files=True,
    )

    display_mode = st.radio(
        "Display mode",
        ["Overlay (compare on same axes)", "Stacked (one panel per file)"],
        index=0 if qp_get_str("mode", "overlay").lower() == "overlay" else 1,
    )

    x_mode = st.radio(
        "X-axis",
        ["Timestamp", "Relative time (seconds from file start)"],
        index=0 if qp_get_str("x", "ts") == "ts" else 1,
    )

    # Resampling & aggregation
    resample_rule = st.selectbox(
        "Resample (time aggregation)",
        ["None", "1s", "5s", "10s", "30s", "1min", "5min"],
        index=["None", "1s", "5s", "10s", "30s", "1min", "5min"].index(qp_get_str("rs", "5s")),
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
        default=["pressure_bar"],
    )
    band_abs = st.number_input(
        "Absolute band (same units as signal)", value=qp_get_float("db_abs", 0.02), step=0.01, format="%.5f"
    )
    band_pct = st.number_input(
        "Percentage band (%)", value=qp_get_float("db_pct", 0.0), step=0.1, format="%.3f"
    ) / 100.0
    pct_ref = st.selectbox(
        "Percent band reference", ["previous_output", "range", "constant"],
        index=["previous_output", "range", "constant"].index(qp_get_str("db_ref", "previous_output"))
    )
    pct_ref_const = (
        st.number_input("Constant reference magnitude", value=qp_get_float("db_c", 1.0), step=0.1)
        if pct_ref == "constant"
        else None
    )

    st.divider()
    st.subheader("Appearance")
    overlay_opacity = st.slider(
        "Overlay line opacity", 0.1, 1.0, float(qp_get_float("op", 0.6)), 0.05, help="Used in Overlay mode"
    )
    legend_location = st.radio(
        "Legend location",
        ["Below", "Right", "Top"],
        index={"below": 0, "right": 1, "top": 2}.get(qp_get_str("leg", "below"), 0),
        help="Place legend under the plot to maximize chart area",
    )

    st.divider()
    st.subheader("Pressure units")
    pressure_units = st.radio("Units for pressure_bar", ["bar", "kPa"],
                              index=0 if qp_get_str("u", "bar") == "bar" else 1)

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
            rs=resample_rule,
            agg=aggregator,
            relay="1" if show_relay else "0",
            db="1" if enable_deadband else "0",
            db_abs=str(band_abs),
            db_pct=str(band_pct * 100.0),
            db_ref=pct_ref,
            db_c=str(pct_ref_const or 1.0),
            op=str(overlay_opacity),
            leg=legend_location.lower(),
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
def load_one(file_bytes: bytes, name: str) -> pd.DataFrame:
    """
    Load uploaded CSV. If pyarrow is available, persist a parquet file keyed by file content hash
    for faster reloads across sessions. Falls back to parsing CSV directly.
    """
    df = None
    if PARQUET_OK and PARQ_DIR:
        import pyarrow as pa
        import pyarrow.parquet as pq

        h = sha1(file_bytes)
        parq_path = os.path.join(PARQ_DIR, f"{h}.parquet")
        if os.path.exists(parq_path):
            try:
                table = pq.read_table(parq_path)
                df = table.to_pandas()
            except Exception:
                df = None
        if df is None:
            df = pd.read_csv(io.BytesIO(file_bytes), low_memory=False, parse_dates=["timestamp_iso"])
            try:
                table = pa.Table.from_pandas(df)
                pq.write_table(table, parq_path)
            except Exception:
                pass
    else:
        df = pd.read_csv(io.BytesIO(file_bytes), low_memory=False, parse_dates=["timestamp_iso"])

    df["source"] = name
    return df


loaded: Dict[str, pd.DataFrame] = {f.name: load_one(f.getvalue(), f.name) for f in files}

# Numeric intersection across files (only offer columns present everywhere)
numeric_by_file = {k: v.select_dtypes(include="number").columns.tolist() for k, v in loaded.items()}
numeric_intersection = set(numeric_by_file[next(iter(loaded))])
for _, cols in numeric_by_file.items():
    numeric_intersection &= set(cols)
numeric_intersection = sorted(numeric_intersection)

# Prefer common order
ordered_numeric = [
    c for c in ["pressure_bar", "voltage_v", "current_mA", "moisture2_v", "moisture3_v", "relay_on"]
    if c in numeric_intersection
]
for c in numeric_intersection:
    if c not in ordered_numeric:
        ordered_numeric.append(c)

# Y columns to plot (we'll extend with computed channels later)
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
        format="YYYY-MM-DD HH:mm:ss",
    )
    time_window = (pd.Timestamp(t0), pd.Timestamp(t1))
else:
    max_rel = 0.0
    for _, df0 in loaded.items():
        rel = (df0["timestamp_iso"] - df0["timestamp_iso"].min()).dt.total_seconds()
        max_rel = max(max_rel, float(rel.max() if len(rel) else 0.0))
    r0, r1 = st.slider(
        "Time range (relative seconds from each file start)",
        min_value=0.0,
        max_value=float(max_rel),
        value=(0.0, float(max_rel)),
        step=0.1,
    )
    time_window = (r0, r1)  # seconds


# ──────────────────────────────────────────────────────────────────────────────
# Per-file processing
# ──────────────────────────────────────────────────────────────────────────────
coercion_audit_global: Dict[str, Dict[str, int]] = {}  # {source: {col: count_coerced}}

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

    # Columns we need:
    need_cols = set(y_cols)
    if show_relay and "relay_on" in d.columns:
        need_cols.add("relay_on")
    if "pressure_bar" in d.columns:  # for computed channels baseline
        need_cols.add("pressure_bar")

    use_cols = ["timestamp_iso"] + sorted(list(need_cols)) + ["source"]
    use_cols = [c for c in use_cols if c in d.columns]
    d = d[use_cols].copy()

    # Audit & coerce numeric (before resampling)
    audit = numeric_coercion_audit(d, [c for c in d.columns if c not in ("timestamp_iso", "source")])
    coercion_audit_global[name] = audit
    for c in d.columns:
        if c not in ("timestamp_iso", "source"):
            if not is_numeric_dtype(d[c]):
                d[c] = pd.to_numeric(d[c], errors="coerce")

    # Safe resample: numeric-only aggregation, then re-attach "source"
    if resample_rule != "None":
        try:
            numeric_cols = [
                c for c in d.columns if c not in ("timestamp_iso", "source") and is_numeric_dtype(d[c])
            ]
            if len(numeric_cols) > 0:
                d = (
                    d[["timestamp_iso"] + numeric_cols]
                    .set_index("timestamp_iso")
                    .resample(resample_rule)
                    .agg(aggregator)
                    .reset_index()
                )
                d["source"] = name
            else:
                st.warning(f"[{name}] No numeric columns to resample; plotting raw data.")
        except Exception as e:
            st.warning(f"[{name}] Resampling failed ({e}); plotting raw data.")

    # Deadband (after resample → cleaner & faster)
    if enable_deadband:
        targets = [c for c in deadband_target if c in d.columns]
        for col in targets:
            if pct_ref == "constant":
                d[col] = apply_deadband(
                    d[col], band_abs=band_abs, band_pct=band_pct, pct_ref=float(pct_ref_const or 1.0)
                )
            else:
                d[col] = apply_deadband(d[col], band_abs=band_abs, band_pct=band_pct, pct_ref=pct_ref)

    # Ensure relative key if needed
    if x_mode.startswith("Relative") and "_rel_s" not in d.columns:
        start = d["timestamp_iso"].min()
        d["_rel_s"] = (d["timestamp_iso"] - start).dt.total_seconds()

    return d


processed = []
for name, df0 in loaded.items():
    d = process_one(df0, name)
    if len(d):
        processed.append(d)

if not processed:
    st.warning("No data left after filters.")
    st.stop()

# Combine processed frames
data = pd.concat(processed, ignore_index=True)

# ──────────────────────────────────────────────────────────────────────────────
# Computed channels (after processing so they match what you visualize)
# ──────────────────────────────────────────────────────────────────────────────
if "pressure rolling mean" in computed_channels and "pressure_bar" in data.columns:
    data["pressure_roll_mean"] = data.groupby("source", sort=False)["pressure_bar"]\
                                     .transform(lambda s: s.rolling(window=int(roll_window), min_periods=1).mean())
    if "pressure_roll_mean" not in y_cols:
        y_cols.append("pressure_roll_mean")

if "pressure derivative (bar/s)" in computed_channels and "timestamp_iso" in data.columns and "pressure_bar" in data.columns:
    def derivative(grp: pd.DataFrame):
        s = grp["pressure_bar"].astype(float)
        dt = grp["timestamp_iso"].diff().dt.total_seconds().replace(0, np.nan)
        return (s.diff() / dt).fillna(0.0)
    data["pressure_dbar_s"] = (
        data.groupby("source", sort=False)
            .apply(derivative)
            .reset_index(level=0, drop=True)
    )
    if "pressure_dbar_s" not in y_cols:
        y_cols.append("pressure_dbar_s")

# ──────────────────────────────────────────────────────────────────────────────
# Pressure unit conversion (bar ⇄ kPa)
# ──────────────────────────────────────────────────────────────────────────────
display_cols_map: Dict[str, str] = {}  # original -> display
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

# Build list of actual columns to plot (replace pressure_* with converted variants)
y_cols_display: List[str] = [display_cols_map.get(c, c) for c in y_cols]

# ──────────────────────────────────────────────────────────────────────────────
# Downsampling for responsiveness (fair per-file)
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
# Plotting helpers (includes robust legend-margin handling)
# ──────────────────────────────────────────────────────────────────────────────
def apply_legend_position(fig: go.Figure, where: str):
    """
    Configure legend placement. Safely reads current margins (Margin object or dict)
    and extends bottom margin when the legend is below.
    """
    # Place legend
    if where == "Below":
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.18,  # push under the x-axis; tweak -0.12 … -0.35 if needed
                xanchor="left",
                x=0.0,
                itemsizing="trace",
                itemwidth=60,
                font=dict(size=11),
            )
        )
        # Safely extend bottom margin
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

        fig.update_layout(margin=dict(l=l, r=r, t=t, b=max(80, b)))

    elif where == "Top":
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0.0,
            )
        )
    else:  # Right
        fig.update_layout(
            legend=dict(
                orientation="v",
                x=1.02, xanchor="left",
                y=1.0,  yanchor="top"
            )
        )


def add_extrema_markers(fig: go.Figure, g: pd.DataFrame, xkey: str, y: str, source: str, opacity=0.8):
    if len(g) == 0 or y not in g.columns:
        return
    s = g[y]
    i_min = s.idxmin() if len(s) else None
    i_max = s.idxmax() if len(s) else None
    for lab, idx in [("min", i_min), ("max", i_max)]:
        if idx is None or idx not in g.index:
            continue
        fig.add_trace(
            go.Scatter(
                x=[g.loc[idx, xkey]],
                y=[g.loc[idx, y]],
                mode="markers+text",
                text=[f"{pretty_name(source)} {y} {lab}"],
                textposition="top center",
                marker=dict(size=7, symbol="x"),
                showlegend=False,
                opacity=opacity,
            )
        )


def add_relay_shading(fig: go.Figure, df: pd.DataFrame, xkey: str, color: str, opacity: float):
    """Add vrects for contiguous segments where relay_on == 1 (per source)."""
    if "relay_on" not in df.columns:
        return
    for _, g in df.groupby("source", sort=False):
        if len(g) == 0:
            continue
        gg = g[[xkey, "relay_on"]].copy().reset_index(drop=True)
        on = gg["relay_on"].astype(int)
        edges = on.diff().fillna(on.iloc[0]).astype(int)
        starts = list(gg.index[edges == 1])
        if on.iloc[0] == 1:
            starts = [0] + starts
        ends = list(gg.index[edges == -1])
        if on.iloc[-1] == 1:
            ends = ends + [len(gg) - 1]
        for s_idx, e_idx in zip(starts, ends):
            x0 = gg.loc[s_idx, xkey]
            x1 = gg.loc[e_idx, xkey]
            fig.add_vrect(x0=x0, x1=x1, fillcolor=color, opacity=opacity, line_width=0)


def make_overlay_figure(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    xkey = "_rel_s" if x_mode.startswith("Relative") else "timestamp_iso"

    # Plot signals
    for y in y_cols_display:
        for source, g in df.groupby("source", sort=False):
            fig.add_trace(
                go.Scatter(
                    x=g[xkey],
                    y=g[y],
                    mode="lines",
                    name=f"{pretty_name(source)} · {y}",
                    legendgroup=source,
                    line=dict(width=1),
                    opacity=overlay_opacity,
                )
            )
            if show_extrema:
                add_extrema_markers(fig, g, xkey, y, source)

    # Optional relay overlay (scaled to ~12% chart height)
    if show_relay and "relay_on" in df.columns:
        try:
            y_min = float(min(df[y].min() for y in y_cols_display))
            y_max = float(max(df[y].max() for y in y_cols_display))
            span = max(1e-9, y_max - y_min)
            for source, g in df.groupby("source", sort=False):
                relay_scaled = y_min + g["relay_on"].astype(float) * span * 0.12
                fig.add_trace(
                    go.Scatter(
                        x=g[xkey],
                        y=relay_scaled,
                        mode="lines",
                        name=f"{pretty_name(source)} · relay_on",
                        legendgroup=source,
                        line=dict(width=1, shape="hv", dash="dot"),
                        opacity=min(overlay_opacity, 0.7),
                    )
                )
        except Exception:
            pass

    # Optional shading for relay_on
    if shade_relay:
        add_relay_shading(fig, df, xkey, color=shade_color, opacity=shade_opacity)

    # Axis titles (convert pressure labels)
    y_title = ", ".join(
        ["pressure ({})".format(pressure_label) if c.startswith("pressure_") else c for c in y_cols_display]
    )

    fig.update_layout(
        height=550,
        margin=dict(l=10, r=10, t=40, b=80),
        xaxis=dict(
            title="time (s)" if x_mode.startswith("Relative") else "timestamp",
            rangeslider=dict(visible=not x_mode.startswith("Relative")),
        ),
        yaxis=dict(title=y_title),
        hovermode="x unified",
        template="plotly" if st.get_option("theme.base") == "light" else "plotly_dark",
    )
    apply_legend_position(fig, legend_location)
    return fig


def make_stacked_figure(df: pd.DataFrame) -> go.Figure:
    sources = list(df["source"].unique())
    rows = len(sources)
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.02,
        subplot_titles=[pretty_name(s) for s in sources]
    )
    xkey = "_rel_s" if x_mode.startswith("Relative") else "timestamp_iso"

    for i, source in enumerate(sources, start=1):
        g = df[df["source"] == source]
        for y in y_cols_display:
            fig.add_trace(
                go.Scatter(
                    x=g[xkey],
                    y=g[y],
                    mode="lines",
                    name=f"{pretty_name(source)} · {y}",
                    line=dict(width=1),
                    opacity=1.0,
                ),
                row=i, col=1,
            )
            if show_extrema:
                add_extrema_markers(fig, g, xkey, y, source)
        if show_relay and "relay_on" in g.columns:
            try:
                y_min = float(min(g[y].min() for y in y_cols_display))
                y_max = float(max(g[y].max() for y in y_cols_display))
                span = max(1e-9, y_max - y_min)
                relay_scaled = y_min + g["relay_on"].astype(float) * span * 0.12
                fig.add_trace(
                    go.Scatter(
                        x=g[xkey],
                        y=relay_scaled,
                        mode="lines",
                        name=f"{pretty_name(source)} · relay_on",
                        line=dict(width=1, shape="hv", dash="dot"),
                        opacity=0.7,
                    ),
                    row=i, col=1,
                )
            except Exception:
                pass
        if shade_relay:
            add_relay_shading(fig, g, xkey, color=shade_color, opacity=shade_opacity)

    fig.update_layout(
        height=max(260 * rows, 260),
        margin=dict(l=10, r=10, t=40, b=90),
        hovermode="x unified",
        showlegend=True,
        template="plotly" if st.get_option("theme.base") == "light" else "plotly_dark",
    )
    apply_legend_position(fig, legend_location)

    fig.update_xaxes(title_text="time (s)" if x_mode.startswith("Relative") else "timestamp", row=rows, col=1)
    fig.update_yaxes(
        title_text=", ".join(
            ["pressure ({})".format(pressure_label) if c.startswith("pressure_") else c for c in y_cols_display]
        ),
        col=1
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Draw chart
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("Chart")
if len(loaded) == 1 or display_mode.startswith("Overlay"):
    fig = make_overlay_figure(data)
else:
    fig = make_stacked_figure(data)

st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


# ──────────────────────────────────────────────────────────────────────────────
# Export current window / custom range (CSV)
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("Export")
st.caption("Export the **currently visible, processed** data, or enter a **custom range** to slice & export.")

# Current window export (processed + filtered + resampled + deadbanded data)
st.download_button(
    "Download current window (CSV)",
    data=data.to_csv(index=False).encode("utf-8"),
    file_name="current_window.csv",
    mime="text/csv",
)

# Custom range export
with st.expander("Custom time range export"):
    if x_mode == "Timestamp":
        c_t0 = st.text_input("Start (YYYY-MM-DD HH:MM:SS)", value=str(time_window[0]))
        c_t1 = st.text_input("End   (YYYY-MM-DD HH:MM:SS)", value=str(time_window[1]))
        if st.button("Download custom range (CSV)"):
            try:
                t0 = pd.to_datetime(c_t0); t1 = pd.to_datetime(c_t1)
                mask = (data["timestamp_iso"] >= t0) & (data["timestamp_iso"] <= t1)
                sliced = data.loc[mask]
                st.download_button(
                    "Click to save (CSV)",
                    data=sliced.to_csv(index=False).encode("utf-8"),
                    file_name="custom_range.csv", mime="text/csv", key="dl_custom_ts"
                )
            except Exception as e:
                st.error(f"Invalid timestamps: {e}")
    else:
        c_r0 = st.number_input("Start (relative seconds)", value=float(time_window[0]))
        c_r1 = st.number_input("End   (relative seconds)", value=float(time_window[1]))
        if st.button("Download custom range (CSV)"):
            try:
                mask = (data["_rel_s"] >= float(c_r0)) & (data["_rel_s"] <= float(c_r1))
                sliced = data.loc[mask]
                st.download_button(
                    "Click to save (CSV)",
                    data=sliced.to_csv(index=False).encode("utf-8"),
                    file_name="custom_range_relative.csv", mime="text/csv", key="dl_custom_rel"
                )
            except Exception as e:
                st.error(f"Invalid relative seconds: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Download chart as PNG / SVG (requires kaleido)
# ──────────────────────────────────────────────────────────────────────────────
with st.expander("Download chart as image (PNG / SVG)"):
    col_png, col_svg = st.columns(2)
    with col_png:
        try:
            png_bytes = fig.to_image(format="png", scale=2)
            st.download_button("Download PNG", data=png_bytes, file_name="chart.png", mime="image/png")
        except Exception:
            st.info("PNG export requires **kaleido**. Install with: `pip install -U kaleido`")
    with col_svg:
        try:
            svg_bytes = fig.to_image(format="svg")
            st.download_button("Download SVG", data=svg_bytes, file_name="chart.svg", mime="image/svg+xml")
        except Exception:
            st.info("SVG export requires **kaleido**. Install with: `pip install -U kaleido`")


# ──────────────────────────────────────────────────────────────────────────────
# Download combined filtered data (legacy button)
# ──────────────────────────────────────────────────────────────────────────────
st.download_button(
    "Download combined filtered data (CSV)",
    data=data.to_csv(index=False).encode("utf-8"),
    file_name="combined_filtered.csv",
    mime="text/csv",
)


# ──────────────────────────────────────────────────────────────────────────────
# Optional previews & debug — includes numeric coercion audit
# ──────────────────────────────────────────────────────────────────────────────
if show_preview:
    st.subheader("Per-file previews")
    for name, df0 in loaded.items():
        st.caption(f"{name}: {len(df0):,} rows")
        st.dataframe(df0.head(10), use_container_width=True)

if show_debug:
    st.subheader("Debug / Diagnostics")
    st.write(
        {
            "files": list(loaded.keys()),
            "rows_combined_after_processing": int(data.shape[0]),
            "y_cols_plot": y_cols_display,
            "display_mode": display_mode,
            "overlay_opacity": overlay_opacity,
            "legend_location": legend_location,
            "x_mode": x_mode,
            "resample_rule": resample_rule,
            "deadband": {
                "enabled": enable_deadband,
                "targets": deadband_target,
                "band_abs": band_abs,
                "band_pct": band_pct,
                "pct_ref": pct_ref,
            },
            "pressure_units": pressure_units,
            "computed_channels": computed_channels,
            "parquet_cache": {"enabled": PARQUET_OK, "dir": PARQ_DIR},
        }
    )
    st.write("Numeric coercion audit (values coerced to NaN per file/column):")
    st.json(coercion_audit_global, expanded=False)
