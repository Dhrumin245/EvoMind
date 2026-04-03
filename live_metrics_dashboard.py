"""Live Plotly dashboard for EvoMind training metrics.

This dashboard watches a metrics CSV file (supports both metrics.csv and
metrices.csv naming) and renders live-updating graphs while training runs.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import logging
import time
import urllib.error
import urllib.request
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from dash import Dash, Input, Output, State, ctx, dcc, html
import matplotlib.cm as cm
from matplotlib import colors as mcolors
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class VisualTheme(TypedDict):
    line_colors: List[str]
    accent_warm: str
    accent_cool: str
    surface: str
    card: str
    grid: str
    text: str
    muted: str


def _load_seaborn() -> Optional[Any]:
    try:
        return importlib.import_module("seaborn")
    except ImportError:
        return None


def _build_visual_theme() -> VisualTheme:
    sns = _load_seaborn()
    if sns is not None:
        sns.set_theme(style="whitegrid", context="notebook")
        seaborn_palette = sns.color_palette("crest", 10)
        line_colors = [mcolors.to_hex(color) for color in seaborn_palette]
    else:
        viridis = cm.get_cmap("viridis")
        line_colors = [mcolors.to_hex(viridis(i / 9.0)) for i in range(10)]

    plasma = cm.get_cmap("plasma")
    accent_warm = mcolors.to_hex(plasma(0.70))
    accent_cool = mcolors.to_hex(plasma(0.30))

    return {
        "line_colors": line_colors,
        "accent_warm": accent_warm,
        "accent_cool": accent_cool,
        "surface": "#f7fbfc",
        "card": "#ffffff",
        "grid": "#d9e4ec",
        "text": "#173042",
        "muted": "#4f6474",
    }


def _discover_metrics_file(user_path: Optional[str]) -> Path:
    if user_path:
        return Path(user_path)

    candidates = [
        Path("data/metrics.csv"),
        Path("data/metrices.csv"),
        Path("metrics.csv"),
        Path("metrices.csv"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path("data/metrices.csv")


def _try_parse_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _read_numeric_rows(metrics_path: Path) -> Tuple[List[Dict[str, float]], List[str], Optional[str]]:
    if not metrics_path.exists():
        return [], [], f"File not found: {metrics_path}"

    # File gets atomically replaced while training; small retries avoid transient read errors.
    last_error: Optional[str] = None
    for _ in range(3):
        try:
            with metrics_path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                raw_rows = list(reader)
            break
        except OSError as exc:
            last_error = str(exc)
            time.sleep(0.05)
    else:
        return [], [], f"Unable to read {metrics_path}: {last_error or 'unknown error'}"

    numeric_rows: List[Dict[str, float]] = []
    numeric_keys = set()

    for row in raw_rows:
        parsed: Dict[str, float] = {}
        for key, value in row.items():
            maybe = _try_parse_float(value)
            if maybe is not None:
                parsed[key] = maybe
                numeric_keys.add(key)
        if parsed:
            numeric_rows.append(parsed)

    numeric_cols = sorted(numeric_keys)
    return numeric_rows, numeric_cols, None


def _fetch_train_status(api_base_url: str, timeout_seconds: float = 1.5) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    base = api_base_url.rstrip("/")
    url = f"{base}/train/status"

    try:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
            if response.status != 200:
                return None, f"API status error HTTP {response.status}"
            payload = response.read().decode("utf-8")
            return json.loads(payload), None
    except urllib.error.HTTPError as exc:
        return None, f"API status error HTTP {exc.code}"
    except urllib.error.URLError as exc:
        return None, f"API status unavailable: {exc.reason}"
    except Exception as exc:
        return None, f"API status unavailable: {exc}"


def _x_axis(rows: List[Dict[str, float]]) -> List[float]:
    if rows and "generation" in rows[0]:
        return [row.get("generation", idx) for idx, row in enumerate(rows)]
    return [float(idx) for idx in range(len(rows))]


def _pick_default_metrics(columns: List[str], limit: int = 4) -> List[str]:
    if not columns:
        return []

    priority_patterns = [
        "best_prey_fitness",
        "best_predator_fitness",
        "avg_prey_fitness",
        "avg_predator_fitness",
        "fitness",
        "reward",
        "survival",
        "capture",
        "diversity",
        "plastic",
    ]

    selected: List[str] = []
    lowered = {c: c.lower() for c in columns}

    for pattern in priority_patterns:
        for col in columns:
            if col == "generation" or col in selected:
                continue
            if pattern in lowered[col]:
                selected.append(col)
                if len(selected) >= limit:
                    return selected

    for col in columns:
        if col == "generation" or col in selected:
            continue
        selected.append(col)
        if len(selected) >= limit:
            break
    return selected


def _pick_metrics_by_keywords(columns: List[str], keywords: List[str], limit: int = 12) -> List[str]:
    if not columns:
        return []

    selected: List[str] = []
    lowered = {c: c.lower() for c in columns}

    for keyword in keywords:
        for col in columns:
            if col == "generation" or col in selected:
                continue
            if keyword in lowered[col]:
                selected.append(col)
                if len(selected) >= limit:
                    return selected

    return selected


def _metrics_for_preset(columns: List[str], preset: str) -> List[str]:
    if preset == "fitness":
        return _pick_metrics_by_keywords(
            columns,
            keywords=["fitness", "reward", "survival", "capture"],
            limit=12,
        )
    if preset == "diversity":
        return _pick_metrics_by_keywords(
            columns,
            keywords=["diversity", "novelty", "entropy", "cluster"],
            limit=12,
        )
    if preset == "plasticity":
        return _pick_metrics_by_keywords(
            columns,
            keywords=["plastic", "learning", "delta_w", "adapt"],
            limit=12,
        )
    return []


def _build_overview_figure(
    rows: List[Dict[str, float]],
    columns: List[str],
    max_points: int,
    theme: VisualTheme,
) -> go.Figure:
    metrics = _pick_default_metrics(columns, limit=4)
    subplot_titles = tuple(metrics + [f"Metric {i}" for i in range(len(metrics) + 1, 5)])
    fig = make_subplots(rows=2, cols=2, subplot_titles=subplot_titles)
    fig.update_layout(
        height=720,
        title="Live Training Metrics Overview",
        template="plotly_white",
        paper_bgcolor=theme["surface"],
        plot_bgcolor=theme["card"],
        font={"family": "Avenir Next, Segoe UI, sans-serif", "color": theme["text"]},
        title_font={"size": 24, "color": theme["text"]},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.03, "x": 0.01},
        margin={"l": 50, "r": 30, "t": 80, "b": 40},
    )

    if not rows or not metrics:
        fig.update_layout(title="Live Training Metrics Overview (waiting for data)")
        return fig

    sliced_rows = rows[-max_points:] if max_points > 0 else rows
    x = _x_axis(sliced_rows)

    for i, metric in enumerate(metrics):
        r = (i // 2) + 1
        c = (i % 2) + 1
        y = [row.get(metric) for row in sliced_rows]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=metric,
                line={"width": 3, "color": theme["line_colors"][i % len(theme["line_colors"])]},
            ),
            row=r,
            col=c,
        )
        fig.update_xaxes(title_text="Generation", row=r, col=c)
        fig.update_yaxes(title_text=metric, row=r, col=c)

    fig.update_xaxes(showgrid=True, gridcolor=theme["grid"], zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=theme["grid"], zeroline=False)

    return fig


def _build_selected_figure(
    rows: List[Dict[str, float]],
    selected_metrics: List[str],
    max_points: int,
    theme: VisualTheme,
) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        height=520,
        title="Selected Metrics",
        template="plotly_white",
        paper_bgcolor=theme["surface"],
        plot_bgcolor=theme["card"],
        font={"family": "Avenir Next, Segoe UI, sans-serif", "color": theme["text"]},
        title_font={"size": 22, "color": theme["text"]},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.01},
        margin={"l": 50, "r": 30, "t": 80, "b": 40},
        hovermode="x unified",
    )

    if not rows or not selected_metrics:
        fig.update_layout(title="Selected Metrics (choose one or more metrics)")
        return fig

    sliced_rows = rows[-max_points:] if max_points > 0 else rows
    x = _x_axis(sliced_rows)

    for i, metric in enumerate(selected_metrics):
        y = [row.get(metric) for row in sliced_rows]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=metric,
                line={"width": 3, "color": theme["line_colors"][i % len(theme["line_colors"])]},
            )
        )

    fig.update_xaxes(title="Generation")
    fig.update_yaxes(title="Value")
    fig.update_xaxes(showgrid=True, gridcolor=theme["grid"], zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=theme["grid"], zeroline=False)
    return fig


def create_app(metrics_path: Path, refresh_seconds: float, max_points: int, api_base_url: str) -> Dash:
    app = Dash(__name__)
    theme = _build_visual_theme()

    app.layout = html.Div(
        style={
            "fontFamily": "Avenir Next, Segoe UI, sans-serif",
            "maxWidth": "1240px",
            "margin": "0 auto",
            "padding": "20px",
            "color": theme["text"],
            "background": f"linear-gradient(135deg, {theme['surface']} 0%, #eef5fa 55%, #f8fbff 100%)",
            "minHeight": "100vh",
        },
        children=[
            html.Div(
                style={
                    "background": theme["card"],
                    "border": f"1px solid {theme['grid']}",
                    "borderRadius": "16px",
                    "padding": "20px",
                    "boxShadow": "0 12px 28px rgba(10, 34, 55, 0.08)",
                    "marginBottom": "18px",
                },
                children=[
                    html.H2(
                        "EvoMind Live Training Dashboard",
                        style={"margin": "0 0 8px", "fontWeight": "700", "letterSpacing": "0.2px"},
                    ),
                    html.Div(
                        "Seaborn + Matplotlib themed real-time training telemetry",
                        style={"color": theme["muted"], "marginBottom": "10px"},
                    ),
                    html.Div(
                        id="status-text",
                        style={
                            "padding": "10px 12px",
                            "borderRadius": "10px",
                            "background": f"linear-gradient(90deg, {theme['accent_cool']}1A 0%, {theme['accent_warm']}22 100%)",
                            "border": f"1px solid {theme['grid']}",
                            "fontSize": "14px",
                        },
                    ),
                ],
            ),
            html.Div(
                style={
                    "background": theme["card"],
                    "border": f"1px solid {theme['grid']}",
                    "borderRadius": "16px",
                    "padding": "16px",
                    "boxShadow": "0 8px 20px rgba(10, 34, 55, 0.06)",
                    "marginBottom": "18px",
                },
                children=[
                    html.Label("Metric preset:", style={"fontWeight": "600"}),
                    dcc.RadioItems(
                        id="preset-selector",
                        options=[
                            {"label": " Fitness-only", "value": "fitness"},
                            {"label": " Diversity-only", "value": "diversity"},
                            {"label": " Plasticity-only", "value": "plasticity"},
                            {"label": " Custom", "value": "custom"},
                        ],
                        value="fitness",
                        inline=True,
                        style={"margin": "8px 0 12px", "color": theme["muted"]},
                        inputStyle={"marginRight": "6px"},
                    ),
                    html.Label("Select metrics to overlay:", style={"fontWeight": "600"}),
                    dcc.Dropdown(id="metric-selector", multi=True, placeholder="Choose metrics..."),
                ],
            ),
            html.Div(
                style={
                    "background": theme["card"],
                    "border": f"1px solid {theme['grid']}",
                    "borderRadius": "16px",
                    "padding": "8px",
                    "boxShadow": "0 8px 20px rgba(10, 34, 55, 0.06)",
                    "marginBottom": "18px",
                },
                children=[dcc.Graph(id="overview-graph")],
            ),
            html.Div(
                style={
                    "background": theme["card"],
                    "border": f"1px solid {theme['grid']}",
                    "borderRadius": "16px",
                    "padding": "8px",
                    "boxShadow": "0 8px 20px rgba(10, 34, 55, 0.06)",
                },
                children=[dcc.Graph(id="selected-graph")],
            ),
            dcc.Interval(id="refresh-interval", interval=max(int(refresh_seconds * 1000), 250), n_intervals=0),
            dcc.Store(id="metrics-path", data=str(metrics_path)),
        ],
    )

    @app.callback(
        Output("status-text", "children"),
        Output("metric-selector", "options"),
        Output("metric-selector", "value"),
        Output("preset-selector", "value"),
        Output("overview-graph", "figure"),
        Output("selected-graph", "figure"),
        Input("refresh-interval", "n_intervals"),
        Input("preset-selector", "value"),
        Input("metric-selector", "value"),
        State("metrics-path", "data"),
    )
    def _update_dashboard(_ticks: int, preset: Optional[str], selected: Optional[List[str]], path_text: str):
        path = Path(path_text)
        rows, columns, err = _read_numeric_rows(path)
        api_status, api_err = _fetch_train_status(api_base_url)
        active_preset = preset if preset in {"fitness", "diversity", "plasticity", "custom"} else "custom"

        options = [{"label": col, "value": col} for col in columns if col != "generation"]

        if err is not None:
            if api_err is not None:
                api_text = api_err
            else:
                api_text = (
                    f"API status={api_status.get('status', 'unknown')} "
                    f"gen={api_status.get('generation', 'n/a')}"
                )

            status = f"Watching: {path} | {err} | {api_text}"
            return (
                status,
                options,
                selected or [],
                active_preset,
                _build_overview_figure([], [], max_points, theme),
                _build_selected_figure([], [], max_points, theme),
            )

        available_metrics = [col for col in columns if col != "generation"]
        if active_preset == "custom":
            if selected:
                normalized_selection = [m for m in selected if m in available_metrics]
            else:
                normalized_selection = _pick_default_metrics(available_metrics, limit=4)
        else:
            preset_selection = _metrics_for_preset(available_metrics, active_preset)
            if not preset_selection:
                preset_selection = _pick_default_metrics(available_metrics, limit=4)

            # If user manually edits dropdown while a preset is active, switch to custom.
            if ctx.triggered_id == "metric-selector":
                normalized_selection = [m for m in (selected or []) if m in available_metrics]
                if not normalized_selection:
                    normalized_selection = preset_selection
                active_preset = "custom"
            else:
                normalized_selection = preset_selection

        latest_gen = int(rows[-1].get("generation", len(rows) - 1)) if rows else -1
        if api_err is not None:
            api_text = api_err
        else:
            api_text = (
                f"API status={api_status.get('status', 'unknown')} "
                f"gen={api_status.get('generation', 'n/a')} "
                f"stage={api_status.get('curriculum_stage', 'unknown')}"
            )

        status = (
            f"Watching: {path} | rows={len(rows)} | latest_generation={latest_gen} "
            f"| preset={active_preset} | refresh={refresh_seconds:.2f}s | {api_text}"
        )

        overview = _build_overview_figure(rows, columns, max_points, theme)
        selected_fig = _build_selected_figure(rows, normalized_selection, max_points, theme)

        return status, options, normalized_selection, active_preset, overview, selected_fig

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live Plotly dashboard for EvoMind training metrics.")
    parser.add_argument(
        "--metrics-path",
        type=str,
        default=None,
        help="Path to metrics CSV (defaults to auto-detect: data/metrics.csv or data/metrices.csv).",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Dash server host.")
    parser.add_argument("--port", type=int, default=8050, help="Dash server port.")
    parser.add_argument(
        "--refresh-seconds",
        type=float,
        default=2.0,
        help="Dashboard polling interval in seconds.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=500,
        help="Maximum latest points per graph (<=0 means all points).",
    )
    parser.add_argument(
        "--api-base-url",
        type=str,
        default="http://127.0.0.1:8000",
        help="EvoMind API base URL used for GET /train/status polling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_path = _discover_metrics_file(args.metrics_path)

    # Silence Flask/Werkzeug request logs (GET/POST lines) in terminal output.
    werkzeug_logger = logging.getLogger("werkzeug")
    werkzeug_logger.setLevel(logging.ERROR)
    werkzeug_logger.disabled = True
    logging.getLogger("dash").setLevel(logging.ERROR)

    app = create_app(
        metrics_path=metrics_path,
        refresh_seconds=args.refresh_seconds,
        max_points=args.max_points,
        api_base_url=args.api_base_url,
    )
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
