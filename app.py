#!/usr/bin/env python3
"""Streamlit dashboard for the CBB betting algorithm."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Iterable, List

import pandas as pd
import streamlit as st

from cbb_betting_algorithm import run_analysis

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:  # pragma: no cover - optional dependency fallback
    st_autorefresh = None


DEFAULT_BOOK_OPTIONS = [
    (68, "DraftKings"),
    (69, "FanDuel"),
    (71, "BetRivers"),
    (75, "BetMGM"),
    (79, "bet365"),
]


def flatten_dict(data: Dict[str, Any], prefix: str = "", max_depth: int = 2) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for key, value in data.items():
        next_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict) and max_depth > 0:
            output.update(flatten_dict(value, next_key, max_depth=max_depth - 1))
        elif isinstance(value, list):
            if not value:
                output[next_key] = []
            elif isinstance(value[0], dict):
                output[next_key] = json.dumps(value)
            else:
                output[next_key] = ", ".join(str(x) for x in value)
        else:
            output[next_key] = value
    return output


def records_to_frame(
    records: Iterable[Dict[str, Any]],
    *,
    max_depth: int = 2,
) -> pd.DataFrame:
    normalized = [flatten_dict(record, max_depth=max_depth) for record in records]
    if not normalized:
        return pd.DataFrame()
    return pd.DataFrame(normalized)


def build_engine_args(config: Dict[str, Any]) -> argparse.Namespace:
    return argparse.Namespace(
        league=config["league"],
        public_threshold=config["public_threshold"],
        public_metric=config["public_metric"],
        book_ids=config["book_ids"],
        max_workers=config["max_workers"],
        home_court_advantage=config["home_court_advantage"],
        alt_target_low=config["alt_target_low"],
        alt_target_high=config["alt_target_high"],
        alt_target_mid=config["alt_target_mid"],
        output="",
        watch=False,
        interval_seconds=0,
        max_iterations=0,
        latest_output="",
        archive_dir="",
    )


@st.cache_data(show_spinner=False)
def load_report_cached(config_json: str, refresh_key: int) -> Dict[str, Any]:
    del refresh_key  # cache key input only
    config = json.loads(config_json)
    args = build_engine_args(config)
    return run_analysis(args)


def filter_matchup(records: List[Dict[str, Any]], term: str) -> List[Dict[str, Any]]:
    if not term:
        return records
    query = term.lower().strip()
    return [r for r in records if query in str(r.get("matchup", "")).lower()]


def filter_model_supported(records: List[Dict[str, Any]], only_supported: bool) -> List[Dict[str, Any]]:
    if not only_supported:
        return records
    filtered: List[Dict[str, Any]] = []
    for record in records:
        alignment = record.get("model_alignment") or {}
        if alignment.get("model_supports_pick"):
            filtered.append(record)
    return filtered


def filter_edge_threshold(records: List[Dict[str, Any]], min_edge: float) -> List[Dict[str, Any]]:
    if min_edge <= 0:
        return records
    filtered: List[Dict[str, Any]] = []
    for record in records:
        alignment = record.get("model_alignment") or {}
        edge = alignment.get("edge_points")
        if edge is None:
            continue
        try:
            if float(edge) >= min_edge:
                filtered.append(record)
        except (TypeError, ValueError):
            continue
    return filtered


def render_table(records: List[Dict[str, Any]], *, max_depth: int = 2) -> None:
    if not records:
        st.info("No rows match the current filters.")
        return
    frame = records_to_frame(records, max_depth=max_depth)
    st.dataframe(frame, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(
        page_title="CBB Betting Dashboard",
        page_icon="🏀",
        layout="wide",
    )
    st.title("🏀 College Basketball Betting Algorithm Dashboard")
    st.caption(
        "Live market + public split signals with model projections "
        "(power/efficiency proxy/injury/home-court)."
    )

    with st.sidebar:
        st.header("Run settings")
        league = st.text_input("League", value="ncaab")
        public_threshold = st.slider("Public threshold (%)", 50.0, 95.0, 70.0, 1.0)
        public_metric = st.selectbox("Public metric", ["money", "tickets"], index=0)
        selected_books = st.multiselect(
            "Sportsbooks",
            options=[book_id for book_id, _ in DEFAULT_BOOK_OPTIONS],
            default=[book_id for book_id, _ in DEFAULT_BOOK_OPTIONS],
            format_func=lambda book_id: next(
                (name for candidate_id, name in DEFAULT_BOOK_OPTIONS if candidate_id == book_id),
                str(book_id),
            ),
        )
        home_court_advantage = st.slider("Home-court advantage", 0.0, 6.0, 2.7, 0.1)
        max_workers = st.slider("Parallel workers", 1, 20, 10, 1)

        st.subheader("Parameter 3 odds target")
        alt_target_low = st.number_input("Target low", value=-250, step=5)
        alt_target_high = st.number_input("Target high", value=-200, step=5)
        alt_target_mid = st.number_input("Target midpoint", value=-225, step=5)

        st.subheader("Refresh")
        auto_refresh = st.toggle("Auto-refresh", value=True)
        refresh_seconds = st.slider("Refresh interval (sec)", 15, 600, 120, 15)
        manual_refresh = st.button("Refresh now")

    if not selected_books:
        st.error("Select at least one sportsbook in the sidebar.")
        st.stop()

    refresh_key = 0
    if manual_refresh:
        st.session_state["manual_refresh_nonce"] = st.session_state.get(
            "manual_refresh_nonce", 0
        ) + 1
    refresh_key += int(st.session_state.get("manual_refresh_nonce", 0))

    if auto_refresh:
        if st_autorefresh is None:
            st.warning(
                "Auto-refresh package not installed. "
                "Install `streamlit-autorefresh` or use manual refresh."
            )
        else:
            tick_count = st_autorefresh(
                interval=int(refresh_seconds * 1000),
                key="cbb_auto_refresh",
            )
            refresh_key += int(tick_count)

    config = {
        "league": league.strip() or "ncaab",
        "public_threshold": float(public_threshold),
        "public_metric": public_metric,
        "book_ids": ",".join(str(book_id) for book_id in selected_books),
        "max_workers": int(max_workers),
        "home_court_advantage": float(home_court_advantage),
        "alt_target_low": int(alt_target_low),
        "alt_target_high": int(alt_target_high),
        "alt_target_mid": int(alt_target_mid),
    }
    config_json = json.dumps(config, sort_keys=True)

    with st.spinner("Pulling live games, lines, and model context..."):
        try:
            report = load_report_cached(config_json, refresh_key)
        except Exception as exc:
            st.error(f"Failed to load report: {exc}")
            st.stop()

    metadata = report.get("metadata", {})
    all_games_snapshot = report.get("all_games_snapshot", [])
    model_projections = report.get("model_projections", [])
    parameter_1 = report.get("parameter_1", [])
    parameter_2_spread = report.get("parameter_2", {}).get("spread", [])
    parameter_2_totals = report.get("parameter_2", {}).get("totals", [])
    parameter_3 = report.get("parameter_3", [])

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Games", len(all_games_snapshot))
    col2.metric("Model projections", len(model_projections))
    col3.metric("Parameter 1", len(parameter_1))
    col4.metric("Param 2 spread", len(parameter_2_spread))
    col5.metric("Param 2 totals", len(parameter_2_totals))
    col6.metric("Parameter 3", len(parameter_3))

    st.caption(
        "Generated: {generated} | Public metric: {metric} | Threshold: {threshold}% | Books: {books}".format(
            generated=metadata.get("generated_at_utc", "unknown"),
            metric=metadata.get("public_metric", "unknown"),
            threshold=metadata.get("public_threshold_pct", "unknown"),
            books=", ".join(str(x) for x in metadata.get("preferred_book_ids", [])),
        )
    )

    st.divider()
    st.subheader("Filters")
    filter_col1, filter_col2, filter_col3 = st.columns([2, 1, 1])
    matchup_filter = filter_col1.text_input("Matchup contains")
    model_supported_only = filter_col2.toggle("Only model-supported picks", value=False)
    min_model_edge = filter_col3.number_input("Min model edge", value=0.0, step=0.25)

    tabs = st.tabs(
        [
            "Overview",
            "Parameter 1",
            "Parameter 2 (Spread)",
            "Parameter 2 (Totals)",
            "Parameter 3",
            "Model Projections",
            "Raw JSON",
        ]
    )

    with tabs[0]:
        st.markdown("### Game Snapshot (selected sportsbooks)")
        filtered_games = filter_matchup(all_games_snapshot, matchup_filter)
        render_table(filtered_games, max_depth=3)

    with tabs[1]:
        st.markdown("### Parameter 1 Picks")
        rows = filter_matchup(parameter_1, matchup_filter)
        rows = filter_model_supported(rows, model_supported_only)
        rows = filter_edge_threshold(rows, min_model_edge)
        render_table(rows, max_depth=3)

    with tabs[2]:
        st.markdown("### Parameter 2 Spread Picks")
        rows = filter_matchup(parameter_2_spread, matchup_filter)
        rows = filter_model_supported(rows, model_supported_only)
        rows = filter_edge_threshold(rows, min_model_edge)
        render_table(rows, max_depth=3)

    with tabs[3]:
        st.markdown("### Parameter 2 Total Picks")
        rows = filter_matchup(parameter_2_totals, matchup_filter)
        rows = filter_model_supported(rows, model_supported_only)
        rows = filter_edge_threshold(rows, min_model_edge)
        render_table(rows, max_depth=3)

    with tabs[4]:
        st.markdown("### Parameter 3 Safer Alternate-Style Picks")
        rows = filter_matchup(parameter_3, matchup_filter)
        rows = filter_model_supported(rows, model_supported_only)
        rows = filter_edge_threshold(rows, min_model_edge)
        render_table(rows, max_depth=4)

    with tabs[5]:
        st.markdown("### Model Projections")
        rows = filter_matchup(model_projections, matchup_filter)
        render_table(rows, max_depth=4)

    with tabs[6]:
        st.markdown("### Raw Report JSON")
        st.json(report)


if __name__ == "__main__":
    main()
