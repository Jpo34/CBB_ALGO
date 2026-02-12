#!/usr/bin/env python3
"""Simplified Streamlit dashboard for CBB betting picks."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Any, Dict, List, Tuple

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


def build_engine_args(config: Dict[str, Any]) -> argparse.Namespace:
    return argparse.Namespace(
        league=config["league"],
        public_threshold=config["public_threshold"],
        public_metric=config["public_metric"],
        book_ids=config["book_ids"],
        max_workers=config["max_workers"],
        include_model=False,
        compact_output=True,
        skip_detail_context=config["skip_detail_context"],
        request_timeout=config["request_timeout"],
        request_retries=config["request_retries"],
        home_court_advantage=2.7,
        alt_target_low=-250,
        alt_target_high=-200,
        alt_target_mid=-225,
        output="",
        watch=False,
        interval_seconds=0,
        max_iterations=0,
        latest_output="",
        archive_dir="",
    )


@st.cache_data(show_spinner=False)
def load_report_cached(config_json: str, refresh_key: int) -> Dict[str, Any]:
    del refresh_key
    config = json.loads(config_json)
    args = build_engine_args(config)
    return run_analysis(args)


def format_seed(seed: Any) -> str:
    if seed in (None, "", 0):
        return ""
    return f" ({seed})"


def split_matchup(matchup: str) -> Tuple[str, str]:
    if "@" in matchup:
        away, home = matchup.split("@", 1)
        return away.strip(), home.strip()
    return matchup, ""


def format_matchup(entry: Dict[str, Any]) -> str:
    away = entry.get("away_team")
    home = entry.get("home_team")
    if not away or not home:
        away, home = split_matchup(str(entry.get("matchup", "")))
    away_seed = format_seed(entry.get("away_seed"))
    home_seed = format_seed(entry.get("home_seed"))
    if home:
        return f"{away}{away_seed} @ {home}{home_seed}"
    return f"{away}{away_seed}"


def normalize_parameter_entries(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []

    for record in report.get("parameter_1", []):
        pick = record.get("recommended_pick") or {}
        team = pick.get("team") or ""
        line = pick.get("line")
        odds = pick.get("odds")
        pick_text = f"{team} {line} ({odds})" if line is not None else f"{team} ({odds})"
        normalized.append(
            {
                "parameter": "Parameter 1",
                "game_id": record.get("game_id"),
                "matchup": record.get("matchup"),
                "away_team": record.get("away_team"),
                "home_team": record.get("home_team"),
                "away_seed": record.get("away_seed"),
                "home_seed": record.get("home_seed"),
                "bet_type": "spread",
                "pick_identifier": team.lower(),
                "pick_text": pick_text,
                "sportsbook": record.get("sportsbook"),
                "public_pct": record.get("public_pct"),
                "reason": record.get("reason"),
            }
        )

    for record in report.get("parameter_2", {}).get("spread", []):
        pick = record.get("recommended_pick") or {}
        team = pick.get("team") or ""
        line = pick.get("line")
        odds = pick.get("odds")
        pick_text = f"{team} {line} ({odds})" if line is not None else f"{team} ({odds})"
        normalized.append(
            {
                "parameter": "Parameter 2 - Spread",
                "game_id": record.get("game_id"),
                "matchup": record.get("matchup"),
                "away_team": record.get("away_team"),
                "home_team": record.get("home_team"),
                "away_seed": record.get("away_seed"),
                "home_seed": record.get("home_seed"),
                "bet_type": "spread",
                "pick_identifier": team.lower(),
                "pick_text": pick_text,
                "sportsbook": record.get("sportsbook"),
                "public_pct": record.get("public_pct"),
                "reason": record.get("reason"),
            }
        )

    for record in report.get("parameter_2", {}).get("totals", []):
        pick = record.get("recommended_pick") or {}
        side = str(pick.get("side") or "").title()
        line = pick.get("line")
        odds = pick.get("odds")
        pick_text = f"{side} {line} ({odds})"
        normalized.append(
            {
                "parameter": "Parameter 2 - Total",
                "game_id": record.get("game_id"),
                "matchup": record.get("matchup"),
                "away_team": record.get("away_team"),
                "home_team": record.get("home_team"),
                "away_seed": record.get("away_seed"),
                "home_seed": record.get("home_seed"),
                "bet_type": "total",
                "pick_identifier": side.lower(),
                "pick_text": pick_text,
                "sportsbook": record.get("sportsbook"),
                "public_pct": record.get("public_pct"),
                "reason": record.get("reason"),
            }
        )

    for record in report.get("parameter_3", []):
        safer = record.get("safer_alternate_pick") or {}
        base_pick = record.get("base_pick") or {}
        team = base_pick.get("team") or ""
        sportsbook = safer.get("sportsbook") or record.get("analysis_sportsbook")

        market = safer.get("market")
        if market == "moneyline_proxy":
            bet_type = "moneyline"
            odds = safer.get("odds")
            pick_text = f"{team} ML ({odds})"
            pick_identifier = team.lower()
        elif market == "alternate_spread":
            bet_type = "spread"
            line = safer.get("line")
            odds = safer.get("odds")
            pick_text = f"{team} {line} ({odds})"
            pick_identifier = team.lower()
        else:
            bet_type = "spread"
            line = base_pick.get("line")
            odds = base_pick.get("odds")
            pick_text = f"{team} {line} ({odds})"
            pick_identifier = team.lower()

        trigger = record.get("trigger") or {}
        normalized.append(
            {
                "parameter": "Parameter 3",
                "game_id": record.get("game_id"),
                "matchup": record.get("matchup"),
                "away_team": record.get("away_team"),
                "home_team": record.get("home_team"),
                "away_seed": record.get("away_seed"),
                "home_seed": record.get("home_seed"),
                "bet_type": bet_type,
                "pick_identifier": pick_identifier,
                "pick_text": pick_text,
                "sportsbook": sportsbook,
                "public_pct": trigger.get("public_pct"),
                "reason": "Safer alternate-style version of the spread signal.",
            }
        )

    return normalized


def aggregate_top_recommendations(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, str, str], Dict[str, Any]] = {}
    param_sets: Dict[Tuple[Any, str, str], set] = defaultdict(set)
    public_pct_values: Dict[Tuple[Any, str, str], List[float]] = defaultdict(list)

    for entry in entries:
        key = (
            entry.get("game_id"),
            entry.get("bet_type", ""),
            entry.get("pick_identifier", ""),
        )
        if key not in grouped:
            grouped[key] = dict(entry)
            grouped[key]["recommendation_score"] = 0
        grouped[key]["recommendation_score"] += 1
        param_sets[key].add(entry.get("parameter", ""))
        public_pct = entry.get("public_pct")
        if isinstance(public_pct, (int, float)):
            public_pct_values[key].append(float(public_pct))

    output: List[Dict[str, Any]] = []
    for key, item in grouped.items():
        score = int(item.get("recommendation_score", 0))
        params = sorted(p for p in param_sets[key] if p)
        public_values = public_pct_values[key]
        avg_public = sum(public_values) / len(public_values) if public_values else None
        item["parameters_triggered"] = params
        item["average_public_pct"] = round(avg_public, 2) if avg_public is not None else None
        item["recommendation_score"] = score
        output.append(item)

    output.sort(
        key=lambda x: (
            -int(x.get("recommendation_score", 0)),
            -(float(x.get("average_public_pct")) if x.get("average_public_pct") is not None else -1),
            str(x.get("matchup", "")),
        )
    )
    return output


def filter_matchup(entries: List[Dict[str, Any]], term: str) -> List[Dict[str, Any]]:
    if not term:
        return entries
    query = term.lower().strip()
    return [entry for entry in entries if query in str(entry.get("matchup", "")).lower()]


def render_pick_cards(entries: List[Dict[str, Any]], *, show_score: bool = False) -> None:
    if not entries:
        st.info("No picks match this view.")
        return

    for entry in entries:
        with st.container(border=True):
            st.markdown(f"**{format_matchup(entry)}**")
            st.markdown(f"### Pick: {entry.get('pick_text', 'N/A')}")
            line_parts = [
                entry.get("parameter", ""),
                str(entry.get("bet_type", "")).upper(),
                entry.get("sportsbook", ""),
            ]
            if show_score:
                line_parts.append(f"Score: {entry.get('recommendation_score', 0)}")
            meta = " • ".join(part for part in line_parts if part)
            st.caption(meta)

            if show_score:
                params = entry.get("parameters_triggered") or []
                if params:
                    st.caption(f"Triggered by: {', '.join(params)}")
                avg_public = entry.get("average_public_pct")
                if avg_public is not None:
                    st.caption(f"Average public % across triggers: {avg_public}")

            reason = entry.get("reason")
            if reason:
                st.write(reason)


def main() -> None:
    st.set_page_config(page_title="CBB Betting Picks", page_icon="🏀", layout="wide")
    st.title("🏀 CBB Betting Picks")
    st.caption("Simplified board: parameters, picks, and top recommended bets.")

    with st.sidebar:
        st.header("Settings")
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

        with st.expander("Performance", expanded=False):
            fast_mode = st.toggle("Fast mode", value=True)
            max_workers = st.slider("Parallel workers", 1, 20, 8, 1)
            request_timeout = st.slider("Request timeout (sec)", 5, 40, 12, 1)
            request_retries = st.slider("Request retries", 1, 5, 2, 1)

        st.subheader("Refresh")
        auto_refresh = st.toggle("Auto-refresh", value=True)
        refresh_seconds = st.slider("Refresh every (sec)", 15, 600, 120, 15)
        manual_refresh = st.button("Refresh now")

    if not selected_books:
        st.error("Pick at least one sportsbook.")
        st.stop()

    refresh_key = 0
    if manual_refresh:
        st.session_state["manual_refresh_nonce"] = st.session_state.get(
            "manual_refresh_nonce", 0
        ) + 1
    refresh_key += int(st.session_state.get("manual_refresh_nonce", 0))

    if auto_refresh:
        if st_autorefresh is None:
            st.warning("Install `streamlit-autorefresh` to enable auto-refresh.")
        else:
            ticks = st_autorefresh(
                interval=int(refresh_seconds * 1000),
                key="cbb_refresh_simple",
            )
            refresh_key += int(ticks)

    config = {
        "league": league.strip() or "ncaab",
        "public_threshold": float(public_threshold),
        "public_metric": public_metric,
        "book_ids": ",".join(str(book_id) for book_id in selected_books),
        "max_workers": int(max_workers),
        "skip_detail_context": bool(fast_mode),
        "request_timeout": int(request_timeout),
        "request_retries": int(request_retries),
    }
    config_json = json.dumps(config, sort_keys=True)

    st.info("Loading live picks... If slow, keep Fast mode on.")
    with st.spinner("Fetching games and generating parameter picks..."):
        try:
            report = load_report_cached(config_json, refresh_key)
        except Exception as exc:
            fallback_report = st.session_state.get("last_successful_report")
            if fallback_report:
                st.warning(f"Live refresh failed ({exc}). Showing last successful snapshot.")
                report = fallback_report
            else:
                st.error(f"Failed to load report: {exc}")
                st.stop()
        else:
            st.session_state["last_successful_report"] = report

    metadata = report.get("metadata", {})
    entries = normalize_parameter_entries(report)
    top_recommendations = aggregate_top_recommendations(entries)

    p1_count = len(report.get("parameter_1", []))
    p2_count = len(report.get("parameter_2", {}).get("spread", [])) + len(
        report.get("parameter_2", {}).get("totals", [])
    )
    p3_count = len(report.get("parameter_3", []))
    game_count = len(report.get("all_games_snapshot", []))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Games", game_count)
    m2.metric("Parameter 1 picks", p1_count)
    m3.metric("Parameter 2 picks", p2_count)
    m4.metric("Parameter 3 picks", p3_count)

    st.caption(
        "Generated: {generated} | Metric: {metric} | Threshold: {threshold}% | Books: {books}".format(
            generated=metadata.get("generated_at_utc", "unknown"),
            metric=metadata.get("public_metric", "unknown"),
            threshold=metadata.get("public_threshold_pct", "unknown"),
            books=", ".join(str(x) for x in metadata.get("preferred_book_ids", [])),
        )
    )

    st.divider()
    filter_col1, filter_col2 = st.columns([2, 1])
    matchup_filter = filter_col1.text_input("Filter by matchup")
    min_score = filter_col2.slider("Top bets min score", 1, 3, 1, 1)

    filtered_entries = filter_matchup(entries, matchup_filter)
    filtered_top = filter_matchup(top_recommendations, matchup_filter)
    filtered_top = [
        entry for entry in filtered_top if int(entry.get("recommendation_score", 0)) >= min_score
    ]

    tabs = st.tabs(
        [
            "Top Recommended Bets",
            "Parameter 1",
            "Parameter 2",
            "Parameter 3",
            "All Picks",
        ]
    )

    with tabs[0]:
        st.subheader("Most Recommended Bets (descending)")
        render_pick_cards(filtered_top, show_score=True)

    with tabs[1]:
        st.subheader("Parameter 1 Picks")
        section = [e for e in filtered_entries if e.get("parameter") == "Parameter 1"]
        render_pick_cards(section)

    with tabs[2]:
        st.subheader("Parameter 2 Picks")
        spread = [e for e in filtered_entries if e.get("parameter") == "Parameter 2 - Spread"]
        totals = [e for e in filtered_entries if e.get("parameter") == "Parameter 2 - Total"]
        st.markdown("#### Spread")
        render_pick_cards(spread)
        st.markdown("#### Totals")
        render_pick_cards(totals)

    with tabs[3]:
        st.subheader("Parameter 3 Picks")
        section = [e for e in filtered_entries if e.get("parameter") == "Parameter 3"]
        render_pick_cards(section)

    with tabs[4]:
        st.subheader("All Parameter Picks")
        render_pick_cards(filtered_entries)


if __name__ == "__main__":
    main()
