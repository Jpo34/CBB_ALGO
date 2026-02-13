#!/usr/bin/env python3
"""Simplified Streamlit dashboard for CBB betting picks."""

from __future__ import annotations

import argparse
import json
import os
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
LOCAL_CACHE_PATH = ".streamlit_cache/last_non_empty_report.json"


def load_local_cached_report() -> Dict[str, Any] | None:
    if not os.path.exists(LOCAL_CACHE_PATH):
        return None
    try:
        with open(LOCAL_CACHE_PATH, "r", encoding="utf-8") as file_handle:
            return json.load(file_handle)
    except Exception:
        return None


def save_local_cached_report(report: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(LOCAL_CACHE_PATH), exist_ok=True)
    with open(LOCAL_CACHE_PATH, "w", encoding="utf-8") as file_handle:
        json.dump(report, file_handle)


def build_engine_args(config: Dict[str, Any]) -> argparse.Namespace:
    return argparse.Namespace(
        league=config["league"],
        public_threshold=config["public_threshold"],
        public_metric=config["public_metric"],
        timezone=config["timezone"],
        day_start_offset=config["day_start_offset"],
        days_ahead=config["days_ahead"],
        book_ids=config["book_ids"],
        max_workers=config["max_workers"],
        include_model=False,
        include_advanced_triggers=config["include_advanced_triggers"],
        include_totals=False,
        enable_alt_automation=False,
        rlm_min_points=1.5,
        rlm_strong_points=2.0,
        rlm_late_hours=6.0,
        rlm_min_book_confirmations=2,
        compact_output=True,
        skip_detail_context=config["skip_detail_context"],
        request_timeout=config["request_timeout"],
        request_retries=config["request_retries"],
        home_court_advantage=2.7,
        alt_target_low=-250,
        alt_target_high=-200,
        alt_target_mid=-225,
        trigger_last10_threshold=0.15,
        trigger_top_tier_threshold=0.1,
        trigger_venue_threshold=0.12,
        trigger_ats_threshold=0.15,
        trigger_total_trend_threshold=0.08,
        trigger_top25_min_games=2,
        trigger_over500_min_games=4,
        output="",
        watch=False,
        interval_seconds=0,
        max_iterations=0,
        latest_output="",
        archive_dir="",
    )


def run_report_uncached(config: Dict[str, Any]) -> Dict[str, Any]:
    args = build_engine_args(config)
    return run_analysis(args)


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
        season = record.get("season_alignment") or {}
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
                "start_time_utc": record.get("start_time_utc"),
                "start_time_local": record.get("start_time_local"),
                "local_date": record.get("local_date"),
                "day_bucket": record.get("day_bucket"),
                "bet_type": "spread",
                "pick_identifier": team.lower(),
                "pick_text": pick_text,
                "sportsbook": record.get("sportsbook"),
                "public_pct": record.get("public_pct"),
                "is_underdog_pick": record.get("is_underdog_pick"),
                "is_home_underdog_pick": record.get("is_home_underdog_pick"),
                "rlm_points": record.get("rlm_points"),
                "rlm_book_confirmation_count": record.get("rlm_book_confirmation_count"),
                "rlm_timing_ok": record.get("rlm_timing_ok"),
                "core_rlm_qualified": record.get("core_rlm_qualified"),
                "season_edge": season.get("season_edge"),
                "season_support": season.get("supports_pick"),
                "season_pick_record": season.get("pick_team_record"),
                "season_opp_record": season.get("opponent_record"),
                "advanced_score": record.get("advanced_trigger_score"),
                "advanced_support": record.get("advanced_trigger_support"),
                "advanced_hits": record.get("advanced_trigger_hits"),
                "reason": record.get("reason"),
            }
        )

    for record in report.get("parameter_2", {}).get("spread", []):
        pick = record.get("recommended_pick") or {}
        season = record.get("season_alignment") or {}
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
                "start_time_utc": record.get("start_time_utc"),
                "start_time_local": record.get("start_time_local"),
                "local_date": record.get("local_date"),
                "day_bucket": record.get("day_bucket"),
                "bet_type": "spread",
                "pick_identifier": team.lower(),
                "pick_text": pick_text,
                "sportsbook": record.get("sportsbook"),
                "public_pct": record.get("public_pct"),
                "is_underdog_pick": record.get("is_underdog_pick"),
                "is_home_underdog_pick": record.get("is_home_underdog_pick"),
                "rlm_points": record.get("rlm_points"),
                "rlm_book_confirmation_count": record.get("rlm_book_confirmation_count"),
                "rlm_timing_ok": record.get("rlm_timing_ok"),
                "core_rlm_qualified": record.get("core_rlm_qualified"),
                "season_edge": season.get("season_edge"),
                "season_support": season.get("supports_pick"),
                "season_pick_record": season.get("pick_team_record"),
                "season_opp_record": season.get("opponent_record"),
                "advanced_score": record.get("advanced_trigger_score"),
                "advanced_support": record.get("advanced_trigger_support"),
                "advanced_hits": record.get("advanced_trigger_hits"),
                "reason": record.get("reason"),
            }
        )

    for record in report.get("parameter_2", {}).get("totals", []):
        pick = record.get("recommended_pick") or {}
        season = record.get("season_context") or {}
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
                "start_time_utc": record.get("start_time_utc"),
                "start_time_local": record.get("start_time_local"),
                "local_date": record.get("local_date"),
                "day_bucket": record.get("day_bucket"),
                "bet_type": "total",
                "pick_identifier": side.lower(),
                "pick_text": pick_text,
                "sportsbook": record.get("sportsbook"),
                "public_pct": record.get("public_pct"),
                "is_underdog_pick": record.get("is_underdog_pick"),
                "is_home_underdog_pick": record.get("is_home_underdog_pick"),
                "rlm_points": record.get("rlm_points"),
                "rlm_book_confirmation_count": record.get("rlm_book_confirmation_count"),
                "rlm_timing_ok": record.get("rlm_timing_ok"),
                "core_rlm_qualified": record.get("core_rlm_qualified"),
                "season_edge": None,
                "season_support": None,
                "season_pick_record": season.get("home_record"),
                "season_opp_record": season.get("away_record"),
                "advanced_score": record.get("advanced_trigger_score"),
                "advanced_support": record.get("advanced_trigger_support"),
                "advanced_hits": record.get("advanced_trigger_hits"),
                "reason": record.get("reason"),
            }
        )

    for record in report.get("parameter_3", []):
        safer = record.get("safer_alternate_pick") or {}
        base_pick = record.get("base_pick") or {}
        season = record.get("season_alignment") or {}
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
                "start_time_utc": record.get("start_time_utc"),
                "start_time_local": record.get("start_time_local"),
                "local_date": record.get("local_date"),
                "day_bucket": record.get("day_bucket"),
                "bet_type": bet_type,
                "pick_identifier": pick_identifier,
                "pick_text": pick_text,
                "sportsbook": sportsbook,
                "public_pct": trigger.get("public_pct"),
                "is_underdog_pick": record.get("is_underdog_pick"),
                "is_home_underdog_pick": record.get("is_home_underdog_pick"),
                "rlm_points": record.get("rlm_points"),
                "rlm_book_confirmation_count": record.get("rlm_book_confirmation_count"),
                "rlm_timing_ok": record.get("rlm_timing_ok"),
                "core_rlm_qualified": record.get("core_rlm_qualified"),
                "season_edge": season.get("season_edge"),
                "season_support": season.get("supports_pick"),
                "season_pick_record": season.get("pick_team_record"),
                "season_opp_record": season.get("opponent_record"),
                "advanced_score": record.get("advanced_trigger_score"),
                "advanced_support": record.get("advanced_trigger_support"),
                "advanced_hits": record.get("advanced_trigger_hits"),
                "reason": "Safer alternate-style version of the spread signal.",
            }
        )

    return normalized


def aggregate_top_recommendations(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, str, str], Dict[str, Any]] = {}
    param_sets: Dict[Tuple[Any, str, str], set] = defaultdict(set)
    public_pct_values: Dict[Tuple[Any, str, str], List[float]] = defaultdict(list)
    season_edge_values: Dict[Tuple[Any, str, str], List[float]] = defaultdict(list)
    season_support_votes: Dict[Tuple[Any, str, str], int] = defaultdict(int)
    advanced_score_values: Dict[Tuple[Any, str, str], List[float]] = defaultdict(list)
    underdog_votes: Dict[Tuple[Any, str, str], int] = defaultdict(int)
    home_underdog_votes: Dict[Tuple[Any, str, str], int] = defaultdict(int)

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
        season_edge = entry.get("season_edge")
        if isinstance(season_edge, (int, float)):
            season_edge_values[key].append(float(season_edge))
        season_support = entry.get("season_support")
        if season_support is True:
            season_support_votes[key] += 1
        elif season_support is False:
            season_support_votes[key] -= 1
        advanced_score = entry.get("advanced_score")
        if isinstance(advanced_score, (int, float)):
            advanced_score_values[key].append(float(advanced_score))
        if entry.get("is_underdog_pick") is True:
            underdog_votes[key] += 1
        if entry.get("is_home_underdog_pick") is True:
            home_underdog_votes[key] += 1

    output: List[Dict[str, Any]] = []
    for key, item in grouped.items():
        score = int(item.get("recommendation_score", 0))
        params = sorted(p for p in param_sets[key] if p)
        public_values = public_pct_values[key]
        avg_public = sum(public_values) / len(public_values) if public_values else None
        season_values = season_edge_values[key]
        avg_season_edge = sum(season_values) / len(season_values) if season_values else None
        support_vote = season_support_votes[key]
        season_bonus = 0.0
        if support_vote > 0:
            season_bonus += 0.1
        elif support_vote < 0:
            season_bonus -= 0.1
        if avg_season_edge is not None:
            season_bonus += max(-0.15, min(0.15, avg_season_edge * 0.75))
        advanced_values = advanced_score_values[key]
        avg_advanced_score = (
            sum(advanced_values) / len(advanced_values) if advanced_values else None
        )
        advanced_bonus = 0.0
        if avg_advanced_score is not None:
            advanced_bonus += max(-1.0, min(1.0, avg_advanced_score * 0.4))
        underdog_bonus = 0.0
        if underdog_votes[key] > 0:
            underdog_bonus += 0.2
        if home_underdog_votes[key] > 0:
            underdog_bonus += 0.15
        item["parameters_triggered"] = params
        item["average_public_pct"] = round(avg_public, 2) if avg_public is not None else None
        item["recommendation_score"] = score
        item["average_season_edge"] = (
            round(avg_season_edge, 4) if avg_season_edge is not None else None
        )
        item["average_advanced_score"] = (
            round(avg_advanced_score, 3) if avg_advanced_score is not None else None
        )
        item["season_bonus"] = round(season_bonus, 3)
        item["advanced_bonus"] = round(advanced_bonus, 3)
        item["underdog_bonus"] = round(underdog_bonus, 3)
        item["combined_score"] = round(
            score + season_bonus + advanced_bonus + underdog_bonus,
            3,
        )
        output.append(item)

    output.sort(
        key=lambda x: (
            -float(x.get("combined_score", x.get("recommendation_score", 0))),
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


def filter_day_scope(entries: List[Dict[str, Any]], scope: str) -> List[Dict[str, Any]]:
    if scope == "Today + Tomorrow":
        return entries
    if scope == "Today":
        return [entry for entry in entries if entry.get("day_bucket") == "today"]
    if scope == "Tomorrow":
        return [entry for entry in entries if entry.get("day_bucket") == "tomorrow"]
    return entries


def render_pick_cards(entries: List[Dict[str, Any]], *, show_score: bool = False) -> None:
    if not entries:
        st.info("No picks match this view.")
        return

    for entry in entries:
        with st.container(border=True):
            st.markdown(f"**{format_matchup(entry)}**")
            st.markdown(f"### Pick: {entry.get('pick_text', 'N/A')}")
            local_tip = entry.get("start_time_local")
            if local_tip:
                st.caption(f"Tip-off (local): {local_tip}")
            line_parts = [
                entry.get("parameter", ""),
                str(entry.get("bet_type", "")).upper(),
                entry.get("sportsbook", ""),
            ]
            if show_score:
                line_parts.append(f"Score: {entry.get('recommendation_score', 0)}")
            meta = " • ".join(part for part in line_parts if part)
            st.caption(meta)

            season_edge = entry.get("season_edge")
            if season_edge is not None:
                season_support = entry.get("season_support")
                season_flag = "supports pick" if season_support else "against pick"
                pick_record = entry.get("season_pick_record")
                opp_record = entry.get("season_opp_record")
                record_text = ""
                if pick_record and opp_record:
                    record_text = f" | Records: {pick_record} vs {opp_record}"
                st.caption(f"Season edge: {season_edge} ({season_flag}){record_text}")

            advanced_score = entry.get("advanced_score")
            if advanced_score is not None:
                advanced_support = entry.get("advanced_support")
                advanced_flag = (
                    "supports pick"
                    if advanced_support is True
                    else "against pick"
                    if advanced_support is False
                    else "neutral"
                )
                st.caption(f"Advanced trigger score: {advanced_score} ({advanced_flag})")
                hits = entry.get("advanced_hits") or []
                trigger_labels = [
                    h.get("label") for h in hits if isinstance(h, dict) and h.get("score")
                ]
                if trigger_labels:
                    st.caption("Trigger hits: " + "; ".join(trigger_labels[:4]))

            if show_score:
                params = entry.get("parameters_triggered") or []
                if params:
                    st.caption(f"Triggered by: {', '.join(params)}")
                avg_public = entry.get("average_public_pct")
                if avg_public is not None:
                    st.caption(f"Average public % across triggers: {avg_public}")
                if entry.get("combined_score") is not None:
                    st.caption(
                        "Combined score (params + season + advanced): "
                        f"{entry.get('combined_score')}"
                    )

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
        day_scope = st.selectbox("Game scope", ["Today + Tomorrow", "Today", "Tomorrow"], index=0)
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
            advanced_triggers = st.toggle(
                "Advanced form triggers",
                value=False,
                help=(
                    "Adds last-10/top-tier/venue/ATS style triggers to pick scoring. "
                    "Requires deeper game context."
                ),
            )
            max_workers = st.slider("Parallel workers", 1, 20, 8, 1)
            request_timeout = st.slider("Request timeout (sec)", 5, 40, 12, 1)
            request_retries = st.slider("Request retries", 1, 5, 2, 1)
            timezone_name = st.text_input("Timezone", value="America/New_York")
            if advanced_triggers and fast_mode:
                st.caption(
                    "Fast mode is auto-overridden when advanced triggers are enabled."
                )

        st.subheader("Refresh")
        auto_refresh = st.toggle("Auto-refresh", value=False)
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
            try:
                ticks = st_autorefresh(
                    interval=int(refresh_seconds * 1000),
                    key="cbb_refresh_simple",
                )
                refresh_key += int(ticks)
            except Exception as exc:
                st.warning(
                    "Auto-refresh component failed in this deployment. "
                    f"Falling back to manual refresh. ({exc})"
                )

    config = {
        "league": league.strip() or "ncaab",
        "public_threshold": float(public_threshold),
        "public_metric": public_metric,
        "timezone": timezone_name.strip() or "America/New_York",
        "day_start_offset": 0 if day_scope != "Tomorrow" else 1,
        "days_ahead": 1 if day_scope == "Today + Tomorrow" else 0,
        "book_ids": ",".join(str(book_id) for book_id in selected_books),
        "max_workers": int(max_workers),
        "include_advanced_triggers": bool(advanced_triggers),
        "skip_detail_context": bool(fast_mode and not advanced_triggers),
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
                disk_cache = load_local_cached_report()
                if disk_cache:
                    st.warning(
                        f"Live refresh failed ({exc}). Showing locally cached snapshot."
                    )
                    report = disk_cache
                else:
                    st.error(f"Failed to load report: {exc}")
                    st.stop()
        else:
            st.session_state["last_successful_report"] = report

    game_count = len(report.get("all_games_snapshot", []))
    if game_count == 0:
        last_non_empty = st.session_state.get("last_non_empty_report")
        if isinstance(last_non_empty, dict) and len(last_non_empty.get("all_games_snapshot", [])) > 0:
            st.warning("Live source returned 0 games. Showing last non-empty snapshot.")
            report = last_non_empty
            game_count = len(report.get("all_games_snapshot", []))
        else:
            disk_cache = load_local_cached_report()
            if isinstance(disk_cache, dict) and len(disk_cache.get("all_games_snapshot", [])) > 0:
                st.warning("Live source returned 0 games. Showing locally cached snapshot.")
                report = disk_cache
                game_count = len(report.get("all_games_snapshot", []))
            else:
                retry_config = dict(config)
                retry_config["include_advanced_triggers"] = False
                retry_config["skip_detail_context"] = True
                retry_config["request_retries"] = max(3, int(config["request_retries"]))
                retry_config["request_timeout"] = max(15, int(config["request_timeout"]))
                with st.spinner("Retrying with fallback fast settings..."):
                    try:
                        retry_report = run_report_uncached(retry_config)
                    except Exception:
                        retry_report = {}
                retry_games = len(retry_report.get("all_games_snapshot", []))
                if retry_games > 0:
                    st.success(
                        "Recovered games using fallback fast settings "
                        "(advanced triggers temporarily disabled for this refresh)."
                    )
                    report = retry_report
                    game_count = retry_games
                    st.session_state["last_non_empty_report"] = report
                    save_local_cached_report(report)
                else:
                    st.warning(
                        "Live source returned 0 games. This can happen temporarily while odds providers update."
                    )
    else:
        st.session_state["last_non_empty_report"] = report
        save_local_cached_report(report)

    metadata = report.get("metadata", {})
    entries = normalize_parameter_entries(report)
    top_recommendations = aggregate_top_recommendations(entries)

    p1_count = len(report.get("parameter_1", []))
    p2_count = len(report.get("parameter_2", {}).get("spread", [])) + len(
        report.get("parameter_2", {}).get("totals", [])
    )
    p3_count = len(report.get("parameter_3", []))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Games", game_count)
    m2.metric("Parameter 1 picks", p1_count)
    m3.metric("Parameter 2 picks", p2_count)
    m4.metric("Parameter 3 picks", p3_count)

    st.caption(
        "Generated: {generated} | Metric: {metric} | Threshold: {threshold}% | "
        "Books: {books} | Advanced triggers: {advanced} | Scope: {scope} ({window})".format(
            generated=metadata.get("generated_at_utc", "unknown"),
            metric=metadata.get("public_metric", "unknown"),
            threshold=metadata.get("public_threshold_pct", "unknown"),
            books=", ".join(str(x) for x in metadata.get("preferred_book_ids", [])),
            advanced=(
                "on" if metadata.get("include_advanced_triggers", False) else "off"
            ),
            scope=day_scope,
            window=(
                f"{(metadata.get('day_window') or {}).get('window_start_local_date', '?')} -> "
                f"{(metadata.get('day_window') or {}).get('window_end_local_date', '?')}"
            ),
        )
    )

    st.divider()
    filter_col1, filter_col2 = st.columns([2, 1])
    matchup_filter = filter_col1.text_input("Filter by matchup")
    min_score = filter_col2.slider("Top bets min score", 1, 3, 1, 1)

    filtered_entries = filter_matchup(entries, matchup_filter)
    filtered_entries = filter_day_scope(filtered_entries, day_scope)
    filtered_top = filter_matchup(top_recommendations, matchup_filter)
    filtered_top = filter_day_scope(filtered_top, day_scope)
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
