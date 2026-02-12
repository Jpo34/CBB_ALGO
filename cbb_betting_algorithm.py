#!/usr/bin/env python3
"""
College basketball betting signal engine based on public splits and line movement.

Data sources:
- Action Network NCAAB public betting page (current game board and sportsbook metadata)
- Action Network market history endpoint for opening-to-current line movement
- Action Network game detail pages (trend + injury context for model projections)

The report contains:
1) Parameter 1 picks
2) Parameter 2 picks (spread + total)
3) Parameter 3 picks (safer alternate-style recommendation)
4) Model projections (team power/efficiency proxies/injury adjustments)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import math
import os
import re
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests


ACTION_SITE_ROOT = "https://www.actionnetwork.com"
ACTION_API_ROOT = "https://api.actionnetwork.com/web"
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
)

# NJ book IDs exposed on the Action odds/public board.
DEFAULT_BOOK_IDS = [68, 69, 71, 75, 79]
DEFAULT_PUBLIC_THRESHOLD = 70.0
DEFAULT_REFRESH_INTERVAL_SECONDS = 120
DEFAULT_HOME_COURT_ADVANTAGE = 2.7
DEFAULT_REQUEST_TIMEOUT_SECONDS = 20
DEFAULT_REQUEST_RETRIES = 3

INJURY_STATUS_WEIGHTS = {
    "out_for_season": 1.35,
    "out": 1.0,
    "suspended": 1.0,
    "doubtful": 0.8,
    "questionable": 0.45,
    "day_to_day": 0.35,
    "probable": 0.1,
}


class DataFetchError(RuntimeError):
    """Raised when a required remote payload cannot be fetched/parsed."""


def utc_now_iso() -> str:
    return dt.datetime.now(tz=dt.timezone.utc).replace(microsecond=0).isoformat()


def parse_iso(ts: str) -> dt.datetime:
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return dt.datetime.fromisoformat(ts)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def safe_float(value: Any, fallback: float) -> float:
    try:
        if value is None:
            return fallback
        return float(value)
    except (TypeError, ValueError):
        return fallback


def normalize_team_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def standings_win_pct(team: Dict[str, Any]) -> float:
    standings = team.get("standings") or {}
    wins = safe_float(standings.get("win"), 0.0)
    losses = safe_float(standings.get("loss"), 0.0)
    total = wins + losses
    if total <= 0:
        return 0.5
    return wins / total


def extract_team_seed(team: Dict[str, Any]) -> Optional[Any]:
    direct_keys = ("seed", "tournament_seed", "ap_rank", "rank", "coaches_rank")
    for key in direct_keys:
        value = team.get(key)
        if value not in (None, "", 0):
            return value

    standings = team.get("standings") or {}
    for key in direct_keys:
        value = standings.get(key)
        if value not in (None, "", 0):
            return value
    return None


def find_record_win_pct(
    records: List[Dict[str, Any]], record_type: str, fallback: float
) -> float:
    for record in records:
        if record.get("record_type") == record_type:
            return safe_float(record.get("win_pct"), fallback)
    return fallback


def request_with_retries(
    session: requests.Session,
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = 30,
    retries: int = 4,
) -> requests.Response:
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(
                url,
                params=params,
                timeout=timeout,
                headers={"User-Agent": USER_AGENT},
            )
            if response.status_code >= 500 and attempt < retries:
                time.sleep(2 ** attempt)
                continue
            return response
        except requests.RequestException as exc:  # pragma: no cover - network-dependent
            last_exc = exc
            if attempt < retries:
                time.sleep(2 ** attempt)
                continue
            raise DataFetchError(f"Request failed for {url}: {exc}") from exc
    raise DataFetchError(f"Request failed for {url}: {last_exc}")


def parse_next_data(html: str) -> Dict[str, Any]:
    match = re.search(
        r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
        html,
        flags=re.DOTALL,
    )
    if not match:
        raise DataFetchError("Could not find __NEXT_DATA__ payload in page HTML.")
    return json.loads(match.group(1))


def build_game_links_from_html(html: str, league: str) -> Dict[int, str]:
    game_route_prefix = re.escape(f"/{league}-game/")
    game_href_regex = rf'href="({game_route_prefix}[^"]+/(\d+))"'
    game_links: Dict[int, str] = {}
    for href, game_id_str in re.findall(game_href_regex, html):
        try:
            game_id = int(game_id_str)
        except ValueError:
            continue
        game_links[game_id] = f"{ACTION_SITE_ROOT}{href}"
    return game_links


def extract_board_from_page_props(
    page_props: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    scoreboard = page_props.get("scoreboardResponse", {})
    games = scoreboard.get("games", [])
    all_books = page_props.get("allBooks", {})
    return games, all_books


def parse_board_html_payload(
    html: str, league: str
) -> Optional[Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[int, str], str]]:
    try:
        next_data = parse_next_data(html)
    except Exception:
        return None
    page_props = next_data.get("props", {}).get("pageProps", {})
    games, all_books = extract_board_from_page_props(page_props)
    game_links = build_game_links_from_html(html, league)
    build_id = str(next_data.get("buildId") or "")
    return games, all_books, game_links, build_id


def fetch_next_data_board(
    session: requests.Session,
    *,
    league: str,
    build_id: str,
    route_name: str,
    request_timeout: int,
    request_retries: int,
) -> Optional[Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[int, str]]]:
    url = f"{ACTION_SITE_ROOT}/_next/data/{build_id}/{league}/{route_name}.json"
    response = request_with_retries(
        session,
        url,
        params={"_ts": int(time.time() * 1000)},
        timeout=request_timeout,
        retries=request_retries,
    )
    if response.status_code != 200:
        return None
    try:
        payload = response.json()
    except Exception:
        return None
    page_props = payload.get("pageProps", {})
    games, all_books = extract_board_from_page_props(page_props)
    return games, all_books, {}


def fetch_books_index(
    session: requests.Session,
    *,
    request_timeout: int,
    request_retries: int,
) -> Dict[str, Dict[str, Any]]:
    url = f"{ACTION_API_ROOT}/v1/books"
    response = request_with_retries(
        session,
        url,
        timeout=request_timeout,
        retries=request_retries,
    )
    if response.status_code != 200:
        return {}
    try:
        payload = response.json()
    except Exception:
        return {}
    books = payload.get("books")
    if not isinstance(books, list):
        return {}
    return {str(book.get("id")): book for book in books if isinstance(book, dict)}


def fetch_board_data(
    session: requests.Session,
    league: str,
    *,
    request_timeout: int,
    request_retries: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[int, str]]:
    # Try HTML routes first.
    candidate_paths = [f"/{league}/public-betting", f"/{league}/odds"]
    discovered_build_id = ""
    challenge_seen = False
    for path in candidate_paths:
        url = f"{ACTION_SITE_ROOT}{path}"
        for attempt in range(1, request_retries + 1):
            response = request_with_retries(
                session,
                url,
                params={"_ts": int(time.time() * 1000)},
                timeout=request_timeout,
                retries=request_retries,
            )
            parsed = parse_board_html_payload(response.text, league)
            if parsed:
                games, all_books, game_links, build_id = parsed
                if build_id:
                    discovered_build_id = build_id
                if response.status_code in (200, 202):
                    return games, all_books, game_links
            if response.status_code == 202 and attempt < request_retries:
                challenge_seen = True
                time.sleep(min(4, attempt + 1))
                continue
            break

    # If HTML is challenged or unavailable, use Next.js data route fallback.
    if not discovered_build_id:
        league_url = f"{ACTION_SITE_ROOT}/{league}"
        response = request_with_retries(
            session,
            league_url,
            params={"_ts": int(time.time() * 1000)},
            timeout=request_timeout,
            retries=request_retries,
        )
        parsed = parse_board_html_payload(response.text, league)
        if parsed:
            _, _, _, build_id = parsed
            discovered_build_id = build_id

    if discovered_build_id:
        for route_name in ("public-betting", "odds"):
            next_data_result = fetch_next_data_board(
                session,
                league=league,
                build_id=discovered_build_id,
                route_name=route_name,
                request_timeout=request_timeout,
                request_retries=request_retries,
            )
            if next_data_result is not None:
                games, all_books, game_links = next_data_result
                return games, all_books, game_links

    # Last fallback: API scoreboard + books index.
    scoreboard_url = f"{ACTION_API_ROOT}/v1/scoreboard/{league}"
    response = request_with_retries(
        session,
        scoreboard_url,
        timeout=request_timeout,
        retries=request_retries,
    )
    if response.status_code == 200:
        payload = response.json()
        games = payload.get("games", [])
        all_books = fetch_books_index(
            session,
            request_timeout=request_timeout,
            request_retries=request_retries,
        )
        return games, all_books, {}

    extra = " (202 challenge seen)" if challenge_seen else ""
    raise DataFetchError(
        f"Unable to fetch board data for {league}.{extra} "
        f"Last status: {response.status_code}"
    )


def fetch_market_history(
    session: requests.Session,
    game_id: int,
    book_ids: Iterable[int],
    *,
    request_timeout: int,
    request_retries: int,
) -> Dict[str, Any]:
    url = f"{ACTION_API_ROOT}/v2/markets/event/{game_id}/history"
    params = {"bookIds": ",".join(str(x) for x in book_ids)}
    response = request_with_retries(
        session,
        url,
        params=params,
        timeout=request_timeout,
        retries=request_retries,
    )
    if response.status_code == 404:
        return {}
    if response.status_code != 200:
        raise DataFetchError(
            f"History fetch failed for game {game_id} (status {response.status_code})"
        )
    return response.json()


def fetch_game_detail_context(
    session: requests.Session,
    game_id: int,
    game_url: Optional[str],
    *,
    request_timeout: int,
    request_retries: int,
) -> Dict[str, Any]:
    if not game_url:
        return {}

    response = request_with_retries(
        session,
        game_url,
        params={"_ts": int(time.time() * 1000)},
        timeout=request_timeout,
        retries=request_retries,
    )
    if response.status_code != 200:
        raise DataFetchError(
            f"Game detail fetch failed for game {game_id} (status {response.status_code})"
        )

    page_props = parse_next_data(response.text).get("props", {}).get("pageProps", {})
    game_payload = page_props.get("game", {})
    situational = game_payload.get("trends", {}).get("situational", {})
    injuries = page_props.get("injuries") if isinstance(page_props.get("injuries"), list) else []

    return {
        "situational": situational,
        "injuries": injuries,
    }


def build_team_aliases(team: Dict[str, Any]) -> List[str]:
    aliases = [
        team.get("display_name"),
        team.get("full_name"),
        team.get("short_name"),
        team.get("abbr"),
        team.get("location"),
    ]
    unique_aliases: List[str] = []
    seen = set()
    for value in aliases:
        normalized = normalize_team_name(str(value or ""))
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique_aliases.append(normalized)
    return unique_aliases


def summarize_team_injuries(
    injuries: List[Dict[str, Any]], teams: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, float]]:
    side_aliases = {
        "home": set(build_team_aliases(teams["home"])),
        "away": set(build_team_aliases(teams["away"])),
    }

    summary = {
        "home": {"count": 0.0, "key_count": 0.0, "impact": 0.0},
        "away": {"count": 0.0, "key_count": 0.0, "impact": 0.0},
    }

    for injury in injuries:
        team_block = injury.get("team") or {}
        injury_team_name = normalize_team_name(
            str(team_block.get("display_name") or team_block.get("full_name") or "")
        )
        if not injury_team_name:
            continue

        side = None
        if injury_team_name in side_aliases["home"]:
            side = "home"
        elif injury_team_name in side_aliases["away"]:
            side = "away"
        if not side:
            continue

        status = str(injury.get("status") or "").lower().strip()
        status_weight = INJURY_STATUS_WEIGHTS.get(status, 0.25)
        is_key_player = bool(injury.get("is_key_player"))
        impact = status_weight * (1.35 if is_key_player else 1.0)

        summary[side]["count"] += 1.0
        summary[side]["impact"] += impact
        if is_key_player:
            summary[side]["key_count"] += 1.0

    return summary


def build_team_model_profile(
    *,
    side: str,
    teams: Dict[str, Dict[str, Any]],
    situational: Dict[str, Any],
    injury_summary: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    team = teams[side]
    trend = situational.get(side) or {}
    records = trend.get("records") or []

    win_pct = safe_float(trend.get("win_p"), standings_win_pct(team))
    points_for = safe_float(trend.get("points_for"), 70.0)
    points_against = safe_float(trend.get("points_against"), 70.0)
    point_diff = safe_float(trend.get("point_diff"), points_for - points_against)
    last_10_win_pct = find_record_win_pct(records, "last_10", win_pct)
    conference_win_pct = find_record_win_pct(records, "conference", win_pct)

    injury_count = injury_summary[side]["count"]
    key_injury_count = injury_summary[side]["key_count"]
    injury_impact = injury_summary[side]["impact"]

    power_rating = 100.0
    power_rating += (win_pct - 0.5) * 26.0
    power_rating += point_diff * 1.45
    power_rating += (last_10_win_pct - 0.5) * 10.0
    power_rating += (conference_win_pct - 0.5) * 6.0
    power_rating += (points_for - 70.0) * 0.33
    power_rating += (70.0 - points_against) * 0.33
    power_rating -= injury_impact * 2.15

    return {
        "win_pct": win_pct,
        "last_10_win_pct": last_10_win_pct,
        "conference_win_pct": conference_win_pct,
        "points_for": points_for,
        "points_against": points_against,
        "point_diff": point_diff,
        "injury_count": injury_count,
        "key_injury_count": key_injury_count,
        "injury_impact": injury_impact,
        "power_rating": power_rating,
    }


def build_model_projection(
    *,
    game: Dict[str, Any],
    teams: Dict[str, Dict[str, Any]],
    detail_context: Dict[str, Any],
    home_court_advantage: float,
) -> Dict[str, Any]:
    situational = detail_context.get("situational") or {}
    injuries = detail_context.get("injuries") or []
    injury_summary = summarize_team_injuries(injuries, teams)

    home_profile = build_team_model_profile(
        side="home",
        teams=teams,
        situational=situational,
        injury_summary=injury_summary,
    )
    away_profile = build_team_model_profile(
        side="away",
        teams=teams,
        situational=situational,
        injury_summary=injury_summary,
    )

    home_off_base = (home_profile["points_for"] + away_profile["points_against"]) / 2.0
    away_off_base = (away_profile["points_for"] + home_profile["points_against"]) / 2.0
    power_gap = home_profile["power_rating"] - away_profile["power_rating"]

    projected_home_score = (
        home_off_base
        + home_court_advantage * 0.58
        + power_gap * 0.16
        - home_profile["injury_impact"] * 0.28
        + away_profile["injury_impact"] * 0.12
    )
    projected_away_score = (
        away_off_base
        - home_court_advantage * 0.42
        - power_gap * 0.12
        - away_profile["injury_impact"] * 0.28
        + home_profile["injury_impact"] * 0.12
    )

    projected_home_score = clamp(projected_home_score, 45.0, 110.0)
    projected_away_score = clamp(projected_away_score, 45.0, 110.0)

    projected_margin_home = projected_home_score - projected_away_score
    projected_total = projected_home_score + projected_away_score
    model_home_spread = -projected_margin_home
    home_win_probability = 1.0 / (1.0 + math.exp(-(projected_margin_home / 6.5)))
    home_win_probability = clamp(home_win_probability, 0.01, 0.99)

    confidence_score = clamp(
        50.0
        + abs(power_gap) * 0.9
        + abs(projected_margin_home) * 1.2
        + abs(projected_total - 140.0) * 0.2,
        50.0,
        99.0,
    )

    return {
        "home_team": teams["home"]["display_name"],
        "away_team": teams["away"]["display_name"],
        "home_profile": {
            key: round(value, 4) for key, value in home_profile.items()
        },
        "away_profile": {
            key: round(value, 4) for key, value in away_profile.items()
        },
        "projected_home_score": round(projected_home_score, 3),
        "projected_away_score": round(projected_away_score, 3),
        "projected_margin_home": round(projected_margin_home, 3),
        "projected_total": round(projected_total, 3),
        "model_home_spread": round(model_home_spread, 3),
        "home_win_probability": round(home_win_probability, 4),
        "away_win_probability": round(1.0 - home_win_probability, 4),
        "confidence_score": round(confidence_score, 2),
        "inputs_available": bool(detail_context),
    }


def model_market_edges(
    *,
    model_projection: Dict[str, Any],
    spread_summary: Optional[Dict[str, Any]],
    total_summary: Optional[Dict[str, Any]],
    teams: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    edges: Dict[str, Any] = {"spread": None, "total": None}
    projected_margin_home = float(model_projection["projected_margin_home"])
    projected_total = float(model_projection["projected_total"])

    if spread_summary:
        market_home_spread = float(spread_summary["lines"]["home"]["current"]["line"])
        market_away_spread = float(spread_summary["lines"]["away"]["current"]["line"])
        home_cover_edge = projected_margin_home + market_home_spread
        away_cover_edge = -projected_margin_home + market_away_spread
        preferred_side = "home" if home_cover_edge >= away_cover_edge else "away"
        edges["spread"] = {
            "market_home_spread": market_home_spread,
            "market_away_spread": market_away_spread,
            "home_cover_edge": round(home_cover_edge, 3),
            "away_cover_edge": round(away_cover_edge, 3),
            "preferred_side": preferred_side,
            "preferred_team": teams[preferred_side]["display_name"],
        }

    if total_summary:
        market_total = float(total_summary["current_total"])
        total_edge_over = projected_total - market_total
        edges["total"] = {
            "market_total": market_total,
            "total_edge_over": round(total_edge_over, 3),
            "total_edge_under": round(-total_edge_over, 3),
            "lean": "over" if total_edge_over > 0 else "under",
        }

    return edges


def spread_pick_model_alignment(
    model_edges: Dict[str, Any], pick_side: str
) -> Optional[Dict[str, Any]]:
    spread_edges = model_edges.get("spread")
    if not spread_edges:
        return None
    edge_key = f"{pick_side}_cover_edge"
    pick_edge = spread_edges.get(edge_key)
    if pick_edge is None:
        return None
    return {
        "model_supports_pick": bool(pick_edge > 0),
        "edge_points": round(float(pick_edge), 3),
        "model_preferred_side": spread_edges.get("preferred_side"),
    }


def total_pick_model_alignment(
    model_edges: Dict[str, Any], pick_side: str
) -> Optional[Dict[str, Any]]:
    total_edges = model_edges.get("total")
    if not total_edges:
        return None
    if pick_side == "over":
        edge_points = safe_float(total_edges.get("total_edge_over"), 0.0)
    elif pick_side == "under":
        edge_points = safe_float(total_edges.get("total_edge_under"), 0.0)
    else:
        return None
    return {
        "model_supports_pick": bool(edge_points > 0),
        "edge_points": round(edge_points, 3),
        "model_total_lean": total_edges.get("lean"),
    }


def get_book_name(book_meta: Dict[str, Any]) -> str:
    return (
        book_meta.get("parent_name")
        or book_meta.get("display_name")
        or book_meta.get("source_name")
        or "Unknown Book"
    )


def get_teams(game: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    home_id = game["home_team_id"]
    away_id = game["away_team_id"]
    home = next((t for t in game["teams"] if t["id"] == home_id), None)
    away = next((t for t in game["teams"] if t["id"] == away_id), None)
    if not home or not away:
        raise DataFetchError(f"Could not resolve teams for game {game.get('id')}")
    return {"home": home, "away": away}


def sort_history(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(history, key=lambda x: parse_iso(x["updated_at"]))


def get_open_current(
    outcome: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    history = outcome.get("history") or []
    if not history:
        return None, None
    ordered = sort_history(history)
    opener = next((x for x in ordered if x.get("line_status") == "opener"), ordered[0])
    current = ordered[-1]
    return opener, current


def get_public_percent(
    outcome: Dict[str, Any], metric: str
) -> Optional[float]:
    value = (
        outcome.get("bet_info", {})
        .get(metric, {})
        .get("percent")
    )
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def pick_analysis_book(
    history_data: Dict[str, Any], preferred_book_ids: List[int]
) -> Optional[str]:
    for book_id in preferred_book_ids:
        key = str(book_id)
        event = history_data.get(key, {}).get("event", {})
        if event.get("spread") and event.get("total"):
            return key
    # fallback: first key with spread + total
    for key, value in history_data.items():
        event = value.get("event", {})
        if event.get("spread") and event.get("total"):
            return key
    return None


def by_side(outcomes: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {o.get("side"): o for o in outcomes}


def spread_movement_summary(
    event: Dict[str, Any],
    teams: Dict[str, Dict[str, Any]],
    public_metric: str,
    threshold: float,
) -> Optional[Dict[str, Any]]:
    spread = event.get("spread")
    if not spread or len(spread) < 2:
        return None

    spread_side = by_side(spread)
    home_outcome = spread_side.get("home")
    away_outcome = spread_side.get("away")
    if not home_outcome or not away_outcome:
        return None

    home_open, home_current = get_open_current(home_outcome)
    away_open, away_current = get_open_current(away_outcome)
    if not (home_open and home_current and away_open and away_current):
        return None

    home_open_val = float(home_open["value"])
    away_open_val = float(away_open["value"])
    home_current_val = float(home_current["value"])
    away_current_val = float(away_current["value"])

    favorite_side = "home" if home_open_val < away_open_val else "away"
    favorite_open = home_open_val if favorite_side == "home" else away_open_val
    favorite_current = home_current_val if favorite_side == "home" else away_current_val

    moved_toward_favorite = abs(favorite_current) > abs(favorite_open)
    moved_away_from_favorite = abs(favorite_current) < abs(favorite_open)
    has_moved = not math.isclose(favorite_current, favorite_open, abs_tol=1e-9)

    home_public = get_public_percent(home_outcome, public_metric) or 0.0
    away_public = get_public_percent(away_outcome, public_metric) or 0.0
    heavy_public_side = None
    heavy_public_pct = 0.0
    if max(home_public, away_public) >= threshold:
        heavy_public_side = "home" if home_public >= away_public else "away"
        heavy_public_pct = max(home_public, away_public)

    public_is_favorite = (
        heavy_public_side == favorite_side if heavy_public_side is not None else None
    )
    movement_against_public = False
    if heavy_public_side is not None and has_moved:
        movement_against_public = (
            (public_is_favorite and moved_away_from_favorite)
            or ((not public_is_favorite) and moved_toward_favorite)
        )

    return {
        "release_utc": min(home_open["updated_at"], away_open["updated_at"]),
        "last_update_utc": max(home_current["updated_at"], away_current["updated_at"]),
        "favorite_side": favorite_side,
        "favorite_team": teams[favorite_side]["display_name"],
        "favorite_open_spread": favorite_open,
        "favorite_current_spread": favorite_current,
        "moved_toward_favorite": moved_toward_favorite,
        "moved_away_from_favorite": moved_away_from_favorite,
        "has_moved": has_moved,
        "heavy_public_side": heavy_public_side,
        "heavy_public_team": teams[heavy_public_side]["display_name"]
        if heavy_public_side
        else None,
        "heavy_public_pct": heavy_public_pct,
        "public_metric": public_metric,
        "public_is_favorite": public_is_favorite,
        "movement_against_public": movement_against_public,
        "lines": {
            "home": {
                "opening": {"line": home_open_val, "odds": home_open.get("odds")},
                "current": {
                    "line": home_current_val,
                    "odds": home_current.get("odds"),
                },
            },
            "away": {
                "opening": {"line": away_open_val, "odds": away_open.get("odds")},
                "current": {
                    "line": away_current_val,
                    "odds": away_current.get("odds"),
                },
            },
        },
    }


def total_movement_summary(
    event: Dict[str, Any], public_metric: str, threshold: float
) -> Optional[Dict[str, Any]]:
    total = event.get("total")
    if not total or len(total) < 2:
        return None

    total_side = by_side(total)
    over_outcome = total_side.get("over")
    under_outcome = total_side.get("under")
    if not over_outcome or not under_outcome:
        return None

    over_open, over_current = get_open_current(over_outcome)
    under_open, under_current = get_open_current(under_outcome)
    if not (over_open and over_current and under_open and under_current):
        return None

    open_total = float(over_open["value"])
    current_total = float(over_current["value"])
    has_moved = not math.isclose(open_total, current_total, abs_tol=1e-9)
    moved_down = current_total < open_total
    moved_up = current_total > open_total

    over_public = get_public_percent(over_outcome, public_metric) or 0.0
    under_public = get_public_percent(under_outcome, public_metric) or 0.0
    heavy_side = None
    heavy_pct = 0.0
    if max(over_public, under_public) >= threshold:
        heavy_side = "over" if over_public >= under_public else "under"
        heavy_pct = max(over_public, under_public)

    movement_against_public = False
    if heavy_side and has_moved:
        movement_against_public = (
            (heavy_side == "over" and moved_down)
            or (heavy_side == "under" and moved_up)
        )

    return {
        "release_utc": min(over_open["updated_at"], under_open["updated_at"]),
        "last_update_utc": max(over_current["updated_at"], under_current["updated_at"]),
        "open_total": open_total,
        "current_total": current_total,
        "moved_down": moved_down,
        "moved_up": moved_up,
        "has_moved": has_moved,
        "heavy_public_side": heavy_side,
        "heavy_public_pct": heavy_pct,
        "public_metric": public_metric,
        "movement_against_public": movement_against_public,
        "lines": {
            "over": {
                "opening": {"line": float(over_open["value"]), "odds": over_open.get("odds")},
                "current": {
                    "line": float(over_current["value"]),
                    "odds": over_current.get("odds"),
                },
            },
            "under": {
                "opening": {
                    "line": float(under_open["value"]),
                    "odds": under_open.get("odds"),
                },
                "current": {
                    "line": float(under_current["value"]),
                    "odds": under_current.get("odds"),
                },
            },
        },
    }


def choose_opposite_side(side: str) -> str:
    if side == "home":
        return "away"
    if side == "away":
        return "home"
    if side == "over":
        return "under"
    if side == "under":
        return "over"
    raise ValueError(f"Unknown side: {side}")


def choose_target_candidate(
    candidates: List[Dict[str, Any]],
    target_low: int,
    target_high: int,
    target_mid: int,
) -> Optional[Dict[str, Any]]:
    usable = [
        c
        for c in candidates
        if isinstance(c.get("odds"), (int, float))
    ]
    if not usable:
        return None

    in_range = [
        c for c in usable if target_low <= float(c["odds"]) <= target_high
    ]
    pool = in_range if in_range else usable
    selected = min(pool, key=lambda c: abs(float(c["odds"]) - target_mid))
    selected = dict(selected)
    selected["in_target_range"] = bool(in_range)
    return selected


def safer_alternate_pick(
    history_data: Dict[str, Any],
    all_books: Dict[str, Dict[str, Any]],
    preferred_book_ids: List[int],
    pick_side: str,
    base_spread_line: float,
    *,
    target_low: int = -250,
    target_high: int = -200,
    target_mid: int = -225,
) -> Optional[Dict[str, Any]]:
    # 1) Try explicit alternate spread outcomes from the feed.
    alt_candidates: List[Dict[str, Any]] = []
    for book_id in preferred_book_ids:
        key = str(book_id)
        event = history_data.get(key, {}).get("event", {})
        spread = event.get("spread") or []
        for outcome in spread:
            if outcome.get("side") != pick_side:
                continue
            if not outcome.get("is_alt_market"):
                continue
            value = float(outcome.get("value"))
            # "Easier to hit": favorite => less negative, underdog => more positive.
            if base_spread_line < 0 and value <= base_spread_line:
                continue
            if base_spread_line > 0 and value >= base_spread_line:
                continue
            alt_candidates.append(
                {
                    "market": "alternate_spread",
                    "book_id": int(key),
                    "sportsbook": get_book_name(all_books.get(key, {})),
                    "line": value,
                    "odds": outcome.get("odds"),
                }
            )

    selected_alt = choose_target_candidate(
        alt_candidates,
        target_low=target_low,
        target_high=target_high,
        target_mid=target_mid,
    )
    if selected_alt:
        selected_alt["note"] = "Explicit alternate spread pulled from sportsbook feed."
        return selected_alt

    # 2) If no alt spread in feed, use the closest moneyline proxy.
    ml_candidates: List[Dict[str, Any]] = []
    for book_id in preferred_book_ids:
        key = str(book_id)
        event = history_data.get(key, {}).get("event", {})
        moneyline = event.get("moneyline") or []
        for outcome in moneyline:
            if outcome.get("side") != pick_side:
                continue
            ml_candidates.append(
                {
                    "market": "moneyline_proxy",
                    "book_id": int(key),
                    "sportsbook": get_book_name(all_books.get(key, {})),
                    "line": None,
                    "odds": outcome.get("odds"),
                }
            )

    selected_ml = choose_target_candidate(
        ml_candidates,
        target_low=target_low,
        target_high=target_high,
        target_mid=target_mid,
    )
    if selected_ml:
        selected_ml["note"] = (
            "No alternate spread was returned by this feed; "
            "using a safer moneyline proxy from sportsbook odds."
        )
        return selected_ml

    return None


def build_game_snapshot(
    game: Dict[str, Any],
    history_data: Dict[str, Any],
    all_books: Dict[str, Dict[str, Any]],
    preferred_book_ids: List[int],
    *,
    include_books: bool,
) -> Dict[str, Any]:
    teams = get_teams(game)
    books_snapshot: List[Dict[str, Any]] = []
    if include_books:
        for book_id in preferred_book_ids:
            key = str(book_id)
            event = history_data.get(key, {}).get("event", {})
            if not event:
                continue

            spread_summary = spread_movement_summary(
                event, teams, public_metric="money", threshold=0.0
            )
            total_summary = total_movement_summary(
                event, public_metric="money", threshold=0.0
            )
            moneyline = event.get("moneyline") or []
            moneyline_side = by_side(moneyline)
            books_snapshot.append(
                {
                    "book_id": book_id,
                    "sportsbook": get_book_name(all_books.get(key, {})),
                    "spread": spread_summary["lines"] if spread_summary else None,
                    "spread_release_utc": spread_summary["release_utc"]
                    if spread_summary
                    else None,
                    "spread_last_update_utc": spread_summary["last_update_utc"]
                    if spread_summary
                    else None,
                    "total": total_summary["lines"] if total_summary else None,
                    "total_release_utc": total_summary["release_utc"]
                    if total_summary
                    else None,
                    "total_last_update_utc": total_summary["last_update_utc"]
                    if total_summary
                    else None,
                    "moneyline": {
                        side: {
                            "odds": outcome.get("odds"),
                            "public_tickets_pct": get_public_percent(outcome, "tickets"),
                            "public_money_pct": get_public_percent(outcome, "money"),
                        }
                        for side, outcome in moneyline_side.items()
                    },
                }
            )

    away = teams["away"]["display_name"]
    home = teams["home"]["display_name"]
    return {
        "game_id": game["id"],
        "matchup": f"{away} @ {home}",
        "away_team": away,
        "home_team": home,
        "away_seed": extract_team_seed(teams["away"]),
        "home_seed": extract_team_seed(teams["home"]),
        "start_time_utc": game.get("start_time"),
        "num_bets": game.get("num_bets"),
        "books": books_snapshot,
    }


def run_analysis(args: argparse.Namespace) -> Dict[str, Any]:
    session = requests.Session()

    games, all_books, game_links = fetch_board_data(
        session,
        args.league,
        request_timeout=args.request_timeout,
        request_retries=args.request_retries,
    )
    preferred_book_ids = [int(x) for x in args.book_ids.split(",") if x.strip()]
    include_model = bool(args.include_model)

    histories: Dict[int, Dict[str, Any]] = {}
    detail_contexts: Dict[int, Dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_map: Dict[concurrent.futures.Future, Tuple[str, int]] = {}
        for game in games:
            game_id = game["id"]
            future_map[
                executor.submit(
                    fetch_market_history,
                    session,
                    game_id,
                    preferred_book_ids,
                    request_timeout=args.request_timeout,
                    request_retries=args.request_retries,
                )
            ] = ("history", game_id)

            game_url = game_links.get(game_id)
            if game_url and include_model and not args.skip_detail_context:
                future_map[
                    executor.submit(
                        fetch_game_detail_context,
                        session,
                        game_id,
                        game_url,
                        request_timeout=args.request_timeout,
                        request_retries=args.request_retries,
                    )
                ] = ("detail", game_id)

        for future in concurrent.futures.as_completed(future_map):
            fetch_kind, game_id = future_map[future]
            try:
                result = future.result()
                if fetch_kind == "history":
                    histories[game_id] = result
                else:
                    detail_contexts[game_id] = result
            except Exception as exc:  # pragma: no cover - network-dependent
                if fetch_kind == "history":
                    histories[game_id] = {}
                    warning_context = "history"
                else:
                    detail_contexts[game_id] = {}
                    warning_context = "detail context"
                print(
                    f"[warn] Failed to fetch {warning_context} for game {game_id}: {exc}",
                    file=sys.stderr,
                )

    all_games_snapshot: List[Dict[str, Any]] = []
    model_projections: List[Dict[str, Any]] = []
    parameter_1: List[Dict[str, Any]] = []
    parameter_2_spread: List[Dict[str, Any]] = []
    parameter_2_total: List[Dict[str, Any]] = []
    parameter_3: List[Dict[str, Any]] = []

    for game in games:
        game_id = game["id"]
        history_data = histories.get(game_id) or {}
        all_games_snapshot.append(
            build_game_snapshot(
                game=game,
                history_data=history_data,
                all_books=all_books,
                preferred_book_ids=preferred_book_ids,
                include_books=not args.compact_output,
            )
        )

        teams = get_teams(game)
        away_team_name = teams["away"]["display_name"]
        home_team_name = teams["home"]["display_name"]
        away_seed = extract_team_seed(teams["away"])
        home_seed = extract_team_seed(teams["home"])

        analysis_book = pick_analysis_book(history_data, preferred_book_ids)
        matchup = f"{away_team_name} @ {home_team_name}"
        game_url = game_links.get(game_id)
        sportsbook = (
            get_book_name(all_books.get(analysis_book, {}))
            if analysis_book
            else None
        )
        event = history_data.get(analysis_book, {}).get("event", {}) if analysis_book else {}
        spread_summary = (
            spread_movement_summary(
                event=event,
                teams=teams,
                public_metric=args.public_metric,
                threshold=args.public_threshold,
            )
            if analysis_book
            else None
        )
        total_summary = (
            total_movement_summary(
                event=event,
                public_metric=args.public_metric,
                threshold=args.public_threshold,
            )
            if analysis_book
            else None
        )
        model_edges: Dict[str, Any] = {}
        if include_model:
            detail_context = detail_contexts.get(game_id) or {}
            model_projection = build_model_projection(
                game=game,
                teams=teams,
                detail_context=detail_context,
                home_court_advantage=args.home_court_advantage,
            )
            model_edges = model_market_edges(
                model_projection=model_projection,
                spread_summary=spread_summary,
                total_summary=total_summary,
                teams=teams,
            )

            model_projections.append(
                {
                    "game_id": game_id,
                    "matchup": matchup,
                    "away_team": away_team_name,
                    "home_team": home_team_name,
                    "away_seed": away_seed,
                    "home_seed": home_seed,
                    "start_time_utc": game.get("start_time"),
                    "game_url": game_url,
                    "analysis_sportsbook": sportsbook,
                    "analysis_book_id": int(analysis_book) if analysis_book else None,
                    "line_release_utc": spread_summary["release_utc"] if spread_summary else None,
                    "line_last_update_utc": spread_summary["last_update_utc"] if spread_summary else None,
                    "total_release_utc": total_summary["release_utc"] if total_summary else None,
                    "total_last_update_utc": total_summary["last_update_utc"] if total_summary else None,
                    "model_projection": model_projection,
                    "market_edges": model_edges,
                }
            )

        if not analysis_book:
            continue

        game_url = game_links.get(game_id)

        # ----------------------------
        # Parameter 1
        # Public >= threshold on one spread side, while line is not moving toward favorite.
        # ----------------------------
        if spread_summary and spread_summary["heavy_public_side"]:
            if not spread_summary["moved_toward_favorite"]:
                fade_side = choose_opposite_side(spread_summary["heavy_public_side"])
                fade_team = teams[fade_side]["display_name"]
                fade_line = spread_summary["lines"][fade_side]["current"]["line"]
                fade_odds = spread_summary["lines"][fade_side]["current"]["odds"]
                parameter_1.append(
                    {
                        "game_id": game_id,
                        "matchup": matchup,
                        "away_team": away_team_name,
                        "home_team": home_team_name,
                        "away_seed": away_seed,
                        "home_seed": home_seed,
                        "start_time_utc": game.get("start_time"),
                        "game_url": game_url,
                        "sportsbook": sportsbook,
                        "book_id": int(analysis_book),
                        "line_release_utc": spread_summary["release_utc"],
                        "line_last_update_utc": spread_summary["last_update_utc"],
                        "public_side": spread_summary["heavy_public_team"],
                        "public_pct": spread_summary["heavy_public_pct"],
                        "public_metric": spread_summary["public_metric"],
                        "favorite_team": spread_summary["favorite_team"],
                        "favorite_open_spread": spread_summary["favorite_open_spread"],
                        "favorite_current_spread": spread_summary["favorite_current_spread"],
                        "moved_toward_favorite": spread_summary["moved_toward_favorite"],
                        "recommended_pick": {
                            "market": "spread",
                            "side": fade_side,
                            "team": fade_team,
                            "line": fade_line,
                            "odds": fade_odds,
                        },
                        "reason": (
                            f"Public is {spread_summary['heavy_public_pct']:.1f}% on "
                            f"{spread_summary['heavy_public_team']} but line did not move "
                            "toward the favorite."
                        ),
                    }
                )
                if include_model:
                    parameter_1[-1]["model_alignment"] = spread_pick_model_alignment(
                        model_edges,
                        fade_side,
                    )

        # ----------------------------
        # Parameter 2 (Spread)
        # Public-heavy side + line movement against public side => fade public.
        # ----------------------------
        if spread_summary and spread_summary["heavy_public_side"]:
            if spread_summary["has_moved"] and spread_summary["movement_against_public"]:
                fade_side = choose_opposite_side(spread_summary["heavy_public_side"])
                fade_team = teams[fade_side]["display_name"]
                fade_line = spread_summary["lines"][fade_side]["current"]["line"]
                fade_odds = spread_summary["lines"][fade_side]["current"]["odds"]
                parameter_2_spread.append(
                    {
                        "game_id": game_id,
                        "matchup": matchup,
                        "away_team": away_team_name,
                        "home_team": home_team_name,
                        "away_seed": away_seed,
                        "home_seed": home_seed,
                        "start_time_utc": game.get("start_time"),
                        "game_url": game_url,
                        "sportsbook": sportsbook,
                        "book_id": int(analysis_book),
                        "line_release_utc": spread_summary["release_utc"],
                        "line_last_update_utc": spread_summary["last_update_utc"],
                        "public_side": spread_summary["heavy_public_team"],
                        "public_pct": spread_summary["heavy_public_pct"],
                        "public_metric": spread_summary["public_metric"],
                        "favorite_team": spread_summary["favorite_team"],
                        "favorite_open_spread": spread_summary["favorite_open_spread"],
                        "favorite_current_spread": spread_summary["favorite_current_spread"],
                        "movement_against_public": spread_summary["movement_against_public"],
                        "recommended_pick": {
                            "market": "spread",
                            "side": fade_side,
                            "team": fade_team,
                            "line": fade_line,
                            "odds": fade_odds,
                        },
                        "reason": (
                            "Public side and spread movement are in conflict "
                            "(reverse line movement signal)."
                        ),
                    }
                )
                if include_model:
                    parameter_2_spread[-1]["model_alignment"] = spread_pick_model_alignment(
                        model_edges,
                        fade_side,
                    )

                # ----------------------------
                # Parameter 3
                # Same conflict condition as Parameter 2 spread, but recommend safer
                # alternate-style exposure around -200 to -250 when available.
                # ----------------------------
                alt_pick = safer_alternate_pick(
                    history_data=history_data,
                    all_books=all_books,
                    preferred_book_ids=preferred_book_ids,
                    pick_side=fade_side,
                    base_spread_line=float(fade_line),
                    target_low=args.alt_target_low,
                    target_high=args.alt_target_high,
                    target_mid=args.alt_target_mid,
                )
                parameter_3.append(
                    {
                        "game_id": game_id,
                        "matchup": matchup,
                        "away_team": away_team_name,
                        "home_team": home_team_name,
                        "away_seed": away_seed,
                        "home_seed": home_seed,
                        "start_time_utc": game.get("start_time"),
                        "game_url": game_url,
                        "analysis_sportsbook": sportsbook,
                        "analysis_book_id": int(analysis_book),
                        "line_release_utc": spread_summary["release_utc"],
                        "line_last_update_utc": spread_summary["last_update_utc"],
                        "trigger": {
                            "public_side": spread_summary["heavy_public_team"],
                            "public_pct": spread_summary["heavy_public_pct"],
                            "public_metric": spread_summary["public_metric"],
                            "favorite_team": spread_summary["favorite_team"],
                            "favorite_open_spread": spread_summary["favorite_open_spread"],
                            "favorite_current_spread": spread_summary["favorite_current_spread"],
                            "movement_against_public": spread_summary["movement_against_public"],
                        },
                        "base_pick": {
                            "market": "spread",
                            "side": fade_side,
                            "team": fade_team,
                            "line": fade_line,
                            "odds": fade_odds,
                        },
                        "safer_alternate_pick": alt_pick,
                        "target_odds_window": {
                            "min": args.alt_target_low,
                            "max": args.alt_target_high,
                            "midpoint": args.alt_target_mid,
                        },
                    }
                )
                if include_model:
                    parameter_3[-1]["model_alignment"] = spread_pick_model_alignment(
                        model_edges,
                        fade_side,
                    )

        # ----------------------------
        # Parameter 2 (Totals)
        # If public is heavy on over/under and total moves opposite, fade public side.
        # ----------------------------
        if total_summary and total_summary["heavy_public_side"]:
            if total_summary["has_moved"] and total_summary["movement_against_public"]:
                fade_side = choose_opposite_side(total_summary["heavy_public_side"])
                fade_line = total_summary["lines"][fade_side]["current"]["line"]
                fade_odds = total_summary["lines"][fade_side]["current"]["odds"]
                parameter_2_total.append(
                    {
                        "game_id": game_id,
                        "matchup": matchup,
                        "away_team": away_team_name,
                        "home_team": home_team_name,
                        "away_seed": away_seed,
                        "home_seed": home_seed,
                        "start_time_utc": game.get("start_time"),
                        "game_url": game_url,
                        "sportsbook": sportsbook,
                        "book_id": int(analysis_book),
                        "line_release_utc": total_summary["release_utc"],
                        "line_last_update_utc": total_summary["last_update_utc"],
                        "public_side": total_summary["heavy_public_side"],
                        "public_pct": total_summary["heavy_public_pct"],
                        "public_metric": total_summary["public_metric"],
                        "open_total": total_summary["open_total"],
                        "current_total": total_summary["current_total"],
                        "movement_against_public": total_summary["movement_against_public"],
                        "recommended_pick": {
                            "market": "total",
                            "side": fade_side,
                            "line": fade_line,
                            "odds": fade_odds,
                        },
                        "reason": (
                            "Public total side and total movement are in conflict "
                            "(reverse line movement signal)."
                        ),
                    }
                )
                if include_model:
                    parameter_2_total[-1]["model_alignment"] = total_pick_model_alignment(
                        model_edges,
                        fade_side,
                    )

    return {
        "metadata": {
            "generated_at_utc": utc_now_iso(),
            "league": args.league,
            "public_threshold_pct": args.public_threshold,
            "public_metric": args.public_metric,
            "home_court_advantage": args.home_court_advantage,
            "detail_context_enabled": not args.skip_detail_context,
            "request_timeout_seconds": args.request_timeout,
            "request_retries": args.request_retries,
            "include_model": include_model,
            "compact_output": bool(args.compact_output),
            "preferred_book_ids": preferred_book_ids,
            "preferred_sportsbooks": [
                {
                    "book_id": bid,
                    "sportsbook": get_book_name(all_books.get(str(bid), {})),
                }
                for bid in preferred_book_ids
            ],
            "data_sources": [
                f"{ACTION_SITE_ROOT}/{args.league}/public-betting",
                f"{ACTION_API_ROOT}/v2/markets/event/<game_id>/history?bookIds=<ids>",
            ],
            "notes": [
                "Line release date is taken from the earliest 'opener' timestamp in market history.",
                "Parameter 3 attempts explicit alternate spread odds first; if unavailable in feed, it falls back to a safer moneyline proxy.",
                (
                    "Model projections use trend and injury context from each game detail page."
                    if include_model
                    else "Model projections disabled for compact output mode."
                ),
            ],
        },
        "all_games_snapshot": all_games_snapshot,
        "model_projections": model_projections,
        "parameter_1": parameter_1,
        "parameter_2": {
            "spread": parameter_2_spread,
            "totals": parameter_2_total,
        },
        "parameter_3": parameter_3,
    }


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a college basketball betting signal report from public splits, "
            "opening/current line movement, and safer alternate-style recommendations."
        )
    )
    parser.add_argument(
        "--league",
        default="ncaab",
        help="Action Network league slug (default: ncaab).",
    )
    parser.add_argument(
        "--public-threshold",
        type=float,
        default=DEFAULT_PUBLIC_THRESHOLD,
        help="Public split threshold percent for triggers (default: 70).",
    )
    parser.add_argument(
        "--public-metric",
        choices=["money", "tickets"],
        default="money",
        help="Public split metric to evaluate (default: money).",
    )
    parser.add_argument(
        "--book-ids",
        default=",".join(str(x) for x in DEFAULT_BOOK_IDS),
        help=(
            "Comma-separated sportsbook book IDs to use for lines/history "
            "(default: 68,69,71,75,79)."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="Max worker threads for per-game history calls (default: 10).",
    )
    parser.add_argument(
        "--include-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable model projection payload and model alignment fields (default: on).",
    )
    parser.add_argument(
        "--compact-output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reduce report size by omitting per-book snapshot details (default: off).",
    )
    parser.add_argument(
        "--skip-detail-context",
        action="store_true",
        help=(
            "Skip per-game detail page pulls (trends/injuries) for faster loads. "
            "Model will still run with reduced context."
        ),
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        help=f"HTTP request timeout seconds (default: {DEFAULT_REQUEST_TIMEOUT_SECONDS}).",
    )
    parser.add_argument(
        "--request-retries",
        type=int,
        default=DEFAULT_REQUEST_RETRIES,
        help=f"HTTP request retries (default: {DEFAULT_REQUEST_RETRIES}).",
    )
    parser.add_argument(
        "--home-court-advantage",
        type=float,
        default=DEFAULT_HOME_COURT_ADVANTAGE,
        help="Home-court points added in model projections (default: 2.7).",
    )
    parser.add_argument(
        "--alt-target-low",
        type=int,
        default=-250,
        help="Parameter 3 lower bound for preferred odds window (default: -250).",
    )
    parser.add_argument(
        "--alt-target-high",
        type=int,
        default=-200,
        help="Parameter 3 upper bound for preferred odds window (default: -200).",
    )
    parser.add_argument(
        "--alt-target-mid",
        type=int,
        default=-225,
        help="Parameter 3 odds midpoint target (default: -225).",
    )
    parser.add_argument(
        "--output",
        default="",
        help=(
            "Output JSON path. "
            "Default: reports/cbb_betting_report_<UTC timestamp>.json"
        ),
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuously refresh reports on an interval.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=DEFAULT_REFRESH_INTERVAL_SECONDS,
        help=f"Refresh interval for --watch mode (default: {DEFAULT_REFRESH_INTERVAL_SECONDS}).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=0,
        help="Max loops in --watch mode. 0 means run indefinitely (default: 0).",
    )
    parser.add_argument(
        "--latest-output",
        default="reports/latest_cbb_report.json",
        help=(
            "Latest report path written each watch iteration "
            "(default: reports/latest_cbb_report.json)."
        ),
    )
    parser.add_argument(
        "--archive-dir",
        default="reports/archive",
        help=(
            "Archive directory for timestamped watch reports. "
            "Set empty string to disable archives."
        ),
    )
    return parser.parse_args(argv)


def ensure_parent_dir(file_path: str) -> None:
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def write_report_json(report: Dict[str, Any], output_path: str) -> None:
    ensure_parent_dir(output_path)
    with open(output_path, "w", encoding="utf-8") as file_handle:
        json.dump(report, file_handle, indent=2)


def market_signature_map(report: Dict[str, Any]) -> Dict[int, Tuple[Any, Any]]:
    signatures: Dict[int, Tuple[Any, Any]] = {}
    for game in report.get("all_games_snapshot", []):
        game_id = game.get("game_id")
        books = game.get("books") or []
        if game_id is None or not books:
            continue
        primary_book = books[0]
        spread_home = None
        total_value = None
        if primary_book.get("spread"):
            spread_home = (
                primary_book["spread"]
                .get("home", {})
                .get("current", {})
                .get("line")
            )
        if primary_book.get("total"):
            total_value = (
                primary_book["total"]
                .get("over", {})
                .get("current", {})
                .get("line")
            )
        signatures[int(game_id)] = (spread_home, total_value)
    return signatures


def game_name_map(report: Dict[str, Any]) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for game in report.get("all_games_snapshot", []):
        game_id = game.get("game_id")
        matchup = game.get("matchup")
        if game_id is None:
            continue
        mapping[int(game_id)] = str(matchup or game_id)
    return mapping


def run_watch_mode(args: argparse.Namespace) -> int:
    iteration = 0
    previous_report: Optional[Dict[str, Any]] = None

    while True:
        iteration += 1
        started = time.time()
        try:
            report = run_analysis(args)
        except Exception as exc:
            print(f"[watch][error] Iteration {iteration} failed: {exc}", file=sys.stderr)
            if args.max_iterations and iteration >= args.max_iterations:
                return 1
            time.sleep(max(1, int(args.interval_seconds)))
            continue

        latest_path = args.latest_output or args.output
        if latest_path:
            write_report_json(report, latest_path)

        archived_path = ""
        if args.archive_dir:
            ts = dt.datetime.now(tz=dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            archived_path = os.path.join(
                args.archive_dir,
                f"cbb_betting_report_{ts}.json",
            )
            write_report_json(report, archived_path)

        current_ids = {int(g["game_id"]) for g in report.get("all_games_snapshot", [])}
        current_signatures = market_signature_map(report)
        current_game_names = game_name_map(report)

        new_games: List[str] = []
        removed_games: List[str] = []
        changed_lines = 0
        if previous_report is not None:
            previous_ids = {
                int(g["game_id"]) for g in previous_report.get("all_games_snapshot", [])
            }
            previous_names = game_name_map(previous_report)
            previous_signatures = market_signature_map(previous_report)
            new_ids = sorted(current_ids - previous_ids)
            removed_ids = sorted(previous_ids - current_ids)

            new_games = [current_game_names.get(game_id, str(game_id)) for game_id in new_ids]
            removed_games = [previous_names.get(game_id, str(game_id)) for game_id in removed_ids]

            shared_ids = previous_ids & current_ids
            for game_id in shared_ids:
                if previous_signatures.get(game_id) != current_signatures.get(game_id):
                    changed_lines += 1

        runtime = time.time() - started
        print(
            "[watch] iteration={iteration} games={games} "
            "param1={p1} param2_spread={p2s} param2_total={p2t} param3={p3} "
            "new_games={new_count} removed_games={removed_count} line_changes={line_changes} "
            "runtime_sec={runtime:.1f}".format(
                iteration=iteration,
                games=len(report.get("all_games_snapshot", [])),
                p1=len(report.get("parameter_1", [])),
                p2s=len(report.get("parameter_2", {}).get("spread", [])),
                p2t=len(report.get("parameter_2", {}).get("totals", [])),
                p3=len(report.get("parameter_3", [])),
                new_count=len(new_games),
                removed_count=len(removed_games),
                line_changes=changed_lines,
                runtime=runtime,
            )
        )
        if new_games:
            print(f"[watch] new games: {', '.join(new_games[:10])}")
        if removed_games:
            print(f"[watch] removed games: {', '.join(removed_games[:10])}")
        if latest_path:
            print(f"[watch] latest report: {latest_path}")
        if archived_path:
            print(f"[watch] archived report: {archived_path}")

        previous_report = report
        if args.max_iterations and iteration >= args.max_iterations:
            return 0

        sleep_for = max(0.0, float(args.interval_seconds) - runtime)
        time.sleep(sleep_for)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    if args.watch:
        return run_watch_mode(args)

    try:
        report = run_analysis(args)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    ts = dt.datetime.now(tz=dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = args.output or f"reports/cbb_betting_report_{ts}.json"
    write_report_json(report, output_path)

    print(f"Report written to: {output_path}")
    print(f"Games captured: {len(report['all_games_snapshot'])}")
    print(f"Model projections: {len(report['model_projections'])}")
    print(f"Parameter 1 picks: {len(report['parameter_1'])}")
    print(f"Parameter 2 spread picks: {len(report['parameter_2']['spread'])}")
    print(f"Parameter 2 total picks: {len(report['parameter_2']['totals'])}")
    print(f"Parameter 3 picks: {len(report['parameter_3'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
