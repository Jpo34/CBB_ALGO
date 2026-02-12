#!/usr/bin/env python3
"""
College basketball betting signal engine based on public splits and line movement.

Data sources:
- Action Network NCAAB public betting page (current game board and sportsbook metadata)
- Action Network market history endpoint for opening-to-current line movement

The report contains:
1) Parameter 1 picks
2) Parameter 2 picks (spread + total)
3) Parameter 3 picks (safer alternate-style recommendation)
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


class DataFetchError(RuntimeError):
    """Raised when a required remote payload cannot be fetched/parsed."""


def utc_now_iso() -> str:
    return dt.datetime.now(tz=dt.timezone.utc).replace(microsecond=0).isoformat()


def parse_iso(ts: str) -> dt.datetime:
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return dt.datetime.fromisoformat(ts)


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


def fetch_board_data(
    session: requests.Session, league: str
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[int, str]]:
    page_url = f"{ACTION_SITE_ROOT}/{league}/public-betting"
    response = request_with_retries(session, page_url)
    if response.status_code != 200:
        raise DataFetchError(
            f"Unexpected status code {response.status_code} for {page_url}"
        )

    html = response.text
    next_data = parse_next_data(html)
    page_props = next_data.get("props", {}).get("pageProps", {})
    scoreboard = page_props.get("scoreboardResponse", {})
    games = scoreboard.get("games", [])
    all_books = page_props.get("allBooks", {})

    # Build a game_id -> detail page URL map.
    game_links: Dict[int, str] = {}
    for href, game_id_str in re.findall(r'href="(/ncaab-game/[^"]+/(\d+))"', html):
        try:
            game_id = int(game_id_str)
        except ValueError:
            continue
        game_links[game_id] = f"{ACTION_SITE_ROOT}{href}"

    return games, all_books, game_links


def fetch_market_history(
    session: requests.Session, game_id: int, book_ids: Iterable[int]
) -> Dict[str, Any]:
    url = f"{ACTION_API_ROOT}/v2/markets/event/{game_id}/history"
    params = {"bookIds": ",".join(str(x) for x in book_ids)}
    response = request_with_retries(session, url, params=params)
    if response.status_code == 404:
        return {}
    if response.status_code != 200:
        raise DataFetchError(
            f"History fetch failed for game {game_id} (status {response.status_code})"
        )
    return response.json()


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
) -> Dict[str, Any]:
    teams = get_teams(game)
    books_snapshot: List[Dict[str, Any]] = []
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
        "start_time_utc": game.get("start_time"),
        "num_bets": game.get("num_bets"),
        "books": books_snapshot,
    }


def run_analysis(args: argparse.Namespace) -> Dict[str, Any]:
    session = requests.Session()

    games, all_books, game_links = fetch_board_data(session, args.league)
    preferred_book_ids = [int(x) for x in args.book_ids.split(",") if x.strip()]

    histories: Dict[int, Dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_map = {
            executor.submit(fetch_market_history, session, game["id"], preferred_book_ids): game["id"]
            for game in games
        }
        for future in concurrent.futures.as_completed(future_map):
            game_id = future_map[future]
            try:
                histories[game_id] = future.result()
            except Exception as exc:  # pragma: no cover - network-dependent
                histories[game_id] = {}
                print(
                    f"[warn] Failed to fetch history for game {game_id}: {exc}",
                    file=sys.stderr,
                )

    all_games_snapshot: List[Dict[str, Any]] = []
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
            )
        )

        teams = get_teams(game)
        analysis_book = pick_analysis_book(history_data, preferred_book_ids)
        if not analysis_book:
            continue

        event = history_data.get(analysis_book, {}).get("event", {})
        spread_summary = spread_movement_summary(
            event=event,
            teams=teams,
            public_metric=args.public_metric,
            threshold=args.public_threshold,
        )
        total_summary = total_movement_summary(
            event=event,
            public_metric=args.public_metric,
            threshold=args.public_threshold,
        )

        matchup = f"{teams['away']['display_name']} @ {teams['home']['display_name']}"
        sportsbook = get_book_name(all_books.get(analysis_book, {}))
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

    return {
        "metadata": {
            "generated_at_utc": utc_now_iso(),
            "league": args.league,
            "public_threshold_pct": args.public_threshold,
            "public_metric": args.public_metric,
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
            ],
        },
        "all_games_snapshot": all_games_snapshot,
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
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    try:
        report = run_analysis(args)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    ts = dt.datetime.now(tz=dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = args.output or f"reports/cbb_betting_report_{ts}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file_handle:
        json.dump(report, file_handle, indent=2)

    print(f"Report written to: {output_path}")
    print(f"Games captured: {len(report['all_games_snapshot'])}")
    print(f"Parameter 1 picks: {len(report['parameter_1'])}")
    print(f"Parameter 2 spread picks: {len(report['parameter_2']['spread'])}")
    print(f"Parameter 2 total picks: {len(report['parameter_2']['totals'])}")
    print(f"Parameter 3 picks: {len(report['parameter_3'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
