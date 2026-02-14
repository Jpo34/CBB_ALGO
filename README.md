# College Basketball Betting Algorithm

This project builds a college basketball betting signal report from live market data.

It pulls:
- Current NCAAB games
- Current spreads, totals, and moneylines
- Public betting splits (money or tickets)
- Opening line timestamps and current line timestamps (for movement tracking)

Then it evaluates three parameter groups **in separate report sections** so they can be cross-referenced.
It also builds a model projection layer (power/efficiency proxies/injuries/home-court).

## Data sources

- `https://www.actionnetwork.com/ncaab/public-betting`
- `https://api.actionnetwork.com/web/v2/markets/event/<game_id>/history?bookIds=<ids>`
- `https://www.actionnetwork.com/ncaab-game/.../<game_id>` (for trends + injuries)

The script is configured for reputable sportsbook feeds by default:
- DraftKings (NJ)
- FanDuel (NJ)
- BetRivers (NJ)
- BetMGM (NJ)
- bet365 (NJ)

## Parameters implemented

1. **Parameter 1 (Core Team Strength / True Line)**
   - Builds a base spread projection from efficiency and form proxies:
     - adjusted scoring margin proxy (`points_for - points_against`) with SOS adjustment
     - tempo proxy (`points_for + points_against`)
     - home-court advantage
     - recent form weighting (last 5 weighted over last 10)
     - injury/rotation impact
   - Keeps games where model-vs-market spread edge is at least a threshold (default `2.0` points).

2. **Parameter 2 (Market Confirmation / Value Layer)**
   - Confirms Parameter 1 edges using market behavior:
     - public skew threshold (default `78%`)
     - reverse line movement against public side (`>= 1.5` points by default)
     - late steam timing (default within `6` hours of tip)
     - cross-book confirmation (`>= 2` books)

3. **Parameter 3 (Portfolio Quality / Discipline Layer)**
   - Final bet set after risk filters:
     - liquidity floor (`num_bets` threshold)
     - avoid big favorites (>10 points) unless edge is very large
     - key-number sensitivity around `3/4/7/10`
     - minimum probability edge vs implied odds (default `>= 3%`)
     - optional underdogs-only mode
   - Totals are off by default for spread-focused discipline.

## Model layer

The script can include a `model_projections` section per game using:

- Team power rating (win rate, point differential, recent/conference form)
- Efficiency proxies (points for / points against)
- Injury adjustments (status + key-player weighting)
- Home-court adjustment (configurable)

Model output includes:
- projected score by team
- projected spread (home line)
- projected total
- win probabilities
- market edge estimates vs current sportsbook line
- true-line input breakdown (tempo, recent form, SOS proxy, injury gap)

Each parameter pick now includes `model_alignment` so you can see whether model edge supports the pick.

## Setup

```bash
python3 -m pip install -r requirements.txt
```

## Streamlit dashboard

Launch the UI:

```bash
python3 -m streamlit run app.py
```

Dashboard features:
- tabs for:
  - Top Recommended Bets (descending)
  - Parameter 1
  - Parameter 2
  - Parameter 3
  - All Picks
- date scope selector: **Today + Tomorrow**, **Today**, or **Tomorrow**
- game cards with matchup + recommended pick text
- team seed display when available from feed
- matchup filter
- auto-refresh interval control (for frequent checks as lines/games update)
- optional advanced trigger pack:
  - last-10 form gap
  - top-25 / over-.500 proxy
  - venue split edge (home/road)
  - ATS last-10 momentum

If loads feel slow, turn on **Fast mode** in the sidebar (skips per-game detail context for quicker refreshes).

## Usage

Run with defaults:

```bash
python3 cbb_betting_algorithm.py
```

Common options:

```bash
python3 cbb_betting_algorithm.py \
  --public-threshold 78 \
  --public-metric money \
  --true-line-min-edge-points 2.0 \
  --min-probability-edge 0.03 \
  --min-liquidity-bets 500 \
  --book-ids 68,69,71,75,79 \
  --home-court-advantage 2.7 \
  --output reports/my_cbb_report.json
```

## Frequent updates (watch mode)

Use watch mode to refresh automatically so new games appear as soon as lines/totals are posted.

```bash
python3 cbb_betting_algorithm.py \
  --watch \
  --interval-seconds 120 \
  --latest-output reports/latest_cbb_report.json \
  --archive-dir reports/archive
```

Useful flags:
- `--interval-seconds 60` for faster polling
- `--max-iterations 10` to run a fixed number of loops
- `--archive-dir ""` to disable archive files and only keep latest output

## Output

The script writes a JSON report containing:

- `all_games_snapshot`: full game + sportsbook line snapshot
- `model_projections`: model-based spread/total/win probability projections
- `parameter_1`: core team-strength true-line edges
- `parameter_2.spread`: market-confirmed spread edges
- `parameter_2.totals`: reserved (totals off by default in hybrid mode)
- `parameter_3`: final discipline-filtered spread bets

It also includes:
- line release timestamps (`opener`)
- latest line update timestamps
- sportsbook attribution and book IDs
- reasoning fields for each recommendation
- model alignment fields for each pick

## Notes

- Lines and splits can change quickly.
- Alternate-line automation exists but is disabled by default in the hybrid setup.
- `--watch` mode is the easiest way to keep the report fresh throughout the day.