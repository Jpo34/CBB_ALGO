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

1. **Parameter 1 (Spread Contrarian)**
   - If public is at or above the threshold on one spread side, and the spread is **not** moving toward the favorite, fade the public side.

2. **Parameter 2 (Line-Movement Conflict)**
   - **Spread:** if line movement is against the public-heavy side, fade the public side.
   - **Totals:** if public is heavy on over/under and the total moves opposite that side, fade the public side.

3. **Parameter 3 (Safer Alternate-Style Pick)**
   - Triggered on spread conflict scenarios (same trigger as Parameter 2 spread).
   - Attempts to find a safer pick around a target odds window (default `-200` to `-250`):
     - first tries explicit alternate spread markets from feed;
     - if unavailable, uses a safer moneyline proxy from sportsbook odds.

## Model layer added

The script now creates a `model_projections` section per game using:

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
  --public-threshold 70 \
  --public-metric money \
  --book-ids 68,69,71,75,79 \
  --home-court-advantage 2.7 \
  --alt-target-low -250 \
  --alt-target-high -200 \
  --alt-target-mid -225 \
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
- `parameter_1`: picks from parameter 1
- `parameter_2.spread`: spread picks from parameter 2
- `parameter_2.totals`: totals picks from parameter 2
- `parameter_3`: safer alternate-style recommendations

It also includes:
- line release timestamps (`opener`)
- latest line update timestamps
- sportsbook attribution and book IDs
- reasoning fields for each recommendation
- model alignment fields for each pick

## Notes

- Lines and splits can change quickly.
- If a market feed does not provide explicit alternate spreads, parameter 3 will fall back to a moneyline proxy and mark that in the report.
- `--watch` mode is the easiest way to keep the report fresh throughout the day.