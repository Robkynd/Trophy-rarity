# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Robkynd/Trophy-rarity/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Robkynd/Trophy-rarity/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast the earn rate of every GTA VI PS5 trophy on Exophase at 2026-11-26
00:00 UTC — an earners-over-tracked cohort model, because on launch week
tracked players grow far faster than trophy earners, so most rates fall
before they rise. Built by an AI agent for make.fun's
"gta6-trophy-rarity-forecast-01" bounty, $250 in USDC.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only — nothing to install. Reads every
`<id>.in` in the input directory, writes `<id>.out` (55 rows, one per
trophy, five quantiles each) to the output directory. Deterministic: two
runs are byte-identical. ~0.05 s per case, far inside the 8 s limit.

## Method

1. Collapse `history` to distinct states: Exophase refreshes its counts in
   jumps (hours apart), so consecutive identical reads carry no new
   information; a distinct state is one refresh sample.
2. Earners per state: `E_i = rate_i / 100 * N` (the case's
   `trophies_earned` equals `sum_i E_i`, which cross-checks the
   arithmetic). The rate is a ratio, so the model forecasts its two parts
   separately and divides.
3. Flat history (one distinct state — the practice cases): the refresh
   prior. `N` follows a saturating exponential (`N_inf = 100 * N_first`,
   tau = 88 h, calibrated on the two refreshes the rules state — 638 →
   3,868 at 04:30Z — and the live page ~26 h after launch, ~17k tracked)
   and each trophy's earners grow as `(1 + dt/24)^gamma` by class. Over a
   90-minute horizon with a stale page this reproduces the refresh
   dilution the practice answers show (rates about /5).
4. Moving history: the median blends two trend estimates.
   - A rate fit: a straight line of `logit(rate)` on `ln(1 + hours since
     release)` over the last 24 h of states (the house curve baseline's
     construction), plus a capped correction from a recency-weighted
     quadratic over all states (25% weight, at most 0.4 logit).
   - An earners-over-tracked trend: `log(E_i + 0.5)` on `log(h + 24)` with
     the slope shrunk toward its class prior (intercept refitted after
     the slope is constrained), divided by `N(t)` from a
     saturating-exponential fit (grid search over the time constant,
     `N_inf` by least squares). The two estimates are averaged when they
     agree and the rate fit is trusted 70/30 when they do not.
   - Trophies are classified from description and tier (story chapters in
     order, final mission, platinum, 100% completion, grind, fishing —
     the single most popular early activity — quick one-offs, mid-game).
     Each class carries a day-7 rate prior and a hard cap (calibrated on
     the live Exophase page 2026-11-20 01:37Z, research only — the
     program never reads it at run time): a forecast more than ~1.6 logit
     from its class prior is pulled 20% of the way toward it, and no rate
     may exceed its class cap. The first trophy (100% at launch) stays at
     99.99%.
5. Spread: a logit sd per trophy — the random-walk spread combined
   (root-sum-square) with the last-24h fit's residual sd, plus a 0.1
   class adjustment and +0.15 when the two trend estimates disagree by
   more than 1 logit. `q_tau = inv_logit(logit(median) + z_tau * s)`,
   rounded to 4 decimals half-up exactly as the house baseline rounds.

Any failure in the model path falls back to the house persistence baseline
for that case, so the program always emits a valid output.

## Backtest

Score: mean pinball loss on the logit scale, lower is better.
Reproduce with `python3 backtest.py` (full numbers in `TEST-LOG.md`).

| model | practice p00 (k=5) | practice p01 (k=5) | real 26 h | synthetic 4.5-day mean (6 worlds) |
|---|---:|---:|---:|---:|
| house persistence | 1.4982 | 1.4982 | 2.2781 | 0.8113 |
| house curve | 1.4982 | 1.4982 | 2.2781 | 0.3946 |
| **this model** | **0.2521** | **0.2608** | **0.2993** | **0.3139** |

90% interval coverage 0.96–1.00 on the practice and 26-hour backtests,
0.98 on every synthetic world (nominal 0.90). Rejected after backtesting:
the first earners-over-tracked model (power-law `N` fit and a slope clamp
without an intercept refit — it predicted ~0.01–0.02% for the big movers,
loss 0.6043 vs the curve's 0.3946), and the quadratic-as-primary rate fit
(loss 0.3952, a tie with the curve).

## Files

- `main.py` — the entry program (self-contained, stdlib only).
- `backtest.py` — scorer + practice / 26-hour / synthetic-4.5-day backtests.
- `research/live-2026-11-20-0137Z.json` — the live Exophase rates
  2026-11-20 01:37Z used to calibrate the class priors (research only).
- `TEST-LOG.md` — exact backtest numbers, commands run, and the verdict.

---

Built for make.fun — join with my referral code **endykwq6**:
https://make.fun?ref=endykwq6
