# TEST-LOG — gta6-trophy-rarity-forecast-01

Entry: `main.py` (Python 3.13, stdlib only, deterministic). Scorer:
mean pinball loss on the **logit** scale, lower is better, exactly as the
rules score it. All commands run from
`~/workspace/makefun/gta6-trophy-rarity-forecast/` on 2026-11-20.

## Commands

```sh
python3 backtest.py            # full backtest (practice + 26 h + synthetic + validity)
```

`backtest.py` runs `main.py` as a subprocess on every case
(`MAKEFUN_INPUT_DIR` / `MAKEFUN_OUTPUT_DIR`), checks output validity
(55 rows, 5 quantiles each, non-decreasing, in range, <= 65,536 bytes)
and determinism (two runs byte-identical), and scores the model against
the house persistence baseline (starter `main.py`) and the house curve
baseline (reconstructed from the rules: straight line of logit(rate) on
`ln(1 + h)` over the case's last 24 h of reads, spread = root-sum-square
of the random-walk spread and the fit residual; < 3 reads -> persistence).

## A. Practice cases vs reconstructed answers (uniform dilution y = r/k)

The kit ships no practice answers. A uniform dilution `y = r/k`
reproduces the rules' stated persistence numbers (loss ~1.51, miss
~0.68, coverage ~0) at **k = 5** (computed: loss 1.4982, miss 0.695,
coverage 0.00), i.e. the 04:30Z refresh took N 638 -> 3,868 while
earners rose only ~21%. The sweep over k = 3..6 tests sensitivity to
that reconstruction.

| case | k | ours (loss / miss / cov) | persistence | curve |
|---|---:|---:|---:|---:|
| p00 | 3 | 0.3486 / 0.124 / 0.96 | 0.9955 | 0.9955 |
| p00 | 4 | 0.2883 / 0.055 / 0.98 | 1.2786 | 1.2786 |
| p00 | 5 | **0.2521 / 0.023 / 0.98** | 1.4982 | 1.4982 |
| p00 | 6 | 0.2512 / 0.025 / 0.98 | 1.6775 | 1.6775 |
| p01 | 3 | 0.3633 / 0.135 / 0.96 | 0.9955 | 0.9955 |
| p01 | 4 | 0.3013 / 0.065 / 0.98 | 1.2786 | 1.2786 |
| p01 | 5 | **0.2608 / 0.029 / 0.98** | 1.4982 | 1.4982 |
| p01 | 6 | 0.2431 / 0.017 / 0.98 | 1.6775 | 1.6775 |

`main.py` ran in 0.05 s per case. The model crushes persistence on the
practice across the whole k range (the practice horizon is only 90 min,
so this validates the machinery — the flat-history refresh prior — not
the long-horizon dynamics).

## B. Real 26-hour backtest (launch night -> live Exophase 2026-11-20 01:37Z)

Pseudo-case: checkpoint 2026-11-19T03:00Z, target 2026-11-20T01:37Z,
scored against the live Exophase page fetched that night (rates ~26 h
after launch; players tracked estimated 14k–20k, central ~17k).
**CAVEAT: the class priors were calibrated on this snapshot, so it is
in-sample for the priors** — it validates the machinery and the dilution
handling, not the prior values.

| model | loss | miss | coverage |
|---|---:|---:|---:|
| ours | **0.2993** | 0.053 | 0.98 |
| persistence | 2.2781 | 0.757 | 0.05 |
| curve | 2.2781 | 0.757 | 0.05 |

`main.py` ran in 0.05 s. Core dynamic confirmed: rates FELL from launch
night because tracked players grew ~27x (638 -> ~17k) while earners grew
only ~1–22x depending on the trophy.

## C. Synthetic 4.5-day backtest (6 cohort worlds anchored to the live page)

Final-like case: history to day 2.5 (reads every 10 min, values changing
only at 4-hourly Exophase refreshes), answer at day 7 (target
2026-11-26 00:00 UTC). Each world's `N(h)` (saturating exponential) and
per-trophy delays are solved so its day-1.5 snapshot matches the live
page (N ~17k, Ch1 0.73%, fishing 1.89%, most 0.03–0.06%); the FUTURE
varies (N_inf 65k–250k, tau_N 88–220 h, f-scale 0.7–1.3, delay-scale
0.5–2.0). Earners follow a cohort integral: players who join at hour s
earn trophy i after a Weibull delay. Day-7 answers across worlds:
Ch1 3.62–17.2%, Ch2 0.55–3.03%, fishing 8.07–23.14%, platinum 0.04–0.17%.

| world | ours (cov) | persistence | curve |
|---|---:|---:|---:|
| 0 (N_inf=65k, tau_N=88h, f x1.0, delay x1.0) | **0.3357** (0.98) | 0.8679 | 0.4446 |
| 1 (N_inf=100k, tau_N=115h, f x1.0, delay x1.0) | **0.3016** (0.98) | 0.8311 | 0.3060 |
| 2 (N_inf=150k, tau_N=150h, f x1.3, delay x0.7) | **0.3241** (0.98) | 0.8937 | 0.4347 |
| 3 (N_inf=80k, tau_N=100h, f x0.7, delay x1.3) | **0.3311** (0.98) | 0.7854 | 0.4298 |
| 4 (N_inf=250k, tau_N=220h, f x1.0, delay x0.5) | 0.3068 (0.98) | 0.8668 | **0.2820** |
| 5 (N_inf=120k, tau_N=130h, f x0.85, delay x2.0) | **0.2840** (0.98) | 0.6232 | 0.4709 |
| **MEAN** | **0.3139** | 0.8113 | 0.3946 |

The model beats persistence in all 6 worlds and the curve in 5 of 6
(loses only world 4, the biggest/slowest world, where the curve's linear
extrapolation is nearly perfect: 0.3068 vs 0.2820).

## D. Validity / determinism (practice cases, two runs each)

| case | run 0 | run 1 | byte-identical |
|---|---:|---:|---|
| p00 | 2,790 bytes, 0.06 s | 2,790 bytes, 0.05 s | True |
| p01 | 2,748 bytes, 0.07 s | 2,748 bytes, 0.07 s | True |

55 rows, 5 quantiles each, non-decreasing, in [0, 100], well under
65,536 bytes and the 8 s limit.

## Model evolution (what was tried, what was rejected)

- **v1 — earners-over-tracked with a power-law N fit and a slope clamp
  without an intercept refit.** Synthetic mean 0.6043 vs curve 0.3946 —
  LOST. Diagnosis (per-trophy, worlds 1 and 4): the E/N trend predicted
  ~0.01–0.02% for the big movers (truth 10–20%) because clamping the
  fitted slope to 2.5 while keeping its intercept is not a line through
  the data, and the quadratic N fit blew up so N_target was clamped to
  N_last. The 0.6/0.4 blend then dragged the decent class prior down
  toward the broken trend.
- **v2 — quadratic rate fit as primary, E/N as second estimate,
  spread 1.25–1.55.** Synthetic mean 0.3952 vs curve 0.3946 — a TIE
  (4 worlds won, 2 lost by more). The quadratic's steep transition slope
  overshoots (Ch1 rate-fit 19.88% vs truth 10.39% in world 1) and the
  wide spread cost ~0.05–0.1 per item when the median was near the truth.
- **v3 (shipped) — last-24h linear rate fit as primary + capped
  quadratic correction, E/N trend with a saturating-exponential N fit
  (grid search over tau, N_inf by least squares), fishing as its own
  class (f7 18%, cap 30% — the live page shows it at 1.89%, 5x any other
  quick trophy), spread = sqrt(rw^2 + recent-resid^2) + 0.1 (+0.15 on
  disagreement).** Synthetic mean **0.3139** vs curve 0.3946 — WINS.
  Practice and 26-hour numbers also improved (p00 0.2801 -> 0.2521,
  p01 0.2888 -> 0.2608, 26 h 0.3182 -> 0.2993), so the gains are not
  synthetic-world overfitting.

## Verdict

The entry beats the house persistence baseline everywhere (practice
0.25 vs 1.50, 26 h 0.30 vs 2.28, synthetic mean 0.31 vs 0.81) and beats
the house curve baseline on the 4.5-day horizon that decides the prize
(synthetic mean 0.3139 vs 0.3946, 5 of 6 worlds). It is the strongest
entry built in this session.

## Honest caveats

1. **The 4.5-day extrapolation is the whole risk.** The practice cases
   cover 90 minutes; the final horizon is ~4.5 days. The synthetic
   worlds are the only 4.5-day test, and they are the author's own
   construction (cohort integral + saturating N, anchored to the live
   page) — they validate the machinery and the dilution handling, not
   the real world's future. If the real launch week behaves very
   differently from a smooth cohort model (e.g. a second launch spike,
   a platform-wide sync outage, or Exophase changing its refresh
   cadence), the model can miss.
2. **The house AI model (superbot) is untested.** The prize requires
   beating the BEST house entry, which includes a house AI model built
   once for the final at 2026-11-21 09:30Z. Its loss is unknown; this
   entry beats the two house baselines that can be reconstructed, not
   necessarily the AI model.
3. **Class priors are calibrated on one live snapshot** (2026-11-20
   01:37Z, ~26 h after launch). They are research-only inputs, but if
   the trophy difficulty ordering shifts as more of the player base
   reaches the late game, the priors (and caps) could bind wrongly —
   most visibly the fishing cap (30%) and the story-chapter caps.
4. **Practice answers are reconstructed, not shipped** (uniform
   dilution, k = 5 chosen to reproduce the rules' stated persistence
   numbers). The practice verdict is robust across k = 3..6, but it is
   not a ground-truth score.
