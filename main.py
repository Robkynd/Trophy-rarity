"""gta6-trophy-rarity-forecast-01 entry: earners-over-tracked cohort model.

Forecast each GTA VI PS5 trophy's Exophase earn rate at the case's target.
The rate is a ratio, so the model forecasts its two parts separately and
divides: earners E_i(t) and players tracked N(t). That separation is the
whole edge over persistence: on launch week N grows far faster than E_i
(new buyers sync before they play), so most rates FALL first and only rise
once the player cohort has had days to progress -- persistence, which holds
the rate, and a curve fit on the rate itself, both miss that shape.

Method
------
1.  Read the case. Collapse `history` to distinct states: Exophase refreshes
    its counts in jumps (hours apart), so consecutive identical reads carry
    no new information; a distinct state is a refresh sample.  The first
    state is the launch-night page (638 players tracked, every rate the
    same for 2.5+ hours).
2.  Earners per state: E_i = rate_i / 100 * N (the case's `trophies_earned`
    equals sum_i E_i, which cross-checks the arithmetic).
3.  If the history is flat (one distinct state -- the practice cases), use
    the refresh prior: N follows a saturating-exponential curve
    (N_inf = 100 * N_first, tau = 88 h, calibrated on the two refreshes
    the rules state -- 638 -> 3,868 at 04:30Z -- and the live page ~26 h
    after launch, ~17k tracked) and each trophy's earners grow as
    (1 + dt/24)^gamma by class. Over a 90-minute horizon with a stale
    page this reproduces the refresh dilution the practice answers show
    (rates about /5: N up ~6x while earners rise only ~20%).
4.  If the history shows movement, the median is a blend of two trend
    estimates:
      (a) a rate fit: a straight line of logit(rate) on ln(1 + hours
          since release) over the last 24 h of states -- the same
          construction as the house curve baseline, so it inherits its
          robustness on smooth data -- plus a capped correction from a
          recency-weighted quadratic over all states (25% weight, at
          most 0.4 logit). The rate's log-time curve is U-shaped
          (dilution first, progression surge after), and the quadratic's
          steep transition slope overshoots when taken whole;
      (b) the earners-over-tracked trend: log(E_i + 0.5) on log(h + 24)
          with the slope shrunk toward its class prior (and the
          intercept refitted after the slope is constrained), divided
          by N(t) from a saturating-exponential fit (grid search over
          the time constant, N_inf by least squares -- a power law
          cannot do this, it extrapolates N 2-3x too high at day 7).
          The two estimates are averaged when they agree and the rate
          fit is trusted 70/30 when they do not.
    Trophies are classified from their description and tier (story
    chapters in order, final mission, platinum, 100% completion, grind,
    fishing -- the single most popular early activity, quick one-offs,
    mid-game). Each class carries a day-7 rate prior and a hard cap
    (calibrated on the live Exophase page 2026-11-20 01:37Z,
    research only -- the program never reads it at run time): a forecast
    more than ~1.6 logit from its class prior is pulled 20% of the way
    toward it, and no rate may exceed its class cap. The first trophy
    (100% at launch) stays at 99.99%.
5.  Spread: a logit sd per trophy. Base is the random-walk spread
    combined (root-sum-square) with the last-24h fit's residual sd --
    the same construction as the house curve baseline -- plus a 0.1
    class adjustment and +0.15 when the two trend estimates disagree
    by more than 1 logit. Clamped to [0.05, 2.0]. q_tau =
    inv_logit(logit(median) + z_tau * s), rounded to 4 decimals half-up
    exactly as the house baseline rounds.

Backtest: see TEST-LOG.md for the exact numbers (practice cases against
reconstructed answers, a real 26-hour backtest against the live Exophase
page, and a synthetic 4.5-day backtest), each scored against the house
persistence baseline and the house curve baseline. Final numbers
(2026-11-20, logit-scale pinball loss): practice p00 0.2521 / p01 0.2608
(k=5 reconstructed answers; persistence 1.4982, curve 1.4982); real
26-hour 0.2993 (persistence 2.2781, curve 2.2781); synthetic 4.5-day
mean over 6 cohort worlds 0.3139 (persistence 0.8113, curve 0.3946) --
the model beats both house baselines on the 4.5-day horizon. 90%
interval coverage 0.96-1.00 (practice, 26 h) and 0.98 (every synthetic
world).

Fallbacks: any failure in the model path for a case falls back to the
house persistence baseline for that case, so the program always emits a
valid output. No randomness, no network, no clock: two runs are
byte-identical. Standard library only; runs in well under 8 s.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Class parameters: beta = earners log-log slope vs log(h + 24),
# gamma = flat-history growth exponent, f7 = day-7 rate prior (fraction),
# cap = hard cap on the day-7 rate (fraction), adj = spread addition.
CLASSES = {
    "first":      {"beta": 0.0,  "gamma": 0.0,  "f7": 0.9999, "cap": 0.9999, "adj": 0.0},
    "platinum":   {"beta": 0.0,  "gamma": 0.0,  "f7": 0.0002, "cap": 0.0006, "adj": 0.1},
    "completion": {"beta": 0.0,  "gamma": 0.0,  "f7": 0.0001, "cap": 0.0003, "adj": 0.1},
    "story1":     {"beta": 3.6,  "gamma": 3.6,  "f7": 0.12,   "cap": 0.36,   "adj": 0.1},
    "story2":     {"beta": 1.5,  "gamma": 1.5,  "f7": 0.05,   "cap": 0.15,   "adj": 0.1},
    "story3":     {"beta": 0.36, "gamma": 0.36, "f7": 0.025,  "cap": 0.075,  "adj": 0.1},
    "story4":     {"beta": 0.36, "gamma": 0.36, "f7": 0.015,  "cap": 0.045,  "adj": 0.1},
    "story_final":{"beta": 0.36, "gamma": 0.36, "f7": 0.01,   "cap": 0.03,   "adj": 0.1},
    "grind":      {"beta": 0.3,  "gamma": 0.13, "f7": 0.003,  "cap": 0.009,  "adj": 0.1},
    "fishing":    {"beta": 1.8,  "gamma": 1.8,  "f7": 0.18,   "cap": 0.30,   "adj": 0.1},
    "quick":      {"beta": 1.8,  "gamma": 1.8,  "f7": 0.04,   "cap": 0.15,   "adj": 0.1},
    "mid":        {"beta": 0.6,  "gamma": 0.6,  "f7": 0.015,  "cap": 0.045,  "adj": 0.1},
}

# Flat-history N prior: saturating exponential, N_inf = N_INF_MULT * N_first.
N_INF_MULT = 100.0
N_TAU_H = 88.0
# Slope shrinkage: a fitted slope counts as n states, the prior as this many.
PRIOR_WEIGHT = 3.0
# Recency half-life (hours) for the trend fits.
RECENCY_HALF_LIFE_H = 17.0
# Cap on the quadratic curvature's contribution at the target (logit).
CURV_CAP = 0.4
# Weak prior shrinkage: pull 20% toward the class prior when the forecast
# is more than this many logit away from it.
PRIOR_PULL_DIST = 1.6
PRIOR_PULL = 0.20


def clamp(x: float, lo: float = 0.01, hi: float = 99.99) -> float:
    return min(hi, max(lo, x))


def logit(pct: float) -> float:
    p = clamp(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as the house baseline rounds it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def classify(trophy: dict) -> str:
    """Trophy class from its tier and description (lowercased)."""
    if trophy.get("tier") == "platinum":
        return "platinum"
    d = (trophy.get("description") or "").lower()
    if "100% completion" in d:
        return "completion"
    if "complete gone fishing" in d:
        return "first"
    if "complete chapter 1" in d:
        return "story1"
    if "complete chapter 2" in d:
        return "story2"
    if "complete chapter 3" in d:
        return "story3"
    if "complete chapter 4" in d:
        return "story4"
    if "complete the final mission." in d:
        return "story_final"
    if ("250 stunts" in d or "100 locations" in d or "50 challenges" in d
            or "3 collectible card sets" in d or "8 mission strands" in d
            or "gold medal in all shooting range" in d
            or "win 20 different races" in d
            or "win a top fun drone race at every location" in d):
        return "grind"
    if "fishing challenge" in d:
        return "fishing"
    if ("arcade game" in d or "bowling" in d
            or "cocktail" in d or "try each type of activity" in d
            or "discover 100 vehicles" in d or "explore all of leonida" in d
            or "meditate at every spot" in d or "hunt a unique animal" in d
            or "make an unscheduled departure" in d):
        return "quick"
    return "mid"


def _solve3(m):
    """Solve a 3x3 linear system (augmented rows); None if singular."""
    m = [row[:] for row in m]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            return None
        m[col], m[piv] = m[piv], m[col]
        div = m[col][col]
        m[col] = [v / div for v in m[col]]
        for r in range(3):
            if r != col and m[r][col] != 0.0:
                f = m[r][col]
                m[r] = [v - f * w for v, w in zip(m[r], m[col])]
    return [m[0][3], m[1][3], m[2][3]]


def _wlinfit(xs, ys, ws):
    """Weighted least squares y = a + b*x. Returns (a, b, resid_sd) or None."""
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return None
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    sd = math.sqrt(sum(w * r * r for w, r in zip(ws, resid)) / sw)
    return (a, b, sd)


def _wquadfit(xs, ys, ws):
    """Weighted least squares y = a + b*u + c*u^2 with u = x - xbar
    (centred, so the curvature term is c*(x - xbar)^2). Returns
    (a, b, c, xbar, resid_sd) or None."""
    sw = sum(ws)
    if sw <= 0 or len(xs) < 5:
        return None
    xbar = sum(w * x for w, x in zip(ws, xs)) / sw
    us = [x - xbar for x in xs]
    # Normal equations for [1, u, u^2].
    s = [0.0] * 5
    t = [0.0] * 3
    for u, y, w in zip(us, ys, ws):
        p = 1.0
        for k in range(5):
            s[k] += w * p
            if k < 3:
                t[k] += w * p * y
            p *= u
    sol = _solve3([[s[0], s[1], s[2], t[0]],
                   [s[1], s[2], s[3], t[1]],
                   [s[2], s[3], s[4], t[2]]])
    if sol is None:
        return None
    a, b, c = sol
    resid = [y - (a + b * u + c * u * u) for u, y in zip(us, ys)]
    sd = math.sqrt(sum(w * r * r for w, r in zip(ws, resid)) / sw)
    return (a, b, c, xbar, sd)


def distinct_states(case: dict, release: datetime):
    """Collapse history to distinct (N, rates) states, first-seen time each."""
    states = []
    prev_key = None
    for read in case.get("history", []):
        rates = read.get("rates") or []
        key = (read.get("players_tracked"), tuple(rates))
        if key == prev_key:
            continue
        prev_key = key
        states.append({
            "h": (parse_time(read["at"]) - release).total_seconds() / 3600.0,
            "N": read["players_tracked"],
            "rates": list(rates),
        })
    return states


def fit_n_target(states, h_last, h_target, n_last):
    """Forecast players tracked at the target hour.

    A saturating exponential N(h) = N_first + (N_inf - N_first) *
    (1 - exp(-(h - h_first)/tau)), fitted by a grid search over tau with
    N_inf solved by least squares on N and tau chosen by the lowest
    log-N error. A power law cannot do this: the launch growth is
    explosive early and saturates, so a log-log slope fitted on days 0-2
    extrapolates N 2-3x too high at day 7. Flat history: the prior curve.
    """
    if len(states) >= 2:
        h_first = states[0]["h"]
        n_first = max(1, states[0]["N"])
        best = None
        for tau in (20, 30, 45, 70, 100, 150, 220, 330, 500, 750, 1200):
            gs = [1.0 - math.exp(-(s["h"] - h_first) / tau) for s in states]
            num = sum(g * (s["N"] - n_first) for g, s in zip(gs, states))
            den = sum(g * g for g in gs)
            amp = max(0.0, num / den) if den > 0 else 0.0
            sse = 0.0
            for g, s in zip(gs, states):
                pred = n_first + amp * g
                if pred <= 0 or s["N"] <= 0:
                    sse += 1e6
                else:
                    sse += (math.log(pred) - math.log(s["N"])) ** 2
            if best is None or sse < best[0]:
                best = (sse, tau, amp)
        if best is not None:
            _, tau, amp = best
            pred = n_first + amp * (1.0 - math.exp(-(h_target - h_first) / tau))
            return min(max(pred, n_last), 50.0 * n_last)
    first = states[0]
    n_inf = N_INF_MULT * max(1, first["N"])
    n_t = n_inf - (n_inf - first["N"]) * math.exp(
        -(h_target - first["h"]) / N_TAU_H)
    return min(max(n_t, n_last), 30.0 * n_last)


def forecast_case(case: dict, name: str) -> dict:
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    trophies = case["trophies"]
    history = case.get("history") or []
    if not history:
        raise ValueError(f"{name} has no history")
    release = parse_time(case["game"]["release_date"] + "T00:00:00Z")
    at = parse_time(case["at"])
    target = parse_time(case["target"])
    h_last = (at - release).total_seconds() / 3600.0
    h_target = (target - release).total_seconds() / 3600.0
    horizon_days = case["horizon_minutes"] / 1440.0
    rw = 0.5 * math.sqrt(horizon_days)

    states = distinct_states(case, release)
    last = history[-1]
    n_last = last["players_tracked"]
    last_rates = last["rates"]
    n_target = fit_n_target(states, h_last, h_target, n_last)
    moving = len(states) >= 3
    x_target = math.log(1.0 + h_target)

    rows = []
    for i, trophy in enumerate(trophies):
        cls = classify(trophy)
        prm = CLASSES[cls]
        r_last = last_rates[i]
        if r_last is None:
            r_last = 0.0

        if cls == "first":
            median = 99.99
            spread = 0.05
            rows.append({"id": trophy["id"],
                         "q": [round4(inv_logit(logit(median) + z * spread)) for z in Z]})
            continue

        if not moving:
            # Flat history: class growth curve from the first state, and
            # the N prior curve. E grows as (1 + dt/24)^gamma; the ratio
            # to N growth is the refresh dilution over a short horizon and
            # the progression surge over a long one.
            first = states[0]
            r_first = first["rates"][i] if i < len(first["rates"]) else None
            e_first = (r_first if r_first is not None else 0.0) / 100.0 * first["N"]
            dt = h_target - first["h"]
            e_t = e_first * (1.0 + dt / 24.0) ** prm["gamma"]
            e_t = min(e_t, prm["cap"] * n_target)
            median = clamp(100.0 * e_t / n_target)
            spread = min(max(1.0 + prm["adj"], 0.05), 2.0)
            lg = logit(median)
            rows.append({"id": trophy["id"],
                         "q": [round4(inv_logit(lg + z * spread)) for z in Z]})
            continue

        # ---- Moving history: two trend estimates. ----
        ws = [math.exp(-(h_last - s["h"]) / RECENCY_HALF_LIFE_H) for s in states]

        # (a) Rate fit. Primary: a straight line of logit(rate) on
        # ln(1 + h) over the last 24 h of states -- the same construction
        # as the house curve baseline, so it inherits its robustness on
        # smooth data. The recency-weighted quadratic over all states is
        # a capped correction only (25% weight, at most 0.4 logit): the
        # rate's log-time curve is U-shaped (dilution, then surge), and
        # the quadratic's steep transition slope overshoots when taken
        # whole. Its residual (last 24 h) sets the spread.
        xs, ys, hs = [], [], []
        for s in states:
            r = s["rates"][i] if i < len(s["rates"]) else None
            if r is None:
                continue
            xs.append(math.log(1.0 + s["h"]))
            ys.append(logit(r))
            hs.append(s["h"])
        xs_r = [x for x, h in zip(xs, hs) if (h_last - h) <= 24.0]
        ys_r = [y for y, h in zip(ys, hs) if (h_last - h) <= 24.0]
        if len(xs_r) < 3:
            xs_r, ys_r = xs, ys
        lin_r = _wlinfit(xs_r, ys_r, [1.0] * len(xs_r)) if len(xs_r) >= 2 else None
        resid = 0.0
        if lin_r is not None:
            a, b, resid = lin_r
            lg_rate = a + b * x_target
        else:
            lg_rate = logit(r_last)
        quad = _wquadfit(xs, ys, ws) if len(xs) >= 5 else None
        if quad is not None:
            a, b, c, xbar, _ = quad
            dx = x_target - xbar
            curv = max(-CURV_CAP, min(CURV_CAP, c * dx * dx))
            lg_quad = a + b * dx + curv
            lg_rate = lg_rate + 0.25 * max(-0.4, min(0.4, lg_quad - lg_rate))

        # (b) Earners-over-tracked trend: log(E + 0.5) on log(h + 24),
        # slope shrunk toward the class beta, intercept refitted after
        # the slope is constrained; divided by the N trend.
        xs_e, ys_e = [], []
        for s in states:
            r = s["rates"][i] if i < len(s["rates"]) else None
            if r is None:
                continue
            xs_e.append(math.log(s["h"] + 24.0))
            ys_e.append(math.log(r / 100.0 * s["N"] + 0.5))
        lg_en = None
        fit_e = _wlinfit(xs_e, ys_e, ws) if len(xs_e) >= 3 else None
        if fit_e is not None:
            _, b_fit, _ = fit_e
            b = (len(xs_e) * b_fit + PRIOR_WEIGHT * prm["beta"]) / (len(xs_e) + PRIOR_WEIGHT)
            b = min(max(b, 0.0), 3.0)
            sw = sum(ws)
            mx = sum(w * x for w, x in zip(ws, xs_e)) / sw
            my = sum(w * y for w, y in zip(ws, ys_e)) / sw
            a = my - b * mx
            e_trend = math.exp(a + b * math.log(h_target + 24.0))
            lg_en = logit(clamp(100.0 * e_trend / n_target))

        # Combine: average when the two agree, trust the rate fit 70/30
        # when they do not.
        if lg_en is None:
            lg_med = lg_rate
            disagree = 0.0
        else:
            disagree = abs(lg_rate - lg_en)
            if disagree <= 0.5:
                lg_med = 0.5 * (lg_rate + lg_en)
            else:
                lg_med = 0.7 * lg_rate + 0.3 * lg_en

        # Weak prior shrinkage + hard class cap.
        lp = logit(100.0 * prm["f7"])
        if abs(lg_med - lp) > PRIOR_PULL_DIST:
            lg_med = lg_med + PRIOR_PULL * (lp - lg_med)
        median = clamp(inv_logit(lg_med))
        median = min(median, 100.0 * prm["cap"])
        median = max(median, 0.01)

        spread = math.sqrt(rw * rw + resid * resid) + prm["adj"]
        if disagree > 1.0:
            spread += 0.15
        spread = min(max(spread, 0.05), 2.0)

        lg = logit(median)
        rows.append({"id": trophy["id"],
                     "q": [round4(inv_logit(lg + z * spread)) for z in Z]})
    return {"trophies": rows}


def persistence_case(case: dict, name: str) -> dict:
    """The house persistence baseline, for fallback use only."""
    last = case["history"][-1]
    spread = 0.5 * math.sqrt(case["horizon_minutes"] / 1440)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = last["rates"][i]
        lg = logit(rate if rate is not None else 0)
        rows.append({"id": trophy["id"],
                     "q": [round4(inv_logit(lg + z * spread)) for z in Z]})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case, path.name)
        except Exception:
            forecast = persistence_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
