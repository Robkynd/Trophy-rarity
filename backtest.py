"""Backtest for the gta6-trophy-rarity-forecast-01 entry (main.py).

Scoring is exactly the rules' pinball loss on the logit scale, mean over
every trophy and quantile; lower is better. Three forecasts are compared:

  * ours      -- main.py (earners-over-tracked cohort model)
  * persist   -- the house persistence baseline (starter/main.py logic)
  * curve     -- the house curve baseline as the rules describe it: per
                 trophy, least squares of logit(rate) on ln(1 + hours since
                 2026-11-19T00:00:00Z) over the last 24 h of reads, read at
                 the target; spread = random-walk spread combined (root-sum-
                 square) with the fit residual; < 3 reads -> persistence.
                 (The rules do not publish the house's exact spread
                 combination; this is a faithful reconstruction.)

Three backtests, each honest about what it can and cannot prove:

  A. Practice cases p00/p01 against RECONSTRUCTED answers. The kit does
     not ship the practice answers, but the rules state the persistence
     baseline's practice numbers (loss ~1.51, typical miss ~0.68 points,
     90% coverage ~0). A uniform dilution y = r/k reproduces those numbers
     at k = 5 (loss 1.498, miss 0.695, coverage 0.000) -- i.e. the 04:30Z
     refresh took N 638 -> 3,868 while earners rose only ~21%. The
     backtest scores all three forecasts on those reconstructed answers,
     with a sensitivity sweep over k in [3, 6]. p01's answer is taken as
     the same refreshed page (no second refresh is recorded between
     04:30Z and 05:00Z).
  B. Real 26-hour backtest: a pseudo-case cut from the launch-night page
     (checkpoint 2026-11-19T03:00Z, target 2026-11-20T01:37Z) scored
     against the LIVE Exophase page fetched 2026-11-20 01:37Z
     (research/live-2026-11-20-0137Z.json). Real data, real dilution +
     progression. CAVEAT: the model's class priors were calibrated on this
     same snapshot, so this test is in-sample for the priors; it validates
     the machinery and the dilution handling, not the prior values.
  C. Synthetic 4.5-day backtest: six cohort worlds ANCHORED to the real
     present -- each world's N(h) and per-class delays are solved so its
     day-1.5 snapshot matches the live Exophase page (N ~17k, Ch1 0.73%,
     fishing 1.89%, most 0.03-0.06%). The FUTURE varies (N_inf, eventual
     fractions, delay scales). The world uses a cohort integral -- different
     functional forms from the model -- so it tests the 4.5-day
     extrapolation machinery. CAVEAT: the worlds share the snapshot as a
     calibration anchor with the model, so this is a machinery test, not
     proof about the real launch; the spread is the protection if the real
     future leaves the tested range.

Run: python3 backtest.py  (from this directory; uses ./main.py as a
subprocess on the practice cases and the synthetic cases).
"""

import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import main as entry

HERE = Path(__file__).resolve().parent
KIT = Path.home() / "workspace/makefun/trophy-rarity-kit/gta6-trophy-rarity-forecast-01-kit"
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
RELEASE = datetime(2026, 11, 19, 0, 0, tzinfo=timezone.utc)


def clamp(x, lo=0.01, hi=99.99):
    return min(hi, max(lo, x))


def logit(p):
    p = clamp(p)
    return math.log(p / (100.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def pinball(y, q, tau):
    d = logit(y) - logit(q)
    return 2.0 * max(tau * d, (tau - 1.0) * d)


def score(forecast_rows, answers):
    """forecast_rows: list of [q05..q95] in trophy order; answers: rates."""
    tot = 0.0
    miss = 0.0
    cov = 0
    n = 0
    for row, y in zip(forecast_rows, answers):
        miss += abs(y - row[2])
        cov += 1 if row[0] <= y <= row[4] else 0
        for q, tau in zip(row, LEVELS):
            tot += pinball(y, q, tau)
            n += 1
    return tot / n, miss / len(answers), cov / len(answers)


def rows_of(forecast):
    by_id = {r["id"]: r["q"] for r in forecast["trophies"]}
    return [by_id[i] for i in sorted(by_id)]


def linfit(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n; my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    sd = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / n)
    return (a, b, n, sd)


def persistence_forecast(case):
    return entry.persistence_case(case, "baseline")


def curve_forecast(case):
    """House curve baseline, reconstructed from the rules' description."""
    at = entry.parse_time(case["at"])
    target = entry.parse_time(case["target"])
    h_target = (target - RELEASE).total_seconds() / 3600.0
    d = case["horizon_minutes"] / 1440.0
    rw = 0.5 * math.sqrt(d)
    recent = [r for r in case["history"]
              if (at - entry.parse_time(r["at"])).total_seconds() <= 24 * 3600]
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        xs, ys = [], []
        for r in recent:
            rate = r["rates"][i] if i < len(r["rates"]) else None
            if rate is None:
                continue
            h = (entry.parse_time(r["at"]) - RELEASE).total_seconds() / 3600.0
            xs.append(math.log(1.0 + h))
            ys.append(logit(rate))
        fit = linfit(xs, ys) if len(xs) >= 3 else None
        if fit is None:
            lg = logit(case["history"][-1]["rates"][i] or 0)
            spread = rw
        else:
            a, b, _, resid = fit
            lg = a + b * math.log(1.0 + h_target)
            spread = math.sqrt(rw * rw + resid * resid)
        rows.append({"id": trophy["id"],
                     "q": [entry.round4(inv_logit(lg + z * spread)) for z in Z]})
    return {"trophies": rows}


def run_main_on(case, tag):
    """Run main.py as a subprocess on a single case; return its forecast."""
    d_in = HERE / f".bt-{tag}-in"
    d_out = HERE / f".bt-{tag}-out"
    d_in.mkdir(exist_ok=True)
    d_out.mkdir(exist_ok=True)
    (d_in / f"{case['id']}.in").write_text(json.dumps(case), encoding="utf-8")
    env = dict(os.environ, MAKEFUN_INPUT_DIR=str(d_in), MAKEFUN_OUTPUT_DIR=str(d_out))
    t0 = time.time()
    r = subprocess.run([sys.executable, str(HERE / "main.py")], env=env,
                       capture_output=True, text=True)
    dt = time.time() - t0
    assert r.returncode == 0, f"main.py failed on {tag}: {r.stderr}"
    out = json.loads((d_out / f"{case['id']}.out").read_text(encoding="utf-8"))
    return out, dt


def check_valid(forecast, case):
    rows = forecast["trophies"]
    assert len(rows) == len(case["trophies"]), "row count"
    ids = sorted(r["id"] for r in rows)
    assert ids == sorted(t["id"] for t in case["trophies"]), "ids"
    for r in rows:
        q = r["q"]
        assert len(q) == 5, "5 quantiles"
        assert all(0.0 <= v <= 100.0 for v in q), "range"
        assert all(q[k] <= q[k + 1] for k in range(4)), "non-decreasing"
    return True


def main():
    print("=" * 78)
    print("A. PRACTICE CASES vs RECONSTRUCTED ANSWERS (uniform dilution y = r/k)")
    print("   k=5 reproduces the rules' stated persistence numbers")
    print("   (loss ~1.51, miss ~0.68, coverage ~0); sweep k in [3, 6].")
    print("=" * 78)
    for name in ("p00", "p01"):
        case = json.loads((KIT / "input" / f"{name}.in").read_text(encoding="utf-8"))
        last_rates = case["history"][-1]["rates"]
        ours, dt = run_main_on(case, name)
        check_valid(ours, case)
        print(f"\n{name}: main.py ran in {dt:.2f}s")
        for k in (3, 4, 5, 6):
            ans = [100.0 if r == 100 else r / k for r in last_rates]
            lo = score(rows_of(ours), ans)
            lp = score(rows_of(persistence_forecast(case)), ans)
            lc = score(rows_of(curve_forecast(case)), ans)
            print(f"  k={k}: ours loss={lo[0]:.4f} miss={lo[1]:.3f} cov={lo[2]:.2f} | "
                  f"persist loss={lp[0]:.4f} miss={lp[1]:.3f} cov={lp[2]:.2f} | "
                  f"curve loss={lc[0]:.4f}")

    print()
    print("=" * 78)
    print("B. REAL 26-HOUR BACKTEST (launch night -> live Exophase 2026-11-20 01:37Z)")
    print("   Pseudo-case: checkpoint 03:00Z Nov 19, target 01:37Z Nov 20.")
    print("   CAVEAT: class priors were calibrated on this snapshot (in-sample).")
    print("=" * 78)
    live = json.loads((HERE / "research" / "live-2026-11-20-0137Z.json").read_text(encoding="utf-8"))
    base = json.loads((KIT / "input" / "p00.in").read_text(encoding="utf-8"))
    case26 = dict(base)
    case26["id"] = "live26h"
    case26["at"] = "2026-11-19T03:00:00Z"
    case26["horizon_minutes"] = 1357
    case26["target"] = "2026-11-20T01:37:00Z"
    ours, dt = run_main_on(case26, "live26h")
    check_valid(ours, case26)
    ans = live["rates"]
    print(f"live26h: main.py ran in {dt:.2f}s; players tracked then ~{live['players_tracked_est']}")
    for label, fc in (("ours", ours),
                      ("persist", persistence_forecast(case26)),
                      ("curve", curve_forecast(case26))):
        s = score(rows_of(fc), ans)
        print(f"  {label:8s} loss={s[0]:.4f} miss={s[1]:.3f} cov={s[2]:.2f}")

    print()
    print("=" * 78)
    print("C. SYNTHETIC 4.5-DAY BACKTEST (6 cohort worlds anchored to the live page)")
    print("   Final-like case: history to day 2.5, answer at day 7.")
    print("   Each world's N(h) and per-class delays are solved so its")
    print("   day-1.5 snapshot matches the live page (N ~17k, Ch1 0.73%,")
    print("   fishing 1.89%, most 0.03-0.06%). The FUTURE varies.")
    print("=" * 78)
    # (N_inf, tau_N h, f_scale, delay_scale). N(h) is a saturating
    # exponential; the pairs are chosen so N(4.5h) ~ 3.9k, N(25.6h) ~ 17k.
    worlds = [
        (65_000, 88, 1.0, 1.0),
        (100_000, 115, 1.0, 1.0),
        (150_000, 150, 1.3, 0.7),
        (80_000, 100, 0.7, 1.3),
        (250_000, 220, 1.0, 0.5),
        (120_000, 130, 0.85, 2.0),
    ]
    # Long-run completion fractions by class (world truth, before f_scale).
    F_INF = {"first": 1.0, "platinum": 0.008, "completion": 0.003,
             "story1": 0.6, "story2": 0.5, "story3": 0.42, "story4": 0.36,
             "story_final": 0.3, "grind": 0.05, "fishing": 0.3, "quick": 0.3, "mid": 0.15}
    trophies = base["trophies"]
    classes = [entry.classify(t) for t in trophies]
    live_rates = live["rates"]
    n_live = live["players_tracked_est"]
    e_live = [r / 100.0 * n_live for r in live_rates]
    e0 = [r / 100.0 * 638 for r in base["history"][-1]["rates"]]

    def cohort_earners(i, h, n_func, f_i, tau_i):
        """Earners of trophy i at hour h: launch-night earners plus the
        cohort integral over players who joined after release."""
        if classes[i] == "first":
            return n_func(h)
        total = e0[i]
        s = 0.0
        while s < h:
            dn = n_func(s + 1.0) - n_func(s)
            d = h - s
            if dn > 0 and d > 0:
                f_delay = 1.0 - math.exp(-((d / tau_i) ** 1.5))
                total += dn * f_i * f_delay
            s += 1.0
        return total

    tot = {"ours": [], "persist": [], "curve": []}
    for wi, (n_inf, tau_n, f_scale, delay_scale) in enumerate(worlds):
        def n_func(h, n_inf=n_inf, tau_n=tau_n):
            if h <= 0:
                return 638.0
            return 638.0 + (n_inf - 638.0) * (1.0 - math.exp(-h / tau_n))

        # Solve each class's delay scale so the world's day-1.5 earners
        # match the live snapshot (bisection on log tau).
        # Solve EACH trophy's delay scale so the world's day-1.5 earners
        # match that trophy's live-snapshot earners (bisection on log tau).
        tau_i = []
        f_i = []
        for i in range(55):
            if classes[i] == "first":
                tau_i.append(0.1)
                f_i.append(1.0)
                continue
            fi = F_INF[classes[i]] * f_scale
            f_i.append(fi)
            target_e = e_live[i]
            # If the trophy already has >= target earners at launch night,
            # its delay is effectively infinite (no new earners yet).
            if e0[i] >= target_e:
                tau_i.append(5000.0)
                continue
            lo, hi = math.log(0.5), math.log(5000.0)
            for _ in range(40):
                mid = (lo + hi) / 2.0
                tau = math.exp(mid)
                e = cohort_earners(i, 25.6, n_func, fi, tau)
                if e < target_e:
                    hi = mid  # delay too long -> fewer earners; shorten it
                else:
                    lo = mid
            tau_i.append(math.exp((lo + hi) / 2.0) * delay_scale)

        n_45 = n_func(4.5)
        n_256 = n_func(25.6)

        refresh_times = []
        t = RELEASE - timedelta(minutes=12)
        while t < RELEASE + timedelta(days=8):
            refresh_times.append(t)
            t += timedelta(hours=4)
        reads = []
        slot = RELEASE - timedelta(minutes=10)
        end = RELEASE + timedelta(days=2, hours=11, minutes=30)
        while slot < end:
            slot += timedelta(minutes=30) if slot < RELEASE + timedelta(hours=4) else timedelta(minutes=10)
            last_ref = max(r for r in refresh_times if r <= slot)
            h_ref = max(0.0, (last_ref - RELEASE).total_seconds() / 3600.0)
            n = n_func(h_ref)
            rates = []
            for i in range(55):
                e = cohort_earners(i, h_ref, n_func, f_i[i], tau_i[i])
                rates.append(round(100.0 * e / n, 2) if n > 0 else 0.0)
            reads.append({"slot": slot.strftime("%Y-%m-%dT%H:%M:%SZ"),
                          "at": slot.strftime("%Y-%m-%dT%H:%M:%SZ"),
                          "players_tracked": int(round(n)),
                          "trophies_earned": int(round(sum(
                              cohort_earners(i, h_ref, n_func, f_i[i], tau_i[i])
                              for i in range(55)))),
                          "rates": rates})
        case = dict(base)
        case["id"] = f"syn{wi}"
        case["at"] = "2026-11-21T11:30:00Z"
        case["horizon_minutes"] = 6510
        case["target"] = "2026-11-26T00:00:00Z"
        case["history"] = reads
        ans_ref = max(r for r in refresh_times if r <= RELEASE + timedelta(days=7, minutes=30))
        h_ans = (ans_ref - RELEASE).total_seconds() / 3600.0
        n_ans = n_func(h_ans)
        ans = [round(100.0 * cohort_earners(i, h_ans, n_func, f_i[i], tau_i[i]) / n_ans, 2)
               for i in range(55)]
        ours, _ = run_main_on(case, f"syn{wi}")
        check_valid(ours, case)
        print(f"  world {wi} (N_inf={n_inf/1000:.0f}k, tau_N={tau_n}h, f x{f_scale}, "
              f"delay x{delay_scale}): N(4.5h)={n_45:,.0f} N(25.6h)={n_256:,.0f} "
              f"N(7d)={n_ans:,.0f}; day-7 Ch1={ans[2]}% Ch2={ans[3]}% fish={ans[19]}% plat={ans[0]}%")
        line = "    "
        for label, fc in (("ours", ours),
                          ("persist", persistence_forecast(case)),
                          ("curve", curve_forecast(case))):
            s = score(rows_of(fc), ans)
            tot[label].append(s[0])
            line += f"{label}={s[0]:.4f} (cov {s[2]:.2f})  "
        print(line)
    print("  MEAN over 6 worlds: " +
          "  ".join(f"{k}={sum(v)/len(v):.4f}" for k, v in tot.items()))

    print()
    print("=" * 78)
    print("D. VALIDITY / DETERMINISM (practice cases, two runs each)")
    print("=" * 78)
    for name in ("p00", "p01"):
        case = json.loads((KIT / "input" / f"{name}.in").read_text(encoding="utf-8"))
        outs = []
        for rep in range(2):
            d_in = HERE / f".bt-det-{name}-in"
            d_out = HERE / f".bt-det-{name}-out{rep}"
            d_in.mkdir(exist_ok=True)
            d_out.mkdir(exist_ok=True)
            (d_in / f"{name}.in").write_text(json.dumps(case), encoding="utf-8")
            env = dict(os.environ, MAKEFUN_INPUT_DIR=str(d_in), MAKEFUN_OUTPUT_DIR=str(d_out))
            t0 = time.time()
            r = subprocess.run([sys.executable, str(HERE / "main.py")], env=env,
                               capture_output=True, text=True)
            dt = time.time() - t0
            assert r.returncode == 0, r.stderr
            raw = (d_out / f"{name}.out").read_bytes()
            outs.append(raw)
            print(f"  {name} run {rep}: {len(raw)} bytes, {dt:.2f}s")
        assert outs[0] == outs[1], "non-deterministic output!"
        print(f"  {name}: two runs byte-identical: True")


if __name__ == "__main__":
    main()
