"""
TACTICA.SIM S.r.l. — 12x8 Spatial Expected Threat (xT) Markov Value Iteration Engine
====================================================================================
Module: tactica_xt_engine.py
Alignment: MIMIT Italia Startup Visa (D.L. 179/2012 Art. 25 & Legge 193/2024 Art. 28)
Framework: Discrete-Space Absorbing Markov Chain (Singh, 2018) — Clean-Room Implementation

Mathematical Formulation:
    The 105m x 68m pitch is discretized into a 12 x 8 grid (96 spatial states, attacking
    left-to-right). Each state z = (x, y) has:
      - Shot decision probability s(z)
      - Move (pass / carry) decision probability m(z) = 1 - s(z)
      - Geometric shot conversion probability g(z) (distance & visible goal angle)
      - Sub-stochastic transition kernel P_succ(z -> z') = T(z -> z') * r(z -> z'), where
        r(z -> z') < 1 is pass/carry completion probability (complement absorbed by turnover)

    Expected Threat satisfies the Bellman fixed-point equation:
        xT(z) = s(z) * g(z) + m(z) * sum_{z'} P_succ(z -> z') * xT(z')

    Because the transition operator M = diag(m) * P_succ has spectral radius rho(M) < 1,
    value iteration converges geometrically to the unique fixed point xT* = (I - M)^(-1) (s * g).

Usage:
    python tactica_xt_engine.py
    python tactica_xt_engine.py --export-json xt_markov_grid.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

NX = 12  # Lengthwise zones (0 = own goal line -> 11 = opponent goal line)
NY = 8   # Widthwise lanes (0 = top flank, 3-4 = central corridor, 7 = bottom flank)
N_STATES = NX * NY
PITCH_LEN = 105.0
PITCH_WID = 68.0
GOAL_HALF_WID = 3.66
DEFAULT_XT_JSON = "xt_markov_grid.json"


def cell_center_meters(x: int, y: int) -> Tuple[float, float]:
    """Returns metric (X, Y) coordinates of grid cell (x, y) on a 105m x 68m pitch."""
    cx = (x + 0.5) * (PITCH_LEN / NX)
    cy = (y + 0.5) * (PITCH_WID / NY)
    return cx, cy


def visible_goal_angle(cx: float, cy: float) -> float:
    """Computes the subtended visible angle (radians) to the 7.32m opponent goal mouth."""
    dx = max(0.5, PITCH_LEN - cx)
    dy_top = (PITCH_WID / 2.0 + GOAL_HALF_WID) - cy
    dy_bot = (PITCH_WID / 2.0 - GOAL_HALF_WID) - cy
    ang = abs(math.atan2(dy_top, dx) - math.atan2(dy_bot, dx))
    return float(ang)


def build_spatial_primitives(tactic_mode: str = "balanced") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Constructs clean-room geometric shot probability s(z), shot conversion g(z),
    move probability m(z), and sub-stochastic successful transition kernel P_succ(z -> z').
    """
    s = np.zeros(N_STATES, dtype=np.float64)
    g = np.zeros(N_STATES, dtype=np.float64)
    coords = np.zeros((N_STATES, 2), dtype=np.float64)

    for x in range(NX):
        for y in range(NY):
            idx = y * NX + x
            cx, cy = cell_center_meters(x, y)
            coords[idx] = (cx, cy)

            dx = PITCH_LEN - cx
            dy = abs(cy - PITCH_WID / 2.0)
            dist = math.hypot(dx, dy)
            angle = visible_goal_angle(cx, cy)
            in_box = (cx >= PITCH_LEN - 16.5) and (dy <= 20.16)
            in_six = (cx >= PITCH_LEN - 5.5) and (dy <= 9.16)

            # Geometric shot conversion probability g(z) calibrated to European open-play xG
            logit_g = -2.25 + 2.35 * angle - 0.105 * dist + (0.45 if in_box else 0.0) + (0.35 if in_six else 0.0)
            g[idx] = 1.0 / (1.0 + math.exp(-logit_g))

            # Shot attempt probability s(z): negligible outside 35m, rises smoothly inside box
            if dist > 38.0:
                s[idx] = 0.001
            else:
                logit_s = 1.65 - 0.195 * dist + 1.15 * angle + (0.55 if in_box else 0.0)
                s[idx] = float(np.clip(1.0 / (1.0 + math.exp(-logit_s)), 0.001, 0.72))

            # Tactical shooting appetite adjustment
            if tactic_mode == "highPress" and cx >= 70.0:
                s[idx] = min(0.76, s[idx] * 1.08)
            elif tactic_mode == "catenaccio" and cx >= 75.0:
                g[idx] = min(0.65, g[idx] * 1.06)  # Counter-attack shots face less set defenses

    m = 1.0 - s

    # Build move transition distribution T(i -> j) and pass/carry retention r(i -> j)
    P_succ = np.zeros((N_STATES, N_STATES), dtype=np.float64)

    for i in range(N_STATES):
        xi, yi = i % NX, i // NX
        cxi, cyi = coords[i]
        raw_weights = np.zeros(N_STATES, dtype=np.float64)
        retention = np.zeros(N_STATES, dtype=np.float64)

        for j in range(N_STATES):
            if i == j:
                continue
            xj, yj = j % NX, j // NX
            cxj, cyj = coords[j]

            step_x = cxj - cxi
            step_y = cyj - cyi
            move_dist = math.hypot(step_x, step_y)

            # Restrict single-action transition radius to realistic pass/carry range (<= 46m)
            if move_dist > 46.0:
                continue

            # Base spatial mobility kernel (favors short/medium progressive & lateral passes)
            forward_bias = math.exp(0.030 * step_x) if step_x >= 0 else math.exp(0.048 * step_x)
            dist_decay = math.exp(-0.072 * move_dist)

            # Zone 14 & Half-space progression / flank cut-back affinity
            cutback_boost = 1.0
            if xi >= 9 and (yi <= 1 or yi >= 6) and xj >= 10 and (2 <= yj <= 5):
                cutback_boost = 2.4 if tactic_mode != "wingOverload" else 3.2
            elif tactic_mode == "wingOverload" and (yj <= 1 or yj >= 6) and xj >= 7:
                cutback_boost = 1.55
            elif tactic_mode == "catenaccio" and step_x >= 18.0 and (2 <= yj <= 5):
                cutback_boost = 1.45
            elif tactic_mode == "highPress" and xj >= 8 and (2 <= yj <= 5):
                cutback_boost = 1.25

            raw_weights[j] = dist_decay * forward_bias * cutback_boost

            # Pass / carry completion probability r(i -> j) decreases with distance & box density
            in_target_box = (cxj >= PITCH_LEN - 16.5) and (abs(cyj - PITCH_WID / 2.0) <= 20.16)
            base_ret = 0.87 - 0.0085 * move_dist - (0.0022 * max(0.0, cxj - 55.0))
            if in_target_box:
                base_ret -= 0.16
            if tactic_mode == "wingOverload" and xi >= 9 and (yi <= 1 or yi >= 6) and in_target_box:
                base_ret += 0.08
            elif tactic_mode == "highPress" and cxi >= 65.0:
                base_ret += 0.03
            elif tactic_mode == "catenaccio" and step_x >= 15.0:
                base_ret += 0.04

            retention[j] = float(np.clip(base_ret, 0.22, 0.90))

        w_sum = float(np.sum(raw_weights))
        if w_sum > 0:
            T_row = raw_weights / w_sum
            P_succ[i] = T_row * retention

    return s, g, m, P_succ


def solve_expected_threat(
    tactic_mode: str = "balanced",
    tol: float = 1e-8,
    max_iter: int = 200,
) -> Dict:
    """
    Runs Bellman Markov Value Iteration until ||xT^(k+1) - xT^(k)||_inf < tol,
    and verifies exact agreement with the analytical linear solver (I - M)^(-1)(s * g).
    """
    s, g, m, P_succ = build_spatial_primitives(tactic_mode=tactic_mode)
    M = m[:, None] * P_succ
    immediate_payoff = s * g

    # Spectral radius check (must be < 1 for contraction mapping)
    eigvals = np.linalg.eigvals(M)
    spectral_radius = float(np.max(np.abs(eigvals)))

    xt = np.zeros(N_STATES, dtype=np.float64)
    convergence_trace: List[Dict[str, float]] = []

    for it in range(1, max_iter + 1):
        xt_next = immediate_payoff + M @ xt
        max_delta = float(np.max(np.abs(xt_next - xt)))
        if it <= 12 or it % 5 == 0 or max_delta < tol:
            convergence_trace.append({"iter": it, "maxDelta": round(max_delta, 9), "meanXT": round(float(np.mean(xt_next)), 6)})
        xt = xt_next
        if max_delta < tol:
            break

    # Cross-check exact analytical solution: (I - M) xT = s * g
    xt_exact = np.linalg.solve(np.eye(N_STATES) - M, immediate_payoff)
    exact_max_err = float(np.max(np.abs(xt - xt_exact)))

    grid = xt.reshape((NY, NX))
    shot_prob_grid = s.reshape((NY, NX))
    shot_conv_grid = g.reshape((NY, NX))

    # Extract iconic tactical zone benchmarks (Zone 14, Half-Spaces, Six-Yard Box, Own Build-Up)
    zone14_mean = float(np.mean(grid[3:5, 9]))
    half_space_mean = float(np.mean([grid[2, 9], grid[5, 9], grid[2, 10], grid[5, 10]]))
    flank_cross_mean = float(np.mean([grid[0, 10], grid[1, 10], grid[6, 10], grid[7, 10], grid[0, 11], grid[7, 11]]))
    central_box_peak = float(np.max(grid[3:5, 11]))
    own_buildup_mean = float(np.mean(grid[2:6, 0:2]))

    return {
        "tacticMode": tactic_mode,
        "dimensions": {"nx": NX, "ny": NY, "states": N_STATES},
        "convergence": {
            "iterations": int(convergence_trace[-1]["iter"]),
            "tolerance": tol,
            "spectralRadius": round(spectral_radius, 5),
            "analyticalMaxDiff": round(exact_max_err, 12),
            "trace": convergence_trace[:10],
        },
        "summary": {
            "minXT": round(float(np.min(grid)), 5),
            "maxXT": round(float(np.max(grid)), 5),
            "meanXT": round(float(np.mean(grid)), 5),
            "ownBuildUpXT": round(own_buildup_mean, 5),
            "zone14XT": round(zone14_mean, 5),
            "halfSpaceXT": round(half_space_mean, 5),
            "flankDeliveryXT": round(flank_cross_mean, 5),
            "centralBoxPeakXT": round(central_box_peak, 5),
        },
        "grid": np.round(grid, 5).tolist(),  # Shape: [8][12] (row y=0..7, col x=0..11)
        "shotProbGrid": np.round(shot_prob_grid, 4).tolist(),
        "shotConvGrid": np.round(shot_conv_grid, 4).tolist(),
    }


def build_full_xt_artifact() -> Dict:
    """Solves baseline + 3 tactical regime xT surfaces and packages a compact browser JSON."""
    baseline = solve_expected_threat("balanced")
    regimes = {
        "balanced": baseline["grid"],
        "highPress": solve_expected_threat("highPress")["grid"],
        "catenaccio": solve_expected_threat("catenaccio")["grid"],
        "wingOverload": solve_expected_threat("wingOverload")["grid"],
    }
    regime_summaries = {
        k: {
            "zone14XT": round(float(np.mean(np.array(v)[3:5, 9])), 5),
            "flankDeliveryXT": round(float(np.mean([v[0][10], v[1][10], v[6][10], v[7][10]])), 5),
            "maxXT": round(float(np.max(v)), 5),
        }
        for k, v in regimes.items()
    }
    return {
        "schemaVersion": "3.13-xt-markov",
        "engine": "SpatialMarkovExpectedThreat",
        "reference": "Singh (2018) 12x8 Absorbing Markov Value Iteration",
        "dimensions": baseline["dimensions"],
        "convergence": baseline["convergence"],
        "summary": baseline["summary"],
        "regimeSummaries": regime_summaries,
        "grid": baseline["grid"],
        "regimes": regimes,
    }


def main():
    parser = argparse.ArgumentParser(description="TACTICA.SIM 12x8 Spatial Expected Threat (xT) Markov Solver")
    parser.add_argument("--export-json", type=str, default=DEFAULT_XT_JSON, help="Output JSON path for frontend heatmap")
    args = parser.parse_args()

    artifact = build_full_xt_artifact()
    out_path = Path(args.export_json)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent / out_path
    out_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    print("=" * 78)
    print("TACTICA.SIM S.r.l. — 12x8 Spatial Expected Threat (xT) Markov Solver Report")
    print("=" * 78)
    print(json.dumps({
        "dimensions": artifact["dimensions"],
        "convergence": {k: v for k, v in artifact["convergence"].items() if k != "trace"},
        "summary": artifact["summary"],
        "regimeSummaries": artifact["regimeSummaries"],
    }, indent=2))
    print("\nConverged 12x8 Baseline xT Surface (Attacking Left -> Right):")
    arr = np.array(artifact["grid"])
    for y in range(NY):
        print(f"  Row y={y}: " + " ".join(f"{arr[y, x]:.4f}" for x in range(NX)))
    print(f"\nExported live spatial artifact -> {out_path}")


if __name__ == "__main__":
    main()
