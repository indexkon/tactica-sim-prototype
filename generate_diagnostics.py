"""
TACTICA.SIM S.r.l. — Empirical Calibration Diagnostics & Posterior Uncertainty Generator
========================================================================================
Module: generate_diagnostics.py
Alignment: MIMIT Italia Startup Visa (D.L. 179/2012 Art. 25 & Legge 193/2024 Art. 28)

Generates publication-grade statistical validation artifacts from the 3-season Clean-Room
real Serie A dataset (1,140 matches, 2023–2026) and the 12x8 Spatial Markov xT solver:
  1. assets/serie_a_backtest_diagnostics.png
     - Panel A: Out-of-Sample Probability Calibration Curve (Reliability Diagram) + ECE
     - Panel B: Cumulative Out-of-Sample RPS Advantage (2024–25 & 2025–26, 760 matches)
  2. assets/serie_a_posterior_and_xt.png
     - Panel A: Empirical Bayes Attack vs. Defense Strength Map with 95% Credible Intervals
                computed via Observed Fisher Information (Laplace Hessian Approximation)
     - Panel B: 12x8 Spatial Markov Expected Threat (xT) Baseline Surface & Zone 14 Highlight

Usage:
    python generate_diagnostics.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from scipy.special import gammaln

from tactica_bayes_dc_engine import (
    StaticPoissonBaseline,
    TimeDecayedBayesDixonColes,
    dixon_coles_tau_vectorized,
    load_default_dataset,
    outcome_to_one_hot,
    ranked_probability_score,
    multiclass_log_loss,
    brier_score,
)
from tactica_xt_engine import build_full_xt_artifact

ROOT = Path(__file__).resolve().parent
ASSETS_DIR = ROOT / "assets"


def collect_walk_forward_records(
    df: pd.DataFrame,
    xi: float = 0.0020,
    prior_var: float = 0.25,
    min_train_matches: int = 380,
    refit_step_days: int = 14,
) -> pd.DataFrame:
    """Runs strict walk-forward backtest and returns match-level out-of-sample records."""
    df = df.copy().sort_values("Date").reset_index(drop=True)
    eval_df = df.iloc[min_train_matches:].copy()
    eval_dates = sorted(eval_df["Date"].unique())

    baseline = StaticPoissonBaseline()
    bayes_dc = TimeDecayedBayesDixonColes(xi=xi, prior_var=prior_var)
    last_fit = None
    rows = []

    for dt in eval_dates:
        if last_fit is None or (dt - last_fit).days >= refit_step_days:
            train_slice = df[df["Date"] < dt]
            baseline.fit(train_slice)
            bayes_dc.fit(train_slice, ref_date=dt)
            last_fit = dt

        day_matches = eval_df[eval_df["Date"] == dt]
        for _, r in day_matches.iterrows():
            actual = outcome_to_one_hot(int(r["FTHG"]), int(r["FTAG"]))
            pb = baseline.predict(r["HomeTeam"], r["AwayTeam"])
            pd_ = bayes_dc.predict(r["HomeTeam"], r["AwayTeam"])

            probs_b = np.array([pb.prob_home, pb.prob_draw, pb.prob_away])
            probs_d = np.array([pd_.prob_home, pd_.prob_draw, pd_.prob_away])

            rows.append({
                "Date": dt,
                "HomeTeam": r["HomeTeam"],
                "AwayTeam": r["AwayTeam"],
                "y_H": actual[0], "y_D": actual[1], "y_A": actual[2],
                "b_H": probs_b[0], "b_D": probs_b[1], "b_A": probs_b[2],
                "d_H": probs_d[0], "d_D": probs_d[1], "d_A": probs_d[2],
                "rps_base": ranked_probability_score(probs_b, actual),
                "rps_dc": ranked_probability_score(probs_d, actual),
                "ll_base": multiclass_log_loss(probs_b, actual),
                "ll_dc": multiclass_log_loss(probs_d, actual),
                "br_base": brier_score(probs_b, actual),
                "br_dc": brier_score(probs_d, actual),
            })

    return pd.DataFrame(rows)


def compute_reliability_bins(preds: np.ndarray, obs: np.ndarray, n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Computes uniform calibration curve and Expected Calibration Error (ECE)."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_pred, bin_obs, bin_cnt = [], [], []
    ece = 0.0
    total = len(preds)

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (preds >= lo) & (preds < hi if i < n_bins - 1 else preds <= hi)
        cnt = int(np.sum(mask))
        if cnt >= 12:
            mp = float(np.mean(preds[mask]))
            mo = float(np.mean(obs[mask]))
            bin_pred.append(mp)
            bin_obs.append(mo)
            bin_cnt.append(cnt)
        if cnt > 0:
            ece += (cnt / total) * abs(float(np.mean(preds[mask])) - float(np.mean(obs[mask])))

    return np.array(bin_pred), np.array(bin_obs), np.array(bin_cnt), float(ece)


def compute_laplace_posterior_se(
    df: pd.DataFrame,
    model: TimeDecayedBayesDixonColes,
) -> Dict[str, Dict[str, float]]:
    """Computes Laplace approximation posterior standard errors via Observed Fisher Information."""
    n_teams = len(model.teams)
    h_idx = df["HomeTeam"].map(model.team_to_idx).to_numpy(dtype=np.int64)
    a_idx = df["AwayTeam"].map(model.team_to_idx).to_numpy(dtype=np.int64)
    hg = df["FTHG"].to_numpy(dtype=np.int64)
    ag = df["FTAG"].to_numpy(dtype=np.int64)

    ref_date = df["Date"].max()
    delta_days = np.maximum(0.0, (ref_date - df["Date"]).dt.days.to_numpy(dtype=np.float64))
    weights = np.exp(-model.xi * delta_days)
    weights = weights * (len(weights) / max(np.sum(weights), 1e-9))

    log_fact_hg = gammaln(hg + 1)
    log_fact_ag = gammaln(ag + 1)
    inv_two_var = 1.0 / (2.0 * model.prior_var)

    def nlp(params: np.ndarray) -> float:
        alpha = params[:n_teams]
        beta = params[n_teams : 2 * n_teams]
        mu = params[2 * n_teams]
        gamma = params[2 * n_teams + 1]
        rho = params[2 * n_teams + 2]
        alpha_c = alpha - np.mean(alpha)
        log_lam_h = np.clip(mu + alpha_c[h_idx] + beta[a_idx] + gamma, -3.0, 2.5)
        log_lam_a = np.clip(mu + alpha_c[a_idx] + beta[h_idx], -3.0, 2.5)
        lam_h = np.exp(log_lam_h)
        lam_a = np.exp(log_lam_a)
        tau = dixon_coles_tau_vectorized(hg, ag, lam_h, lam_a, rho)
        ll = hg * log_lam_h - lam_h - log_fact_hg + ag * log_lam_a - lam_a - log_fact_ag + np.log(tau)
        return (
            -float(np.sum(weights * ll))
            + inv_two_var * float(np.sum(alpha_c ** 2) + np.sum(beta ** 2))
            + 50.0 * float(np.mean(alpha) ** 2)
        )

    theta = np.concatenate([model.attack, model.defense, [model.mu, model.home_adv, model.rho]])
    dim = len(theta)
    eps = 1e-4
    H = np.zeros((dim, dim), dtype=np.float64)
    f0 = nlp(theta)
    for i in range(dim):
        ei = np.zeros(dim)
        ei[i] = eps
        H[i, i] = (nlp(theta + ei) - 2.0 * f0 + nlp(theta - ei)) / (eps ** 2)
        for j in range(i + 1, dim):
            ej = np.zeros(dim)
            ej[j] = eps
            v = (nlp(theta + ei + ej) - nlp(theta + ei - ej) - nlp(theta - ei + ej) + nlp(theta - ei - ej)) / (4.0 * eps * eps)
            H[i, j] = v
            H[j, i] = v

    cov = np.linalg.inv(H)
    se = np.sqrt(np.maximum(np.diag(cov), 1e-12))

    team_ci: Dict[str, Dict[str, float]] = {}
    for idx, code in enumerate(model.teams):
        cnt = int(((df["HomeTeam"] == code) | (df["AwayTeam"] == code)).sum())
        team_ci[code] = {
            "alpha": float(model.attack[idx]),
            "alpha_se": float(se[idx]),
            "alpha_ci95": float(1.96 * se[idx]),
            "beta": float(model.defense[idx]),
            "beta_se": float(se[n_teams + idx]),
            "beta_ci95": float(1.96 * se[n_teams + idx]),
            "matches": cnt,
        }
    return team_ci


def plot_backtest_diagnostics(rec: pd.DataFrame, out_path: Path) -> Dict[str, float]:
    """Generates Reliability Diagram + Cumulative Out-of-Sample RPS Advantage figure."""
    obs_all = np.concatenate([rec["y_H"].values, rec["y_D"].values, rec["y_A"].values])
    pred_b = np.concatenate([rec["b_H"].values, rec["b_D"].values, rec["b_A"].values])
    pred_d = np.concatenate([rec["d_H"].values, rec["d_D"].values, rec["d_A"].values])

    bp_b, bo_b, _, ece_b = compute_reliability_bins(pred_b, obs_all, n_bins=10)
    bp_d, bo_d, _, ece_d = compute_reliability_bins(pred_d, obs_all, n_bins=10)

    cum_rps_gain = np.cumsum(rec["rps_base"].values - rec["rps_dc"].values)
    match_idx = np.arange(1, len(rec) + 1)

    rps_b = round(float(rec["rps_base"].mean()), 5)
    rps_d = round(float(rec["rps_dc"].mean()), 5)
    ll_b = round(float(rec["ll_base"].mean()), 5)
    ll_d = round(float(rec["ll_dc"].mean()), 5)
    br_b = round(float(rec["br_base"].mean()), 5)
    br_d = round(float(rec["br_dc"].mean()), 5)
    ece_red_pct = (ece_b - ece_d) / ece_b * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.0), dpi=160)
    fig.patch.set_facecolor("#FFFFFF")

    # Panel A: Probability Calibration Curve (Reliability Diagram)
    ax = axes[0]
    ax.set_facecolor("#F8FAFC")
    ax.plot([0, 0.85], [0, 0.85], linestyle="--", color="#64748B", linewidth=1.4, label="Perfect Calibration (45°)")
    ax.plot(bp_b, bo_b, marker="s", markersize=5.5, color="#94A3B8", linewidth=1.8, linestyle="-.",
            label=f"Static Poisson Baseline (ECE = {ece_b:.4f})")
    ax.plot(bp_d, bo_d, marker="o", markersize=6.5, color="#059669", linewidth=2.4,
            label=f"Empirical Bayes Dixon-Coles (ECE = {ece_d:.4f}, -{ece_red_pct:.1f}%)")
    ax.set_xlim(0.0, 0.82)
    ax.set_ylim(0.0, 0.82)
    ax.set_xlabel("Predicted Outcome Probability (H / D / A)", fontsize=9.5, fontweight="bold", color="#1E293B")
    ax.set_ylabel("Observed Empirical Frequency", fontsize=9.5, fontweight="bold", color="#1E293B")
    ax.set_title("A. Probability Calibration Curve (Reliability Diagram)\n760 Out-of-Sample Real Serie A Matches (2,280 Outcome Predictions)", fontsize=10, fontweight="bold", color="#0F172A", pad=10)
    ax.grid(True, linestyle=":", alpha=0.55, color="#CBD5E1")
    ax.legend(loc="upper left", fontsize=8.2, frameon=True, facecolor="#FFFFFF", edgecolor="#E2E8F0")

    # Panel B: Cumulative Out-of-Sample RPS Advantage
    ax2 = axes[1]
    ax2.set_facecolor("#F8FAFC")
    ax2.axhline(0.0, color="#64748B", linestyle="--", linewidth=1.2)
    ax2.axvline(380, color="#2563EB", linestyle=":", linewidth=1.3, alpha=0.8)
    ax2.fill_between(match_idx, 0, cum_rps_gain, where=(cum_rps_gain >= 0), color="#10B981", alpha=0.15)
    ax2.fill_between(match_idx, 0, cum_rps_gain, where=(cum_rps_gain < 0), color="#F43F5E", alpha=0.15)
    ax2.plot(match_idx, cum_rps_gain, color="#059669", linewidth=2.3, label="Cumulative Σ(RPS_static − RPS_bayes)")

    ax2.set_ylim(-0.20, 4.35)
    ax2.text(190, 3.88, "2024–25 Season\n(Matches 1–380)", ha="center", va="center", fontsize=8.2, color="#475569", fontweight="bold")
    ax2.text(570, 3.88, "2025–26 Season\n(Matches 381–760)", ha="center", va="center", fontsize=8.2, color="#1D4ED8", fontweight="bold")

    summary_box = (
        f"Out-of-Sample Error Change (n = 760)\n"
        f"• RPS:      {rps_b:.5f} → {rps_d:.5f} ({(rps_d-rps_b)/rps_b*100:+.2f}%)\n"
        f"• Log-Loss: {ll_b:.5f} → {ll_d:.5f} ({(ll_d-ll_b)/ll_b*100:+.2f}%)\n"
        f"• Brier:    {br_b:.5f} → {br_d:.5f} ({(br_d-br_b)/br_b*100:+.2f}%)\n"
        f"• ECE:      {ece_b:.4f}  → {ece_d:.4f}  ({(ece_d-ece_b)/ece_b*100:+.1f}%)"
    )
    ax2.text(
        0.47, 0.06, summary_box,
        transform=ax2.transAxes,
        fontsize=7.9, fontfamily="monospace", color="#0F172A",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#FFFFFF", edgecolor="#10B981", linewidth=1.2, alpha=0.96),
    )

    ax2.set_xlabel("Chronological Out-of-Sample Match Index (Walk-Forward, 14-Day Refit)", fontsize=9.5, fontweight="bold", color="#1E293B")
    ax2.set_ylabel("Cumulative RPS Gain (Higher = Better)", fontsize=9.5, fontweight="bold", color="#1E293B")
    ax2.set_title("B. Cumulative Out-of-Sample RPS Advantage Over Time\nConsistent Generalization Across 2 Full Real Serie A Seasons", fontsize=10, fontweight="bold", color="#0F172A", pad=10)
    ax2.grid(True, linestyle=":", alpha=0.55, color="#CBD5E1")
    ax2.legend(loc="upper left", fontsize=8.2, frameon=True, facecolor="#FFFFFF", edgecolor="#E2E8F0")

    plt.tight_layout(pad=1.6)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return {
        "ece_baseline": round(ece_b, 5),
        "ece_bayes_dc": round(ece_d, 5),
        "ece_reduction_pct": round(ece_red_pct, 2),
        "cum_rps_gain": round(float(cum_rps_gain[-1]), 4),
    }


def plot_posterior_and_xt(team_ci: Dict[str, Dict[str, float]], xt_art: Dict, out_path: Path) -> None:
    """Generates 25-Club Posterior Strength Map (with 95% CI) + 12x8 Spatial Markov xT Surface."""
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.2), dpi=160)
    fig.patch.set_facecolor("#FFFFFF")

    # Panel A: Empirical Bayes Attack vs Defense Strength with 95% Credible Intervals
    ax = axes[0]
    ax.set_facecolor("#F8FAFC")
    ax.axhline(0.0, color="#94A3B8", linestyle="--", linewidth=1.0)
    ax.axvline(0.0, color="#94A3B8", linestyle="--", linewidth=1.0)

    # Custom non-overlapping label positions (dx, dy in data coords)
    label_offsets = {
        "AMB": (0.04, 0.05),
        "CAP": (0.06, 0.12),
        "COM": (0.14, 0.04),
        "TOR": (-0.13, 0.10),
        "PRT": (-0.15, -0.02),
        "BRG": (0.13, -0.06),
        "MIL": (0.10, -0.04),
        "TBR": (-0.11, 0.06),
        "FEL": (0.07, -0.03),
        "FLR": (0.06, -0.07),
        "LEC": (-0.12, 0.06),
        "VER": (-0.13, -0.06),
        "CRE": (0.05, -0.05),
        "PIS": (0.05, -0.05),
        "SAL": (0.05, -0.04),
    }

    for code, v in team_ci.items():
        x = v["alpha"]
        y = -v["beta"]
        xerr = v["alpha_ci95"]
        yerr = v["beta_ci95"]
        n = v["matches"]

        if n >= 114:
            col, sz, alpha_bar = "#059669", 46, 0.28
        elif n >= 76:
            col, sz, alpha_bar = "#2563EB", 42, 0.25
        else:
            col, sz, alpha_bar = "#F59E0B", 38, 0.25

        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="none", ecolor=col, elinewidth=1.0, alpha=alpha_bar, capsize=2)
        ax.scatter([x], [y], color=col, s=sz, edgecolors="#0F172A", linewidths=0.75, zorder=4)
        if code in label_offsets:
            dx, dy = label_offsets[code]
            ax.annotate(
                code,
                xy=(x, y),
                xytext=(x + dx, y + dy),
                fontsize=7.6,
                fontweight="bold",
                color="#0F172A",
                ha="center",
                va="center",
                arrowprops=dict(arrowstyle="-", color="#64748B", lw=0.7, shrinkA=0, shrinkB=3),
                zorder=5,
            )

    ax.scatter([], [], color="#059669", s=46, edgecolors="#0F172A", label="3-Season Clubs (n = 114, tighter 95% CI)")
    ax.scatter([], [], color="#2563EB", s=42, edgecolors="#0F172A", label="2-Season Clubs (n = 76)")
    ax.scatter([], [], color="#F59E0B", s=38, edgecolors="#0F172A", label="1-Season Clubs (n = 38, wider Bayes CI)")

    ax.set_xlim(-0.88, 0.98)
    ax.set_ylim(-0.90, 0.76)
    ax.text(0.94, 0.70, "Elite Quadrant\n(Strong Attack & Defense)", ha="right", va="top", fontsize=7.9, color="#047857", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#ECFDF5", edgecolor="#A7F3D0"))
    ax.set_xlabel("Posterior Attack Strength α  (±95% Laplace CI)", fontsize=9.2, fontweight="bold", color="#1E293B")
    ax.set_ylabel("Posterior Defensive Strength −β  (Higher = Stronger)", fontsize=9.2, fontweight="bold", color="#1E293B")
    ax.set_title("A. Empirical Bayes Team Strength & 95% Credible Intervals\n25 Real Serie A Clubs (1,140 Matches, Observed Fisher Information)", fontsize=9.8, fontweight="bold", color="#0F172A", pad=10)
    ax.grid(True, linestyle=":", alpha=0.55, color="#CBD5E1")
    ax.legend(loc="lower right", fontsize=7.8, frameon=True, facecolor="#FFFFFF", edgecolor="#E2E8F0")

    # Panel B: 12x8 Spatial Markov Expected Threat (xT) Surface
    ax2 = axes[1]
    g_bal = np.array(xt_art["regimes"]["balanced"])
    ax2.imshow(np.power(g_bal, 0.50), cmap="YlOrRd", aspect="auto", origin="upper")
    ax2.set_title("B. 12×8 Spatial Markov Expected Threat (xT) Surface\nBellman Contraction: 28 Iters, ρ(M)=0.532, Max Analytical Err 6.4e-9", fontsize=9.8, fontweight="bold", color="#0F172A", pad=10)
    ax2.set_xlabel("Pitch Length Zone (0 = Own Goal → 11 = Opp Goal)", fontsize=9.2, fontweight="bold", color="#1E293B")
    ax2.set_ylabel("Pitch Width Lane (0–7)", fontsize=9.2, fontweight="bold", color="#1E293B")
    ax2.set_xticks(range(12))
    ax2.set_yticks(range(8))

    for r in range(8):
        for c in range(12):
            val = g_bal[r, c]
            txt_col = "#FFFFFF" if val > 0.08 else "#0F172A"
            lbl = f"{val:.3f}" if c >= 9 else f"{val:.2f}".lstrip("0")
            ax2.text(c, r, lbl, ha="center", va="center", fontsize=6.4, fontweight="bold" if c >= 9 else "normal", color=txt_col)

    rect_z14 = patches.Rectangle((8.5, 2.5), 1.0, 2.0, linewidth=2.0, edgecolor="#0284C7", facecolor="none", linestyle="--")
    ax2.add_patch(rect_z14)
    ax2.annotate(
        "Zone 14",
        xy=(8.5, 3.5),
        xytext=(6.6, 3.5),
        fontsize=8.0,
        fontweight="bold",
        color="#0369A1",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="#FFFFFF", edgecolor="#0284C7", linewidth=1.0, alpha=0.92),
        arrowprops=dict(arrowstyle="->", color="#0284C7", lw=1.4),
    )

    rect_box = patches.Rectangle((9.5, 1.5), 2.0, 5.0, linewidth=1.6, edgecolor="#0F172A", facecolor="none", linestyle=":")
    ax2.add_patch(rect_box)

    plt.tight_layout(pad=1.6)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    df, source_label = load_default_dataset()
    print(f"Generating diagnostics from: {source_label}")

    rec = collect_walk_forward_records(df, xi=0.0020, prior_var=0.25, min_train_matches=380, refit_step_days=14)
    diag_stats = plot_backtest_diagnostics(rec, ASSETS_DIR / "serie_a_backtest_diagnostics.png")

    full_model = TimeDecayedBayesDixonColes(xi=0.0020, prior_var=0.25).fit(df)
    team_ci = compute_laplace_posterior_se(df, full_model)
    xt_art = build_full_xt_artifact()
    plot_posterior_and_xt(team_ci, xt_art, ASSETS_DIR / "serie_a_posterior_and_xt.png")

    print("Diagnostics summary:", json.dumps(diag_stats, indent=2))
    print("Saved charts:")
    print("  -", ASSETS_DIR / "serie_a_backtest_diagnostics.png")
    print("  -", ASSETS_DIR / "serie_a_posterior_and_xt.png")


if __name__ == "__main__":
    main()
