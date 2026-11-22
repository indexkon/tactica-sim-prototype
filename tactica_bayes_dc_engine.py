"""
TACTICA.SIM S.r.l. — Quantitative Football Calibration & Inference Engine (TRL 5 Backend Core)
==============================================================================================
Module: tactica_bayes_dc_engine.py
Alignment: MIMIT Italia Startup Visa (D.L. 179/2012 Art. 25 & Legge 193/2024 Art. 28)
Architecture: Clean-Room Statistical Pipeline (Zero Third-Party Commercial Data Dependency)

Key Components Implemented:
1. Time-Decayed Empirical Bayes Dixon-Coles (1997) Correlated Double-Poisson MLE:
   - Exponential time decay w_k = exp(-xi * delta_days) to track evolving team form.
   - Empirical Bayes Gaussian L2 shrinkage prior on team attack/defense parameters (prevents
     small-sample overfitting for newly promoted or youth academy squads).
   - Low-score dependence correction tau_rho(h, a) for 0-0, 1-0, 0-1, 1-1 outcomes.
2. Walk-Forward Out-of-Sample Backtest Suite (No Look-Ahead Bias):
   - Compares Static GF/GA Independent Poisson (v3.11 frontend baseline) against
     Time-Decayed Empirical Bayes Dixon-Coles on proper scoring rules:
     * Ranked Probability Score (RPS) — Constantinou & Fenton (2012)
     * Multi-class Log-Loss
     * Brier Score
3. Batch ETL -> Client Inference Bridge (`calibrated_serie_a_params.json`):
   - Automatically loads `serie_a_cleanroom_2023_2026.csv` (1,140 real Serie A final scorelines,
     2023-24 to 2025-26, strictly stripped of any bookmaker odds and anonymized to Clean-Room
     3-letter geographic codes aligned with TACTICA.SIM v3.11).
   - Exports `calibrated_serie_a_params.json` so the static web prototype (`index.html`) loads
     the exact Python-calibrated Empirical Bayes parameters directly in the browser.

Usage:
    python tactica_bayes_dc_engine.py
    python tactica_bayes_dc_engine.py --export-json calibrated_serie_a_params.json
    python tactica_bayes_dc_engine.py --csv 2324.csv 2425.csv 2526.csv --anonymize
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

DEFAULT_REAL_CSV = "serie_a_cleanroom_2023_2026.csv"
DEFAULT_EXPORT_JSON = "calibrated_serie_a_params.json"

# Clean-Room geographic club code mapping aligned with TACTICA.SIM v3.11 (Zero Trademark Risk)
CLEANROOM_TEAM_MAP: Dict[str, str] = {
    "Juventus": "TOR",
    "Milan": "MIL",
    "Inter": "AMB",
    "Napoli": "PRT",
    "Torino": "TRG",
    "Roma": "CAP",
    "Atalanta": "BRG",
    "Lazio": "TBR",
    "Fiorentina": "FLR",
    "Bologna": "FEL",
    "Cagliari": "CAG",
    "Como": "COM",
    "Cremonese": "CRE",
    "Empoli": "EMP",
    "Frosinone": "FRO",
    "Genoa": "GEN",
    "Lecce": "LEC",
    "Monza": "MON",
    "Parma": "PAR",
    "Pisa": "PIS",
    "Salernitana": "SAL",
    "Sassuolo": "SAS",
    "Udinese": "UDI",
    "Venezia": "VEN",
    "Verona": "VER",
}


# ---------------------------------------------------------------------------
# 1. DATA STRUCTURES, CSV INGESTION & PROPER SCORING RULES
# ---------------------------------------------------------------------------
@dataclass
class MatchPrediction:
    home_team: str
    away_team: str
    lambda_h: float
    lambda_a: float
    rho: float
    prob_home: float
    prob_draw: float
    prob_away: float
    modal_score: Tuple[int, int]
    modal_prob: float
    shannon_entropy: float
    matrix: np.ndarray  # 11x11 normalized joint scoreline probability matrix

    def to_frontend_json(self) -> Dict:
        """Exports a JSON payload directly compatible with the TACTICA.SIM v3.11 client."""
        return {
            "home": self.home_team,
            "away": self.away_team,
            "xG_H": round(self.lambda_h, 4),
            "xG_A": round(self.lambda_a, 4),
            "rho": round(self.rho, 4),
            "probabilities": {
                "home": round(self.prob_home * 100, 2),
                "draw": round(self.prob_draw * 100, 2),
                "away": round(self.prob_away * 100, 2),
            },
            "modalScore": {
                "h": int(self.modal_score[0]),
                "a": int(self.modal_score[1]),
                "probPct": round(self.modal_prob * 100, 2),
            },
            "shannonEntropy": round(self.shannon_entropy, 4),
            "matrix6x6": np.round(self.matrix[:6, :6], 5).tolist(),
        }


def parse_flexible_dates(series: pd.Series) -> pd.Series:
    """Safely parses ISO (YYYY-MM-DD) or European (DD/MM/YYYY) date strings without locale ambiguity."""
    s = series.astype(str).str.strip()
    if s.str.match(r"^\d{4}-\d{2}-\d{2}").all():
        return pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    return pd.to_datetime(s, dayfirst=True, errors="coerce")


def load_scoreline_csvs(csv_paths: Sequence[str], anonymize: bool = False) -> pd.DataFrame:
    """
    Loads one or more public historical match CSVs, strictly retaining only
    (Date, HomeTeam, AwayTeam, FTHG, FTAG) and discarding any auxiliary columns.
    """
    frames: List[pd.DataFrame] = []
    req_cols = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    for p in csv_paths:
        tmp = pd.read_csv(p, usecols=lambda c: c in req_cols)
        missing = [c for c in req_cols if c not in tmp.columns]
        if missing:
            raise ValueError(f"CSV {p} is missing required columns: {missing}")
        tmp = tmp[req_cols].copy()
        tmp["Date"] = parse_flexible_dates(tmp["Date"])
        tmp = tmp.dropna(subset=req_cols).copy()
        tmp["HomeTeam"] = tmp["HomeTeam"].astype(str).str.strip()
        tmp["AwayTeam"] = tmp["AwayTeam"].astype(str).str.strip()
        if anonymize:
            tmp["HomeTeam"] = tmp["HomeTeam"].map(lambda x: CLEANROOM_TEAM_MAP.get(x, x[:3].upper()))
            tmp["AwayTeam"] = tmp["AwayTeam"].map(lambda x: CLEANROOM_TEAM_MAP.get(x, x[:3].upper()))
        tmp["FTHG"] = tmp["FTHG"].astype(int)
        tmp["FTAG"] = tmp["FTAG"].astype(int)
        frames.append(tmp)

    return pd.concat(frames, ignore_index=True).sort_values("Date").reset_index(drop=True)


def load_default_dataset() -> Tuple[pd.DataFrame, str]:
    """Loads the 3-season Clean-Room real Serie A CSV if present, else falls back to embedded benchmark."""
    csv_path = Path(__file__).resolve().parent / DEFAULT_REAL_CSV
    if csv_path.exists():
        df = load_scoreline_csvs([str(csv_path)], anonymize=False)
        return df, f"Real Serie A Clean-Room Scorelines 2023-2026 ({len(df)} matches, {DEFAULT_REAL_CSV})"
    df = generate_cleanroom_serie_a_dataset(seed=2026)
    return df, f"Embedded Clean-Room 2-Season Benchmark ({len(df)} matches)"


def outcome_to_one_hot(home_goals: int, away_goals: int) -> np.ndarray:
    """Returns [1, 0, 0] for Home Win, [0, 1, 0] for Draw, [0, 0, 1] for Away Win."""
    if home_goals > away_goals:
        return np.array([1.0, 0.0, 0.0])
    if home_goals == away_goals:
        return np.array([0.0, 1.0, 0.0])
    return np.array([0.0, 0.0, 1.0])


def ranked_probability_score(probs: np.ndarray, actual: np.ndarray) -> float:
    """
    Ranked Probability Score (RPS) for ordered football outcomes (H, D, A).
    Lower is better. Reference: Constantinou & Fenton (2012).
    """
    cdf_p = np.cumsum(probs)
    cdf_a = np.cumsum(actual)
    return float(0.5 * np.sum((cdf_p[:-1] - cdf_a[:-1]) ** 2))


def multiclass_log_loss(probs: np.ndarray, actual: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(probs, eps, 1.0 - eps)
    return float(-np.sum(actual * np.log(p)))


def brier_score(probs: np.ndarray, actual: np.ndarray) -> float:
    return float(np.sum((probs - actual) ** 2))


# ---------------------------------------------------------------------------
# 2. BASELINE MODEL: STATIC GF/GA INDEPENDENT POISSON (v3.11 FRONTEND BASELINE)
# ---------------------------------------------------------------------------
class StaticPoissonBaseline:
    """
    Baseline attack-defense ratio model matching the static client-side formula:
        xG_H = (HomeGF * AwayGA) / LeagueHomeAvg
        xG_A = (AwayGF * HomeGA) / LeagueAwayAvg
    No time decay, no Bayesian shrinkage, independent marginals (rho = 0).
    """

    def __init__(self):
        self.home_avg = 1.56
        self.away_avg = 1.24
        self.stats: Dict[str, Dict[str, float]] = {}

    def fit(self, df: pd.DataFrame) -> "StaticPoissonBaseline":
        self.home_avg = max(0.5, float(df["FTHG"].mean()))
        self.away_avg = max(0.5, float(df["FTAG"].mean()))
        teams = sorted(set(df["HomeTeam"]).union(set(df["AwayTeam"])))

        self.stats = {}
        for tm in teams:
            h_df = df[df["HomeTeam"] == tm]
            a_df = df[df["AwayTeam"] == tm]
            self.stats[tm] = {
                "homeGF": float(h_df["FTHG"].mean()) if len(h_df) > 0 else self.home_avg,
                "homeGA": float(h_df["FTAG"].mean()) if len(h_df) > 0 else self.away_avg,
                "awayGF": float(a_df["FTAG"].mean()) if len(a_df) > 0 else self.away_avg,
                "awayGA": float(a_df["FTHG"].mean()) if len(a_df) > 0 else self.home_avg,
            }
        return self

    def predict(self, home_team: str, away_team: str, max_goals: int = 10) -> MatchPrediction:
        hs = self.stats.get(
            home_team,
            {"homeGF": self.home_avg, "homeGA": self.away_avg, "awayGF": self.away_avg, "awayGA": self.home_avg},
        )
        as_ = self.stats.get(
            away_team,
            {"homeGF": self.home_avg, "homeGA": self.away_avg, "awayGF": self.away_avg, "awayGA": self.home_avg},
        )
        lam_h = float(np.clip((hs["homeGF"] * as_["awayGA"]) / self.home_avg, 0.20, 4.50))
        lam_a = float(np.clip((as_["awayGF"] * hs["homeGA"]) / self.away_avg, 0.20, 4.50))
        return build_dixon_coles_prediction(home_team, away_team, lam_h, lam_a, rho=0.0, max_goals=max_goals)


# ---------------------------------------------------------------------------
# 3. CORE ENGINE: TIME-DECAYED EMPIRICAL BAYES DIXON-COLES (1997)
# ---------------------------------------------------------------------------
def dixon_coles_tau_vectorized(
    hg: np.ndarray, ag: np.ndarray, lam_h: np.ndarray, lam_a: np.ndarray, rho: float
) -> np.ndarray:
    """Vectorized Dixon-Coles low-score correlation adjustment factor tau_rho(h, a)."""
    tau = np.ones_like(lam_h, dtype=np.float64)
    m00 = (hg == 0) & (ag == 0)
    m10 = (hg == 1) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m11 = (hg == 1) & (ag == 1)

    tau[m00] = 1.0 - lam_h[m00] * lam_a[m00] * rho
    tau[m10] = 1.0 + lam_a[m10] * rho
    tau[m01] = 1.0 + lam_h[m01] * rho
    tau[m11] = 1.0 - rho
    return np.clip(tau, 1e-9, None)


def build_dixon_coles_prediction(
    home_team: str,
    away_team: str,
    lambda_h: float,
    lambda_a: float,
    rho: float,
    max_goals: int = 10,
) -> MatchPrediction:
    """Computes normalized 11x11 Dixon-Coles joint scoreline matrix & Shannon entropy."""
    k = np.arange(max_goals + 1)
    p_h = np.exp(-lambda_h + k * np.log(max(lambda_h, 1e-12)) - gammaln(k + 1))
    p_a = np.exp(-lambda_a + k * np.log(max(lambda_a, 1e-12)) - gammaln(k + 1))

    mat = np.outer(p_h, p_a)
    mat[0, 0] *= max(0.0, 1.0 - lambda_h * lambda_a * rho)
    mat[1, 0] *= max(0.0, 1.0 + lambda_a * rho)
    mat[0, 1] *= max(0.0, 1.0 + lambda_h * rho)
    mat[1, 1] *= max(0.0, 1.0 - rho)

    total_mass = float(np.sum(mat))
    if total_mass > 0:
        mat /= total_mass

    prob_home = float(np.sum(np.tril(mat, k=-1)))
    prob_draw = float(np.sum(np.diag(mat)))
    prob_away = float(np.sum(np.triu(mat, k=1)))

    s = prob_home + prob_draw + prob_away
    prob_home, prob_draw, prob_away = prob_home / s, prob_draw / s, prob_away / s

    sub = mat[:6, :6]
    flat_idx = int(np.argmax(sub))
    mh, ma = divmod(flat_idx, 6)
    modal_prob = float(sub[mh, ma])

    probs_vec = np.array([prob_home, prob_draw, prob_away])
    entropy_raw = float(-np.sum(probs_vec * np.log(np.clip(probs_vec, 1e-12, 1.0))))
    shannon_entropy = float(np.clip(entropy_raw / math.log(3.0), 0.0, 1.0))

    return MatchPrediction(
        home_team=home_team,
        away_team=away_team,
        lambda_h=lambda_h,
        lambda_a=lambda_a,
        rho=rho,
        prob_home=prob_home,
        prob_draw=prob_draw,
        prob_away=prob_away,
        modal_score=(mh, ma),
        modal_prob=modal_prob,
        shannon_entropy=shannon_entropy,
        matrix=mat,
    )


class TimeDecayedBayesDixonColes:
    """
    Time-Decayed Empirical Bayes Dixon-Coles Parameter Calibration Engine.

    Model specification:
        log(lambda_H) = mu + alpha_home + beta_away + gamma_home_adv
        log(lambda_A) = mu + alpha_away + beta_home

    Estimation:
        Maximizes time-weighted Dixon-Coles log-likelihood minus Empirical Bayes
        Gaussian prior penalty (L2 shrinkage toward league mean):
            Penalty = (1 / (2 * sigma_prior^2)) * (sum(alpha_i^2) + sum(beta_i^2))
    """

    def __init__(self, xi: float = 0.0020, prior_var: float = 0.25):
        self.xi = xi
        self.prior_var = prior_var
        self.teams: List[str] = []
        self.team_to_idx: Dict[str, int] = {}
        self.attack: np.ndarray = np.array([])
        self.defense: np.ndarray = np.array([])
        self.mu: float = 0.20
        self.home_adv: float = 0.14
        self.rho: float = -0.06
        self.matches_fitted: int = 0

    def fit(self, df: pd.DataFrame, ref_date: Optional[pd.Timestamp] = None) -> "TimeDecayedBayesDixonColes":
        df = df.copy()
        df["Date"] = parse_flexible_dates(df["Date"])
        if ref_date is None:
            ref_date = df["Date"].max()

        self.teams = sorted(set(df["HomeTeam"]).union(set(df["AwayTeam"])))
        self.team_to_idx = {t: i for i, t in enumerate(self.teams)}
        self.matches_fitted = len(df)
        n_teams = len(self.teams)

        h_idx = df["HomeTeam"].map(self.team_to_idx).to_numpy(dtype=np.int64)
        a_idx = df["AwayTeam"].map(self.team_to_idx).to_numpy(dtype=np.int64)
        hg = df["FTHG"].to_numpy(dtype=np.int64)
        ag = df["FTAG"].to_numpy(dtype=np.int64)

        delta_days = np.maximum(0.0, (ref_date - df["Date"]).dt.days.to_numpy(dtype=np.float64))
        weights = np.exp(-self.xi * delta_days)
        weights = weights * (len(weights) / max(np.sum(weights), 1e-9))

        log_fact_hg = gammaln(hg + 1)
        log_fact_ag = gammaln(ag + 1)
        inv_two_var = 1.0 / (2.0 * self.prior_var)

        def neg_log_posterior(params: np.ndarray) -> float:
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

            ll = (
                hg * log_lam_h - lam_h - log_fact_hg
                + ag * log_lam_a - lam_a - log_fact_ag
                + np.log(tau)
            )
            weighted_nll = -float(np.sum(weights * ll))

            eb_penalty = inv_two_var * float(np.sum(alpha_c ** 2) + np.sum(beta ** 2))
            center_penalty = 50.0 * float(np.mean(alpha) ** 2)
            return weighted_nll + eb_penalty + center_penalty

        x0 = np.zeros(2 * n_teams + 3, dtype=np.float64)
        x0[2 * n_teams] = math.log(max(float(df["FTAG"].mean()), 0.5))
        x0[2 * n_teams + 1] = 0.15
        x0[2 * n_teams + 2] = -0.06

        bounds = (
            [(-1.5, 1.5)] * n_teams
            + [(-1.5, 1.5)] * n_teams
            + [(-1.0, 1.5), (-0.5, 0.8), (-0.18, 0.05)]
        )

        res = minimize(neg_log_posterior, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 250})
        opt = res.x
        raw_alpha = opt[:n_teams]
        self.attack = raw_alpha - np.mean(raw_alpha)
        self.defense = opt[n_teams : 2 * n_teams]
        self.mu = float(opt[2 * n_teams])
        self.home_adv = float(opt[2 * n_teams + 1])
        self.rho = float(opt[2 * n_teams + 2])
        return self

    def predict(
        self,
        home_team: str,
        away_team: str,
        tactic_mult_h: float = 1.0,
        tactic_mult_a: float = 1.0,
        max_goals: int = 10,
    ) -> MatchPrediction:
        a_h = self.attack[self.team_to_idx[home_team]] if home_team in self.team_to_idx else 0.0
        d_h = self.defense[self.team_to_idx[home_team]] if home_team in self.team_to_idx else 0.0
        a_a = self.attack[self.team_to_idx[away_team]] if away_team in self.team_to_idx else 0.0
        d_a = self.defense[self.team_to_idx[away_team]] if away_team in self.team_to_idx else 0.0

        lam_h = float(np.clip(math.exp(self.mu + a_h + d_a + self.home_adv) * tactic_mult_h, 0.18, 5.0))
        lam_a = float(np.clip(math.exp(self.mu + a_a + d_h) * tactic_mult_a, 0.18, 5.0))
        return build_dixon_coles_prediction(home_team, away_team, lam_h, lam_a, self.rho, max_goals=max_goals)


# ---------------------------------------------------------------------------
# 4. BATCH ETL EXPORT: PYTHON CALIBRATION -> FRONTEND JSON ARTIFACT
# ---------------------------------------------------------------------------
def export_calibrated_params_json(
    model: TimeDecayedBayesDixonColes,
    summary: Dict,
    df: pd.DataFrame,
    output_path: str | Path = DEFAULT_EXPORT_JSON,
    source_label: str = "",
) -> Dict:
    """
    Exports fitted Empirical Bayes Dixon-Coles parameters (mu, gamma, rho, team alpha/beta)
    plus empirical GF/GA stats and walk-forward backtest metrics to JSON for direct browser
    consumption by `index.html`.
    """
    df_copy = df.copy()
    df_copy["Date"] = parse_flexible_dates(df_copy["Date"])
    league_home_avg = round(float(df_copy["FTHG"].mean()), 4)
    league_away_avg = round(float(df_copy["FTAG"].mean()), 4)

    teams_payload: Dict[str, Dict[str, float]] = {}
    for idx, code in enumerate(model.teams):
        h_df = df_copy[df_copy["HomeTeam"] == code]
        a_df = df_copy[df_copy["AwayTeam"] == code]
        teams_payload[code] = {
            "alpha": round(float(model.attack[idx]), 5),
            "beta": round(float(model.defense[idx]), 5),
            "homeGF": round(float(h_df["FTHG"].mean()), 2) if len(h_df) > 0 else round(league_home_avg, 2),
            "homeGA": round(float(h_df["FTAG"].mean()), 2) if len(h_df) > 0 else round(league_away_avg, 2),
            "awayGF": round(float(a_df["FTAG"].mean()), 2) if len(a_df) > 0 else round(league_away_avg, 2),
            "awayGA": round(float(a_df["FTHG"].mean()), 2) if len(a_df) > 0 else round(league_home_avg, 2),
            "matches": int(len(h_df) + len(a_df)),
        }

    payload = {
        "schemaVersion": "3.12-bayes-etl",
        "engine": "TimeDecayedBayesDixonColes",
        "dataSource": source_label or DEFAULT_REAL_CSV,
        "dateRange": {
            "start": df_copy["Date"].min().strftime("%Y-%m-%d"),
            "end": df_copy["Date"].max().strftime("%Y-%m-%d"),
        },
        "totalMatches": int(len(df_copy)),
        "uniqueClubs": int(len(model.teams)),
        "hyperparameters": {
            "xi": round(float(model.xi), 5),
            "priorVar": round(float(model.prior_var), 4),
        },
        "global": {
            "mu": round(float(model.mu), 5),
            "homeAdvantage": round(float(model.home_adv), 5),
            "rho": round(float(model.rho), 5),
            "leagueHomeAvg": league_home_avg,
            "leagueAwayAvg": league_away_avg,
        },
        "backtest": {
            "evaluatedMatches": int(summary["evaluated_out_of_sample_matches"]),
            "baselineRPS": summary["static_poisson_baseline"]["RPS"],
            "bayesRPS": summary["time_decayed_bayes_dixon_coles"]["RPS"],
            "rpsImprovementPct": summary["improvement_RPS_pct"],
            "baselineLogLoss": summary["static_poisson_baseline"]["LogLoss"],
            "bayesLogLoss": summary["time_decayed_bayes_dixon_coles"]["LogLoss"],
            "logLossImprovementPct": summary["improvement_LogLoss_pct"],
            "baselineBrier": summary["static_poisson_baseline"]["Brier"],
            "bayesBrier": summary["time_decayed_bayes_dixon_coles"]["Brier"],
            "brierImprovementPct": summary["improvement_Brier_pct"],
        },
        "teams": teams_payload,
    }

    out = Path(output_path)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


# ---------------------------------------------------------------------------
# 5. FALLBACK SYNTHETIC GENERATOR (USED ONLY IF CSV IS ABSENT)
# ---------------------------------------------------------------------------
def generate_cleanroom_serie_a_dataset(seed: int = 2026) -> pd.DataFrame:
    """Fallback 2-season (760-match) benchmark used only when no CSV file is present."""
    rng = np.random.default_rng(seed)
    clubs = [
        ("AMB", 0.36, -0.34), ("TOR", 0.22, -0.32), ("MIL", 0.28, -0.18), ("PRT", 0.24, -0.16),
        ("BRG", 0.30, -0.14), ("CAP", 0.18, -0.12), ("TBR", 0.14, -0.10), ("FLR", 0.15, -0.08),
        ("FEL", 0.12, -0.18), ("TRG", -0.05, -0.16), ("GEN", -0.06, 0.04), ("UDI", -0.08, 0.06),
        ("VER", -0.12, 0.10), ("MON", -0.10, 0.08), ("CAG", -0.14, 0.16), ("EMP", -0.18, 0.12),
        ("LEC", -0.20, 0.18), ("PAR", -0.16, 0.20), ("COM", -0.15, 0.19), ("VEN", -0.26, 0.28),
    ]
    base_attack = {c[0]: c[1] for c in clubs}
    base_defense = {c[0]: c[2] for c in clubs}
    names = [c[0] for c in clubs]

    rows = []
    start_date = pd.Timestamp("2024-08-18")
    for season in range(2):
        s_att = {t: base_attack[t] + (rng.normal(0, 0.16) if season == 1 else 0.0) for t in names}
        s_def = {t: base_defense[t] + (rng.normal(0, 0.16) if season == 1 else 0.0) for t in names}
        rot = names.copy()
        pairs: List[List[Tuple[str, str]]] = []
        for r in range(len(names) - 1):
            day = [(rot[i], rot[-(i + 1)]) if (r + i) % 2 == 0 else (rot[-(i + 1)], rot[i]) for i in range(len(names) // 2)]
            pairs.append(day)
            rot = [rot[0]] + [rot[-1]] + rot[1:-1]
        full_schedule = pairs + [[(a, h) for (h, a) in day] for day in pairs]
        for md_idx, day in enumerate(full_schedule):
            match_date = start_date + pd.Timedelta(days=season * 365 + md_idx * 7)
            for h_tm, a_tm in day:
                lam_h = math.exp(0.20 + s_att[h_tm] + s_def[a_tm] + 0.24)
                lam_a = math.exp(0.20 + s_att[a_tm] + s_def[h_tm])
                pred = build_dixon_coles_prediction(h_tm, a_tm, lam_h, lam_a, rho=-0.10, max_goals=8)
                flat_p = pred.matrix.ravel() / pred.matrix.sum()
                hg, ag = divmod(int(rng.choice(len(flat_p), p=flat_p)), pred.matrix.shape[1])
                rows.append({"Date": match_date, "HomeTeam": h_tm, "AwayTeam": a_tm, "FTHG": int(hg), "FTAG": int(ag)})

    return pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 6. WALK-FORWARD OUT-OF-SAMPLE BACKTEST (NO LOOK-AHEAD BIAS)
# ---------------------------------------------------------------------------
def run_walk_forward_backtest(
    df: pd.DataFrame,
    xi: float = 0.0020,
    prior_var: float = 0.25,
    min_train_matches: int = 380,
    refit_step_days: int = 14,
) -> Tuple[Dict, TimeDecayedBayesDixonColes]:
    df = df.copy()
    df["Date"] = parse_flexible_dates(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    eval_df = df.iloc[min_train_matches:].copy()
    eval_dates = sorted(eval_df["Date"].unique())

    metrics = {
        "static_poisson": {"rps": [], "logloss": [], "brier": []},
        "bayes_dc": {"rps": [], "logloss": [], "brier": []},
    }

    last_fit_date: Optional[pd.Timestamp] = None
    baseline = StaticPoissonBaseline()
    bayes_dc = TimeDecayedBayesDixonColes(xi=xi, prior_var=prior_var)

    for dt in eval_dates:
        if last_fit_date is None or (dt - last_fit_date).days >= refit_step_days:
            train_slice = df[df["Date"] < dt]
            baseline.fit(train_slice)
            bayes_dc.fit(train_slice, ref_date=dt)
            last_fit_date = dt

        day_matches = eval_df[eval_df["Date"] == dt]
        for _, row in day_matches.iterrows():
            actual = outcome_to_one_hot(int(row["FTHG"]), int(row["FTAG"]))

            p_base = baseline.predict(row["HomeTeam"], row["AwayTeam"])
            probs_base = np.array([p_base.prob_home, p_base.prob_draw, p_base.prob_away])
            metrics["static_poisson"]["rps"].append(ranked_probability_score(probs_base, actual))
            metrics["static_poisson"]["logloss"].append(multiclass_log_loss(probs_base, actual))
            metrics["static_poisson"]["brier"].append(brier_score(probs_base, actual))

            p_dc = bayes_dc.predict(row["HomeTeam"], row["AwayTeam"])
            probs_dc = np.array([p_dc.prob_home, p_dc.prob_draw, p_dc.prob_away])
            metrics["bayes_dc"]["rps"].append(ranked_probability_score(probs_dc, actual))
            metrics["bayes_dc"]["logloss"].append(multiclass_log_loss(probs_dc, actual))
            metrics["bayes_dc"]["brier"].append(brier_score(probs_dc, actual))

    final_model = TimeDecayedBayesDixonColes(xi=xi, prior_var=prior_var).fit(df)

    summary = {
        "total_dataset_matches": len(df),
        "initial_warmup_matches": min_train_matches,
        "evaluated_out_of_sample_matches": len(eval_df),
        "unique_clubs": len(final_model.teams),
        "hyperparameters": {"xi_time_decay": xi, "empirical_bayes_prior_var": prior_var, "refit_step_days": refit_step_days},
        "fitted_final_params": {
            "mu": round(final_model.mu, 4),
            "home_advantage": round(final_model.home_adv, 4),
            "dixon_coles_rho": round(final_model.rho, 4),
        },
        "static_poisson_baseline": {
            "RPS": round(float(np.mean(metrics["static_poisson"]["rps"])), 5),
            "LogLoss": round(float(np.mean(metrics["static_poisson"]["logloss"])), 5),
            "Brier": round(float(np.mean(metrics["static_poisson"]["brier"])), 5),
        },
        "time_decayed_bayes_dixon_coles": {
            "RPS": round(float(np.mean(metrics["bayes_dc"]["rps"])), 5),
            "LogLoss": round(float(np.mean(metrics["bayes_dc"]["logloss"])), 5),
            "Brier": round(float(np.mean(metrics["bayes_dc"]["brier"])), 5),
        },
    }

    for k in ("RPS", "LogLoss", "Brier"):
        b_val = summary["static_poisson_baseline"][k]
        m_val = summary["time_decayed_bayes_dixon_coles"][k]
        summary[f"improvement_{k}_pct"] = round((b_val - m_val) / b_val * 100.0, 2)

    return summary, final_model


# ---------------------------------------------------------------------------
# 7. CLI ENTRYPOINT & ARTIFACT EXPORT
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="TACTICA.SIM Time-Decayed Empirical Bayes Dixon-Coles Engine")
    parser.add_argument("--csv", nargs="+", default=None, help="One or more public historical CSVs (Date,HomeTeam,AwayTeam,FTHG,FTAG)")
    parser.add_argument("--anonymize", action="store_true", help="Map raw club names to Clean-Room 3-letter geographic codes")
    parser.add_argument("--xi", type=float, default=0.0020, help="Exponential daily time-decay coefficient")
    parser.add_argument("--prior-var", type=float, default=0.25, help="Empirical Bayes Gaussian prior variance")
    parser.add_argument("--export-json", type=str, default=DEFAULT_EXPORT_JSON, help="Output JSON path for frontend live inference")
    args = parser.parse_args()

    if args.csv:
        df = load_scoreline_csvs(args.csv, anonymize=args.anonymize)
        source_label = f"External Public CSV ({', '.join(args.csv)} — {len(df)} matches)"
    else:
        df, source_label = load_default_dataset()

    print("=" * 78)
    print("TACTICA.SIM S.r.l. — Empirical Calibration & Walk-Forward Backtest Report")
    print(f"Data Source : {source_label}")
    print("=" * 78)

    min_train = 380 if len(df) >= 760 else max(120, int(len(df) * 0.50))
    summary, final_model = run_walk_forward_backtest(
        df, xi=args.xi, prior_var=args.prior_var, min_train_matches=min_train
    )
    print(json.dumps(summary, indent=2))

    if args.export_json:
        export_path = Path(args.export_json)
        if not export_path.is_absolute():
            export_path = Path(__file__).resolve().parent / export_path
        export_calibrated_params_json(final_model, summary, df, output_path=export_path, source_label=source_label)
        print(f"\nExported live frontend calibration artifact -> {export_path}")

    teams_avail = final_model.teams
    home_demo = "TOR" if "TOR" in teams_avail else teams_avail[0]
    away_demo = "AMB" if "AMB" in teams_avail else teams_avail[1]
    sample_pred = final_model.predict(home_demo, away_demo)

    print(f"\nSample v3.12 FastAPI /predict Payload ({home_demo} vs {away_demo} — Baseline):")
    print(json.dumps(sample_pred.to_frontend_json(), indent=2))


if __name__ == "__main__":
    main()
