"""
TACTICA.SIM S.r.l. — Automated Verification & Mathematical Invariant Test Suite
===============================================================================
Run locally or in GitHub Actions CI:
    pytest -v
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from tactica_bayes_dc_engine import (
    TimeDecayedBayesDixonColes,
    brier_score,
    build_dixon_coles_prediction,
    load_default_dataset,
    multiclass_log_loss,
    outcome_to_one_hot,
    ranked_probability_score,
)
from tactica_xt_engine import build_full_xt_artifact, solve_expected_threat
from generate_diagnostics import compute_laplace_posterior_se

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_cleanroom_csv_integrity():
    csv_path = REPO_ROOT / "serie_a_cleanroom_2023_2026.csv"
    assert csv_path.exists(), "Real Serie A Clean-Room CSV must exist in repository root"
    df = pd.read_csv(csv_path)
    assert list(df.columns) == ["Date", "Season", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    assert len(df) == 1140
    assert df.isna().sum().sum() == 0
    unique_clubs = set(df["HomeTeam"]).union(set(df["AwayTeam"]))
    assert len(unique_clubs) == 25
    assert (df["FTHG"] >= 0).all() and (df["FTAG"] >= 0).all()


def test_dixon_coles_matrix_invariants():
    pred = build_dixon_coles_prediction("AMB", "MIL", lambda_h=2.0035, lambda_a=1.0549, rho=-0.0321)
    assert pred.matrix.shape == (11, 11)
    assert np.all(pred.matrix >= 0.0)
    assert math.isclose(float(np.sum(pred.matrix)), 1.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(pred.prob_home + pred.prob_draw + pred.prob_away, 1.0, rel_tol=1e-9, abs_tol=1e-9)
    assert 0.0 <= pred.shannon_entropy <= 1.0
    assert pred.modal_prob > 0.0


def test_dixon_coles_low_score_correlation():
    p_indep = build_dixon_coles_prediction("TOR", "AMB", lambda_h=1.20, lambda_a=1.50, rho=0.0)
    p_corr = build_dixon_coles_prediction("TOR", "AMB", lambda_h=1.20, lambda_a=1.50, rho=-0.08)
    # Negative rho increases 0-0 and 1-1 probabilities relative to independent Poisson
    assert p_corr.matrix[0, 0] > p_indep.matrix[0, 0]
    assert p_corr.matrix[1, 1] > p_indep.matrix[1, 1]


def test_bayes_engine_fit_and_json_parity():
    df, _ = load_default_dataset()
    model = TimeDecayedBayesDixonColes(xi=0.0020, prior_var=0.25).fit(df)
    cal = json.loads((REPO_ROOT / "calibrated_serie_a_params.json").read_text(encoding="utf-8"))

    assert cal["totalMatches"] == len(df) == 1140
    assert cal["uniqueClubs"] == len(model.teams) == 25
    assert math.isclose(model.mu, cal["global"]["mu"], abs_tol=1e-3)
    assert math.isclose(model.home_adv, cal["global"]["homeAdvantage"], abs_tol=1e-3)
    assert math.isclose(model.rho, cal["global"]["rho"], abs_tol=1e-3)

    # Verify Lombardy Derby (AMB vs MIL) matches pinned frontend artifact
    pred = model.predict("AMB", "MIL")
    assert math.isclose(pred.lambda_h, 2.0035, abs_tol=1e-2)
    assert math.isclose(pred.lambda_a, 1.0549, abs_tol=1e-2)


def test_empirical_bayes_shrinkage_monotonicity():
    df, _ = load_default_dataset()
    m_loose = TimeDecayedBayesDixonColes(xi=0.0020, prior_var=1.00).fit(df)
    m_tight = TimeDecayedBayesDixonColes(xi=0.0020, prior_var=0.10).fit(df)
    # Stronger Gaussian L2 prior strictly reduces posterior parameter norm
    norm_loose = float(np.sum(m_loose.attack ** 2) + np.sum(m_loose.defense ** 2))
    norm_tight = float(np.sum(m_tight.attack ** 2) + np.sum(m_tight.defense ** 2))
    assert norm_tight < norm_loose


def test_proper_scoring_rules_bounds():
    actual_home = outcome_to_one_hot(2, 0)
    perfect = np.array([1.0, 0.0, 0.0])
    good = np.array([0.7, 0.2, 0.1])
    bad = np.array([0.1, 0.2, 0.7])

    assert ranked_probability_score(perfect, actual_home) == 0.0
    assert brier_score(perfect, actual_home) == 0.0
    assert ranked_probability_score(good, actual_home) < ranked_probability_score(bad, actual_home)
    assert multiclass_log_loss(good, actual_home) < multiclass_log_loss(bad, actual_home)
    assert brier_score(good, actual_home) < brier_score(bad, actual_home)


def test_xt_markov_contraction_and_json_parity():
    sol = solve_expected_threat("balanced")
    assert sol["dimensions"] == {"nx": 12, "ny": 8, "states": 96}
    assert 0.0 < sol["convergence"]["spectralRadius"] < 1.0
    assert sol["convergence"]["analyticalMaxDiff"] < 1e-7

    artifact_disk = json.loads((REPO_ROOT / "xt_markov_grid.json").read_text(encoding="utf-8"))
    artifact_live = build_full_xt_artifact()
    assert np.allclose(np.array(artifact_disk["grid"]), np.array(artifact_live["grid"]), atol=1e-5)


def test_xt_spatial_symmetry_and_zone_ordering():
    sol = solve_expected_threat("balanced")
    grid = np.array(sol["grid"])
    assert grid.shape == (8, 12)
    assert np.all((grid > 0.0) & (grid < 1.0))
    # Top-bottom mirror symmetry across pitch midline
    assert np.allclose(grid, grid[::-1, :], atol=1e-5)
    # Spatial threat hierarchy: Six-Yard Box > Zone 14 > Own Build-Up
    s = sol["summary"]
    assert s["centralBoxPeakXT"] > s["zone14XT"] > s["ownBuildUpXT"]


def test_xt_wing_overload_increases_flank_threat():
    art = build_full_xt_artifact()
    flank_bal = art["regimeSummaries"]["balanced"]["flankDeliveryXT"]
    flank_wing = art["regimeSummaries"]["wingOverload"]["flankDeliveryXT"]
    assert flank_wing > flank_bal


def test_laplace_posterior_credible_intervals():
    df, _ = load_default_dataset()
    model = TimeDecayedBayesDixonColes(xi=0.0020, prior_var=0.25).fit(df)
    team_ci = compute_laplace_posterior_se(df, model)
    assert len(team_ci) == 25
    # 3-season regular club (n=114) must have strictly tighter attack CI than 1-season club (n=38)
    assert team_ci["AMB"]["alpha_ci95"] < team_ci["PIS"]["alpha_ci95"]
    assert team_ci["TOR"]["alpha_ci95"] < team_ci["VEN"]["alpha_ci95"]


def test_frontend_html_pipeline_integrity():
    html = (REPO_ROOT / "index.html").read_text(encoding="utf-8")
    for required_token in [
        "calibrated_serie_a_params.json",
        "xt_markov_grid.json",
        "FALLBACK_BAYES_SERIE_A",
        "FALLBACK_XT_MARKOV",
        "loadCalibratedSerieAParams",
        "loadXtMarkovGrid",
        "toggleXtHeatmap",
    ]:
        assert required_token in html, f"Missing live bridge token in index.html: {required_token}"
    # Lombardy Derby must remain pinned first for immediate evaluation
    assert html.find("classic-spotlight-milano-derby") < html.find("classic-on-this-day-tor-mad")


def test_fastapi_endpoints():
    if importlib.util.find_spec("fastapi") is None or importlib.util.find_spec("httpx") is None:
        return  # Installed unconditionally in GitHub Actions CI
    from fastapi.testclient import TestClient
    from api_server import app

    client = TestClient(app)

    r_health = client.get("/health")
    assert r_health.status_code == 200
    body_h = r_health.json()
    assert body_h["status"] == "ok"
    assert body_h["matches_calibrated"] == 1140
    assert body_h["xt_states"] == 96

    r_pred = client.post("/predict", json={"home_team": "AMB", "away_team": "MIL"})
    assert r_pred.status_code == 200
    body_p = r_pred.json()
    assert body_p["home"] == "AMB" and body_p["away"] == "MIL"
    assert math.isclose(sum(body_p["probabilities"].values()), 100.0, abs_tol=0.1)

    r_bad = client.post("/predict", json={"home_team": "TOR", "away_team": "TOR"})
    assert r_bad.status_code == 400

    r_xt = client.get("/xt-grid", params={"tactic": "wingOverload"})
    assert r_xt.status_code == 200
    body_x = r_xt.json()
    assert body_x["tacticMode"] == "wingOverload"
    assert len(body_x["grid"]) == 8 and len(body_x["grid"][0]) == 12

    r_xt_bad = client.get("/xt-grid", params={"tactic": "invalidMode"})
    assert r_xt_bad.status_code == 422
