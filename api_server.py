"""
TACTICA.SIM S.r.l. — Pan-European FastAPI Inference & Spatial xT Microservice (TRL 5 Demo)
Run locally: uvicorn api_server:app --reload --port 8000
Swagger UI:  http://127.0.0.1:8000/docs
"""

from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from tactica_bayes_dc_engine import (
    LEAGUE_META,
    TimeDecayedBayesDixonColes,
    load_europe_big5_dataset,
)
from tactica_xt_engine import build_full_xt_artifact

app = FastAPI(title="TACTICA.SIM Pan-European Bayes Dixon-Coles & Spatial xT Engine", version="3.15-mimit")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Bootstraps calibration across all 5 European leagues (5,256 real Clean-Room matches, 119 clubs)
_df_big5, _data_source = load_europe_big5_dataset()
league_models: Dict[str, TimeDecayedBayesDixonColes] = {}
team_to_league: Dict[str, str] = {}

for _lg_id in LEAGUE_META:
    _sub = _df_big5[_df_big5["League"] == _lg_id]
    if len(_sub) > 0:
        _m = TimeDecayedBayesDixonColes(xi=0.0020, prior_var=0.25).fit(_sub)
        league_models[_lg_id] = _m
        for _tm in _m.teams:
            team_to_league[_tm] = _lg_id

# Backward-compatible Serie A default model alias
model = league_models.get("calcio") or next(iter(league_models.values()))

# Solves 12x8 Spatial Expected Threat (xT) absorbing Markov chain at startup
xt_artifact = build_full_xt_artifact()


class PredictRequest(BaseModel):
    home_team: str = Field("AMB", examples=["AMB"])
    away_team: str = Field("MIL", examples=["MIL"])
    league: Optional[str] = Field(None, examples=["calcio"])
    tactic_mult_h: float = Field(1.0, ge=0.5, le=1.8)
    tactic_mult_a: float = Field(1.0, ge=0.5, le=1.8)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "3.15-mimit",
        "model": "PanEuropeanTimeDecayedBayesDixonColes + SpatialMarkovXT",
        "data_source": _data_source,
        "leagues_calibrated": len(league_models),
        "matches_calibrated": int(sum(m.matches_fitted for m in league_models.values())),
        "teams_calibrated": int(sum(len(m.teams) for m in league_models.values())),
        "serie_a_rho": round(model.rho, 4),
        "serie_a_home_advantage": round(model.home_adv, 4),
        "xt_states": xt_artifact["dimensions"]["states"],
        "xt_spectral_radius": xt_artifact["convergence"]["spectralRadius"],
    }


@app.get("/leagues")
def list_leagues():
    return {
        lg_id: {
            "name": LEAGUE_META[lg_id]["name"],
            "matches": m.matches_fitted,
            "clubs": len(m.teams),
            "mu": round(m.mu, 4),
            "home_advantage": round(m.home_adv, 4),
            "rho": round(m.rho, 4),
        }
        for lg_id, m in league_models.items()
    }


@app.post("/predict")
def predict(req: PredictRequest):
    h_code = req.home_team.upper()
    a_code = req.away_team.upper()
    if h_code == a_code:
        raise HTTPException(status_code=400, detail="home_team and away_team must differ")

    target_lg = req.league.lower() if req.league else None
    if target_lg is None:
        lg_h = team_to_league.get(h_code)
        lg_a = team_to_league.get(a_code)
        target_lg = lg_h if (lg_h and lg_h == lg_a) else (lg_h or "calcio")

    if target_lg not in league_models:
        raise HTTPException(status_code=404, detail=f"Unknown league '{target_lg}'")

    pred = league_models[target_lg].predict(
        h_code,
        a_code,
        tactic_mult_h=req.tactic_mult_h,
        tactic_mult_a=req.tactic_mult_a,
    )
    out = pred.to_frontend_json()
    out["league"] = target_lg
    return out


@app.get("/xt-grid")
def get_xt_grid(
    tactic: str = Query("balanced", pattern="^(balanced|highPress|catenaccio|wingOverload)$")
):
    return {
        "schemaVersion": xt_artifact["schemaVersion"],
        "tacticMode": tactic,
        "dimensions": xt_artifact["dimensions"],
        "convergence": {k: v for k, v in xt_artifact["convergence"].items() if k != "trace"},
        "summary": xt_artifact["regimeSummaries"][tactic],
        "grid": xt_artifact["regimes"][tactic],
    }
