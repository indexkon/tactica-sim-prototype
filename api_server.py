"""
TACTICA.SIM S.r.l. — FastAPI Inference Microservice (TRL 5 Demo)
Run locally: uvicorn api_server:app --reload --port 8000
Swagger UI:  http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from tactica_bayes_dc_engine import (
    TimeDecayedBayesDixonColes,
    load_default_dataset,
)

app = FastAPI(title="TACTICA.SIM Dixon-Coles & Bayes Engine", version="3.11-mimit")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Bootstraps calibration pipeline on 3-season Clean-Room real Serie A scorelines (1,140 matches)
_df, _data_source = load_default_dataset()
model = TimeDecayedBayesDixonColes(xi=0.0020, prior_var=0.25).fit(_df)


class PredictRequest(BaseModel):
    home_team: str = Field("TOR", examples=["TOR"])
    away_team: str = Field("AMB", examples=["AMB"])
    tactic_mult_h: float = Field(1.0, ge=0.5, le=1.8)
    tactic_mult_a: float = Field(1.0, ge=0.5, le=1.8)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "TimeDecayedBayesDixonColes",
        "data_source": _data_source,
        "matches_calibrated": model.matches_fitted,
        "teams_calibrated": len(model.teams),
        "rho": round(model.rho, 4),
        "home_advantage": round(model.home_adv, 4),
    }


@app.post("/predict")
def predict(req: PredictRequest):
    if req.home_team.upper() == req.away_team.upper():
        raise HTTPException(status_code=400, detail="home_team and away_team must differ")
    pred = model.predict(
        req.home_team.upper(),
        req.away_team.upper(),
        tactic_mult_h=req.tactic_mult_h,
        tactic_mult_a=req.tactic_mult_a,
    )
    return pred.to_frontend_json()
