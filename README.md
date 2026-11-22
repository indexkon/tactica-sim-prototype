# TACTICA.SIM — Quantitative Football Simulation, Spatial xT Engine & Coach Workbench

[![Stage](https://img.shields.io/badge/TRL-4--5%20Pre--Commercial%20Prototype-10B981)](#)
[![Compliance](https://img.shields.io/badge/Italy-D.L.%20179%2F2012%20%26%20L.%20193%2F2024-2563EB)](#regulatory--compliance-firewall)
[![ATECO](https://img.shields.io/badge/ATECO-62.01.00%20Software%20Production-F59E0B)](#)
[![Languages](https://img.shields.io/badge/UI-IT%20%7C%20EN%20%7C%20ZH-6366F1)](#)

**TACTICA.SIM S.r.l.** (planned headquarters: **Turin, Piedmont, Italy**) is a Sports Tech SaaS & Sports EdTech software platform combining a **Time-Decayed Empirical Bayes Dixon-Coles statistical engine**, a **12×8 Spatial Expected Threat (xT) absorbing Markov solver**, a **4,000× Monte Carlo tactical match-prep workbench** for grassroots football coaches, and an interactive **2D spatial pitch simulator & tactical academy**.

> **Live Interactive Prototype (Zero Registration Required):**  
> 👉 [https://indexkon.github.io/tactica-sim-prototype/](https://indexkon.github.io/tactica-sim-prototype/) *(Defaults to Italian `IT`; supports live `IT / EN / ZH` switching; loads Python-calibrated Serie A parameters via `calibrated_serie_a_params.json` and 12×8 Markov xT surfaces via `xt_markov_grid.json`)*

---

## Repository Structure

All files are self-contained in the repository root for frictionless technical evaluation:

```text
tactica-sim-prototype/
├── index.html                        # v3.13 prototype (Bayes JSON + live 2D xT heatmap)
├── calibrated_serie_a_params.json    # Python-fitted Bayes parameters (1,140 real Serie A)
├── xt_markov_grid.json               # Converged 12×8 Spatial Markov xT surfaces (4 regimes)
├── tactica_bayes_dc_engine.py        # Time-Decayed Empirical Bayes Dixon-Coles ETL engine
├── tactica_xt_engine.py              # 12×8 Spatial Expected Threat (xT) Markov solver
├── api_server.py                     # FastAPI inference API (/health, /predict, /xt-grid)
├── serie_a_cleanroom_2023_2026.csv   # 1,140 anonymized real Serie A scores (2023–26)
├── requirements.txt                  # Python runtime dependencies
├── LICENSE                           # Prototype repository license
└── README.md                         # Technical overview & reproduction guide
```

---

## Core Architecture & Live Data Pipeline

```text
serie_a_cleanroom_2023_2026.csv (1,140 real Serie A final scorelines, 2023–26)
        │
        ▼
tactica_bayes_dc_engine.py  (Time-Decayed Empirical Bayes Dixon-Coles MLE, L-BFGS-B)
        ├──► calibrated_serie_a_params.json ──┐
        │                                     ▼
tactica_xt_engine.py (12×8 Bellman xT) ──► xt_markov_grid.json ──► index.html (Live Browser Inference)
        │                                                                ▲
        └───────────────────────────────► api_server.py (/health, /predict, /xt-grid)
```

### 1. Client Interactive Prototype (`index.html` — v3.13 MIMIT Bayes & xT Spatial)
A clean-room, mobile-first HTML5/JavaScript application running natively in desktop and mobile browsers across five functional modules:
* **Live Python Calibration Bridge:** Automatically fetches `calibrated_serie_a_params.json` at startup (with an identical embedded offline fallback) and executes the exact log-linear Empirical Bayes Dixon-Coles arrival rate model (`λ_H = exp(μ + α_H + β_A + γ)`, `λ_A = exp(μ + α_A + β_H)`, `ρ = -0.0321`), starting with the Lombardy Derby (`AMB vs MIL`) pinned first on the home screen.
* **Live 12×8 Spatial Markov xT Heatmap & Cumulative xT Telemetry:** Fetches `xt_markov_grid.json` at startup, renders a toggleable (`xT ●`) 12×8 thermal threat grid on the 2D Canvas pitch oriented to the user's attacking direction, dynamically switches Markov surfaces when tactical commands (*High Press*, *Catenaccio*, *Wing Overload*) are triggered, and tracks unscaled progressive Expected Threat (`xT`) alongside `xG` in both the live HUD and post-match summary.
* **Coach Workbench (*Studio Allenatore* — Core Pro SaaS / B2B Engine):** Custom squad attack/defense parameters (`GF`, `GA`, starting fitness, formation, tactical archetype), venue & league goal baseline controls, adjustable Dixon-Coles correlation (`ρ ∈ [-0.15, 0.00]`), **1,000× per tactic (4,000× total) Monte Carlo sensitivity comparison**, scouting profile library, and structured pre-match tactical brief export.
* **Quantitative Match Matrix & 2D Pitch Simulator (*Partite*):** 60 original clean-room clubs across 6 historical-style leagues, **11×11 Dixon-Coles joint scoreline heatmap**, interactive pre-match **What-If tactical lab**, Shannon uncertainty entropy (`H`), and a 60-second **HTML5 Canvas 2D pitch** supporting 10 tactical formations.
* **Custom Club & Virtual League (*Lega*), Academy (*Accademia*) & SaaS Hub (*Profilo*):** 3-tier virtual league, 5 illustrated tactical/quantitative micro-lessons, and two-tier SaaS architecture (**Elite Player** €3.99/mo, **Pro Coach** €9.99/mo) plus standardized B2B youth academy licensing (€600/yr per club for 5 Pro seats).

### 2. Quantitative Backend & Calibration Engine (`tactica_bayes_dc_engine.py`)
* **Time-Decayed Empirical Bayes Dixon-Coles MLE:** Estimates team attack/defense strength `(αᵢ, βᵢ)`, home advantage (`γ`), and low-score dependence (`ρ`) via `L-BFGS-B` optimization with exponential daily time decay `wₖ = exp(-ξ · Δtₖ)` and Gaussian L2 shrinkage priors.
* **Walk-Forward Out-of-Sample Backtest (No Look-Ahead Bias):** Evaluates rolling out-of-sample predictions against the static independent Poisson baseline across **Ranked Probability Score (RPS)**, **Multi-class Log-Loss**, and **Brier Score**.
* **Automated JSON Artifact Export:** Generates `calibrated_serie_a_params.json` directly from `serie_a_cleanroom_2023_2026.csv` (1,140 real Serie A final scorelines, 2023–24 to 2025–26, strictly stripped of any bookmaker odds and mapped to Clean-Room 3-letter geographic codes).

### 3. 12×8 Spatial Expected Threat (xT) Markov Engine (`tactica_xt_engine.py`)
* **Bellman Value Iteration on Absorbing Markov Chain (Singh, 2018):** Discretizes the 105m × 68m pitch into 96 states (`12 × 8`) and iteratively solves `xT(z) = s(z)·g(z) + m(z)·Σ P_succ(z → z')·xT(z')` until `max |ΔxT| < 1e-8`, cross-verified against the exact analytical matrix solution `(I - M)⁻¹(s · g)`.
* **Tactical Regime Surfaces:** Exports converged 12×8 threat surfaces across four tactical stances (`balanced`, `highPress`, `catenaccio`, `wingOverload`) to `xt_markov_grid.json` for live browser rendering.

### 4. FastAPI Inference Microservice (`api_server.py`)
* `GET /health` — Model calibration status, fitted `ρ`, home advantage `γ`, and xT Markov spectral radius.
* `POST /predict` — Computes `xG_H`, `xG_A`, 1X2 probabilities, modal scoreline, Shannon entropy `H`, and 6×6 scoreline matrix with optional tactical intervention multipliers.
* `GET /xt-grid?tactic=balanced` — Returns the converged 12×8 spatial Markov xT matrix and zone summary for the requested tactical stance.

---

## Quick Start & Technical Verification

### 1. Run the Frontend Prototype
Open the [GitHub Pages Live Demo](https://indexkon.github.io/tactica-sim-prototype/) directly, or open `index.html` locally in any modern browser.

### 2. Run the Python Calibration, Spatial xT Solver & Backtests
Requires Python 3.10+. Install dependencies:

```bash
pip install -r requirements.txt
```

Run the walk-forward Serie A backtest and regenerate `calibrated_serie_a_params.json`:

```bash
python tactica_bayes_dc_engine.py
```

Run the 12×8 Spatial Expected Threat Markov solver and regenerate `xt_markov_grid.json`:

```bash
python tactica_xt_engine.py
```

#### Empirical Walk-Forward Backtest (760 Out-of-Sample Real Serie A Matches, 2024–2026)

Trained on 2023–24 (380-match warm-up) and refitted every 14 days across 2024–25 and 2025–26 using `serie_a_cleanroom_2023_2026.csv` (1,140 real Serie A final scorelines, 25 clubs, zero bookmaker odds, anonymized to Clean-Room geographic codes):

| Metric (Lower is Better) | Static Independent Poisson (Baseline) | Time-Decayed Empirical Bayes Dixon-Coles | Error Reduction |
| :--- | :---: | :---: | :---: |
| **Ranked Probability Score (RPS)** | `0.20172` | **`0.19729`** | **-2.20%** |
| **Multi-class Log-Loss** | `1.00804` | **`0.98832`** | **-1.96%** |
| **Brier Score** | `0.60338` | **`0.59075`** | **-2.09%** |

*(Full 3-season fitted parameters: μ = 0.1087, Home Advantage γ = 0.1091, Dixon-Coles ρ = -0.0321, ξ = 0.0020, σ²_prior = 0.25.)*

#### 12×8 Spatial Markov Expected Threat (xT) Convergence Summary

| Solver Metric | Baseline (`balanced`) | Wing Overload (`wingOverload`) | High Press (`highPress`) | Catenaccio (`catenaccio`) |
| :--- | :---: | :---: | :---: | :---: |
| **Zone 14 Mean xT** | `0.03760` | `0.03622` | **`0.04557`** | `0.04299` |
| **Flank Delivery xT** | `0.04135` | **`0.04788`** | `0.04806` | `0.04462` |
| **Six-Yard Box Peak xT** | `0.39189` | `0.39111` | `0.41324` | **`0.41561`** |

*(Baseline Bellman contraction: spectral radius ρ(M) = 0.53243 < 1, converged in 28 iterations to tol = 1e-8; max deviation vs. exact linear solver (I - M)⁻¹(s·g) = 6.39e-9.)*

### 3. Launch the FastAPI Microservice & Swagger UI

```bash
uvicorn api_server:app --reload --port 8000
```

Open interactive API docs at **`http://127.0.0.1:8000/docs`** or test `/predict` and `/xt-grid`:

```bash
curl -X POST "http://127.0.0.1:8000/predict"   -H "Content-Type: application/json"   -d '{"home_team":"TOR","away_team":"AMB","tactic_mult_h":1.24,"tactic_mult_a":1.01}'

curl "http://127.0.0.1:8000/xt-grid?tactic=wingOverload"
```

---

## Regulatory & Compliance Firewall

Designed to meet **Italia Startup Visa (MIMIT)** standards (*D.L. 179/2012* Art. 25, *Legge 193/2024* Art. 28, ATECO `62.01.00`):
* **Zero Gambling / ADM Exemption:** Pure coaching & educational interface using star-rated tactical drills (`★☆☆ / ★★☆ / ★★★`) with zero odds or betting multipliers. Virtual coins and XP operate strictly in a closed loop with zero cash-out, zero P2P trading, and zero wagering.
* **100% Clean-Room IP:** All 60 clubs, geographic codes (e.g., `TOR` - *Torino Zebre*), and geometric SVG crests are original creations; statistical calibration relies solely on public historical final scorelines stripped of any third-party commercial data or odds.
* **IP Scope Notice:** This public repository serves as a pre-commercial TRL 4–5 demonstration prototype for technical evaluation. Post-incorporation production backend modules (FastAPI/PostgreSQL cluster, automated Bayesian ETL, adaptive AI coaching diagnostics, and B2B academy portal) will be developed as proprietary software and registered with **SIAE** (*Registro Pubblico Speciale per i Programmi per Elaboratore*, Codice 068).
