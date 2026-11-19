# TACTICA.SIM — Bivariate Poisson Football Simulation & Sports EdTech Engine (TRL 6 Prototype)

Interactive web-based prototype of **TACTICA.SIM**, a 2D football tactical simulation and quantitative sports EdTech engine developed for **TACTICA.SIM S.r.l.** (Turin, Italy).

## Overview

This repository hosts the self-contained **TRL 6** interactive demonstration (`index.html`) combining:

- **Stochastic Match Modeling**: Bivariate Poisson and Dixon-Coles low-score correlation adjustments for football match probability estimation and tactical sensitivity analysis.
- **2D Tactical Simulation Canvas**: Real-time interactive pitch visualization allowing coaches, analysts, and students to test formations, pressing intensity, and tactical trade-offs.
- **Sports EdTech & Telemetry Interface**: Client-side analytical dashboard displaying expected goals (xG), scoreline probability matrices, and scenario comparisons without requiring external backend infrastructure.

## Live Demo (GitHub Pages)

Once uploaded to GitHub, enable **GitHub Pages** to generate a shareable live URL for technical reviewers and institutional committees:

1. Open your repository **Settings → Pages**.
2. Under **Build and deployment → Branch**, select **`main`** and folder **`/ (root)`**, then click **Save**.
3. Wait 1–2 minutes for the live URL (`https://indexkon.github.io/tactica-sim-prototype/`) to deploy.

## Local Usage

No build step or server installation is required:

1. Download or clone this repository.
2. Open `index.html` directly in any modern desktop or mobile browser (Chrome, Safari, Firefox, Edge).

## Repository Structure

```text
.
├── index.html      # Self-contained TRL 6 interactive simulation prototype
├── README.md       # Project documentation and deployment guide
├── LICENSE         # Proprietary software copyright notice
└── .gitignore      # Standard OS and editor ignore rules
```

## Intellectual Property

All source code, mathematical modeling implementations, and interface assets are proprietary to **TACTICA.SIM S.r.l.** (in formation) and intended for software copyright registration with the **SIAE** (*Registro Pubblico Speciale per i Programmi per Elaboratore*). See `LICENSE` for details.
