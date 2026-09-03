# nav-twin

Digital twin MVP for a fuel-injected aero piston engine: a parametric
thermodynamic model predicts what every sensor should read, and the twin
flags the gap between prediction and observation, normalised by a computed
Monte Carlo uncertainty band. Faults are injected as single-parameter
physical perturbations; the signatures fall out of the physics.

## Layout

```
simulator/        the engine side: physics, fault injection, missions
  physics/        atmosphere -> intake -> combustion -> cycle -> thermal -> oil
  mission.py      telemetry generator (1 Hz frames, fault scheduling)
  verify_*.py     the checks; run these before trusting any number
app/              the twin side: residuals, anomaly, diagnosis, server, UI
  twin/           twin core (pure Python, no web dependencies)
  server.py       FastAPI: websocket state stream + REST controls
  demo.py         scripted demo runner
  verify_twin.py  headless end-to-end checks
  dashboard/      React + Vite + TypeScript operator interface
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# verify the physics and the twin (all must pass)
cd simulator && python verify_sanity_checks.py && python verify_fault_signatures.py \
  && python verify_uncertainty.py && python verify_order_spectrum.py \
  && python verify_mission.py && python verify_sigma_table.py
cd ../app && python verify_twin.py

# run the app
python server.py          # serves the API and the dashboard on :8000
```

Dashboard development build:

```bash
cd app/dashboard
npm install
npm run build             # output lands in dist/, which server.py serves
npm run dev               # or a Vite dev server with hot reload
```

## The ML layer

The twin ships a rule-based pipeline (band + sign-pattern diagnosis) and a
learned layer beside it. The learned layer trains on a simulator corpus and
deploys only if it passes its gates:

```bash
cd simulator
python ml_corpus.py --healthy 600 --fault 320 --workers 8   # ~2 h on 8 cores
python verify_corpus.py          # coverage, determinism, class separation

cd ../app
python ml_anomaly.py             # isolation forest on healthy features
python verify_anomaly.py         # gate: FPR calibration, latency, coverage
python ml_classify.py            # gradient boosting + attribution
python verify_classify.py        # gate: accuracy, per-class recall, SHAP
                                 # signature consistency
python ml_rul.py                 # observation model for the particle filter
python verify_rul.py             # gate: tracking, no false degradation,
                                 # bounded framing
python verify_ml.py              # integration gate, models present
python verify_twin.py            # must pass with AND without models
```

Artifacts land in `app/models/`. When present, `Twin.step` adds an `ml`
block (forest score, model diagnosis, bounded RUL) alongside the unchanged
rule-based diagnosis. When absent, the twin is rules-only and everything
still passes. RUL output is a bounded relative degradation index, never a
certified time-to-failure.

## Rules of the house

- Every number the system shows comes from the model or a verify script.
  No invented placeholders.
- Fault signatures are never hand-coded; they emerge from parameter
  perturbations of the physics.
- The uncertainty band is computed (Monte Carlo propagation, deployed as a
  lookup table), not a tuned threshold.
