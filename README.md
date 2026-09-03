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

## Rules of the house

- Every number the system shows comes from the model or a verify script.
  No invented placeholders.
- Fault signatures are never hand-coded; they emerge from parameter
  perturbations of the physics.
- The uncertainty band is computed (Monte Carlo propagation, deployed as a
  lookup table), not a tuned threshold.
