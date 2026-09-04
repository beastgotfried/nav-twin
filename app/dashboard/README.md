# Sitaara operator dashboard

React + Vite + TypeScript. Renders the twin state stream defined in
[`../README.md`](../README.md); every number on screen comes from that
protocol. Visual spec: `09-Visuals/out/fig_dashboard_mockup.png`, design
tokens from `09-Visuals/style.py` (accent meanings are fixed: blue
predicted, red observed/fault, orange caution, green nominal).

## Develop

```
npm install
npm run dev
```

Vite proxies `/ws` and `/api` to the twin backend on port 8734
(see `vite.config.ts`). With no backend running, use the mock, which
speaks the same protocol and drives the real simulator:

```
MOCK_TICK_S=0.05 ../../.venv/bin/python mock/mock_server.py
curl -X POST localhost:8734/api/control -H 'Content-Type: application/json' \
  -d '{"action":"start","scenario":"endurance","fault_events":[]}'
```

`mock/mock_server.py` is a development stand-in, not the MVP backend:
observed telemetry and predictions come from `08-Simulator`, but the alarm
persistence window and diagnosis ranking are labelled mock heuristics that
`twin/anomaly.py` and `twin/diagnose.py` replace.

## Build

```
npm run build
```

Outputs `dist/`, which `10-Twin/server.py` serves at `GET /`. The client
always talks to same-origin relative paths (`/ws`, `/api/...`), so the
same build works behind the dev proxy and in production.
