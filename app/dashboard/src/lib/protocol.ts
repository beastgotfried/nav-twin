/**
 * Wire protocol types, mirroring app/README.md exactly. The twin state
 * is emitted once per mission timestep by server.py (WS /ws) and rendered
 * here without modification: every number on screen traces back to one of
 * these fields.
 */

export type AlarmLevel = "nominal" | "caution" | "warning";
export type ChannelStatus = "nominal" | "caution" | "warning";

export interface CylinderState {
  n: number;
  EGT_K: number;
  CHT_K: number;
  EGT_pred_K: number;
  CHT_pred_K: number;
  sigma_EGT_K: number;
  sigma_CHT_K: number;
  z_EGT: number;
  z_CHT: number;
  status: ChannelStatus;
}

export interface OilState {
  p_Pa: number;
  T_K: number;
  p_pred_Pa: number;
  T_pred_K: number;
  z_p: number;
  z_T: number;
}

export interface AlarmState {
  active: boolean;
  level: AlarmLevel;
  since_t_s: number | null;
  channel: string | null;
  cylinder: number | null;
}

export interface DiagnosisEntry {
  rank: number;
  label: string;
  confidence: number;
  evidence: string[];
}

export interface TwinInputs {
  N_rpm: number;
  MAP_Pa: number;
  altitude_m: number;
  fuel_flow_total_kg_h: number;
}

export interface TwinState {
  t_s: number;
  scenario: string;
  inputs: TwinInputs;
  cylinders: CylinderState[];
  oil: OilState;
  alarm: AlarmState;
  diagnosis: DiagnosisEntry[];
}

export type ServerMessage =
  | { type: "hello"; scenarios: string[] }
  | { type: "state"; state: TwinState };

/** GET /api/history response shape (series since the last reset). */
export interface HistoryResponse {
  t_s: number[];
  cylinders: Record<
    string,
    {
      EGT_K: number[];
      EGT_pred_K: number[];
      sigma_EGT_K: number[];
      CHT_K: number[];
      CHT_pred_K: number[];
      sigma_CHT_K: number[];
    }
  >;
}

/** Fault kinds accepted by POST /api/control, from simulator/physics/faults.py. */
export const FAULT_KINDS = [
  "misfire",
  "injector_restriction",
  "detonation",
  "cooling_degradation",
  "bearing_wear",
  "sensor_drift",
  "turbo_degradation",
] as const;
export type FaultKind = (typeof FAULT_KINDS)[number];

/** Engine-wide faults take no cylinder argument (faults.py FaultSpec). */
export const ENGINE_WIDE_FAULTS: readonly FaultKind[] = [
  "bearing_wear",
  "turbo_degradation",
];

export const SENSOR_CHANNELS = ["EGT_K", "CHT_K"] as const;
export type SensorChannel = (typeof SENSOR_CHANNELS)[number];

export const FALLBACK_SCENARIOS = [
  "high_altitude",
  "endurance",
  "hot_weather",
  "rapid_throttle",
];
