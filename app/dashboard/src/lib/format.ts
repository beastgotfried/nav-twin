/**
 * Display formatting. The mockup (09-Visuals/out/fig_dashboard_mockup.png)
 * groups thousands with thin spaces ("5 000 rpm") and always shows the sign
 * on residuals ("z +2.13"); these helpers keep that consistent everywhere.
 */

const THIN = "\u2009";

/** 5000 -> "5 000", 21600.4 with decimals=1 -> "21 600.4". */
export function fmtGrouped(value: number, decimals = 0): string {
  const fixed = value.toFixed(decimals);
  const [int, frac] = fixed.split(".");
  const sign = int.startsWith("-") ? "-" : "";
  const digits = sign ? int.slice(1) : int;
  const grouped = digits.replace(/\B(?=(\d{3})+(?!\d))/g, THIN);
  return sign + grouped + (frac ? "." + frac : "");
}

/** Residuals always carry their sign: 2.134 -> "+2.13". */
export function fmtZ(z: number): string {
  return (z >= 0 ? "+" : "-") + Math.abs(z).toFixed(2);
}

/** Kelvin to the dashboard's display unit, degrees C. */
export function fmtTempC(kelvin: number): string {
  return fmtGrouped(kelvin - 273.15);
}

/** Pa -> kPa, grouped. */
export function fmtKPa(pa: number): string {
  return fmtGrouped(pa / 1000);
}

/** Mission clock, "T+06:00" under an hour, "T+1:06:00" past it. */
export function fmtClock(tS: number): string {
  const s = Math.max(0, Math.floor(tS));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  const mm = String(m).padStart(2, "0");
  const ss = String(sec).padStart(2, "0");
  return h > 0 ? `T+${h}:${mm}:${ss}` : `T+${mm}:${ss}`;
}

/** Elapsed time without the T+ prefix, for report rows. */
export function fmtDuration(tS: number): string {
  return fmtClock(tS).replace("T+", "");
}

export function fmtPct(confidence: number): string {
  return `${Math.round(confidence * 100)}%`;
}

/** Scenario id to display text: "high_altitude" -> "high altitude". */
export function fmtScenario(id: string): string {
  return id.replace(/_/g, " ");
}
