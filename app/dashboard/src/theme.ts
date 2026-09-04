/**
 * The Sitaara Astryx theme.
 *
 * The four accents are NOT design choices to be revisited here. They carry
 * fixed meaning across the whole project (09-Visuals/style.py, and the
 * instrument convention in 10-Twin/README.md):
 *
 *     blue    predicted / the physics model
 *     red     genuinely wrong: fault, flagged, band exceeded
 *     orange  caution, or the second physical quantity. NEVER "series 2"
 *     green   healthy, nominal, inside the band
 *
 * They are reproduced here byte for byte so an Astryx Badge and a
 * hand-drawn SVG trace mean the same thing by colour. Measured against the
 * dark ground, all four clear WCAG AA unchanged (blue 6.53:1, green 7.76:1,
 * orange 6.23:1, red 5.22:1 on the ground; 5.77 / 6.85 / 5.50 / 4.61 on a
 * panel), which is why going dark cost the palette nothing.
 *
 * Values are plain strings rather than [light, dark] tuples: this app is
 * dark only, so a mode switch must not be able to change what an operator
 * sees.
 */

import { defineTheme } from "@astryxdesign/core";

/** Locked accents. Do not edit without changing 09-Visuals/style.py too. */
export const BLUE = "#0099FF";
export const GREEN = "#2FBB45";
export const ORANGE = "#DC762D";
export const RED = "#FB2C55";

/** Dark neutrals. Near-black with a slight cool cast, never pure #000. */
export const GROUND = "#0A0C0E";
export const GROUND_LIFT = "#101316";
export const PANEL = "#171B1F";
export const PANEL_LIFT = "#1D2226";

export const TEXT = "#E8EBED";
export const TEXT_DIM = "#98A1A8";
export const TEXT_FAINT = "#7E888F";

export const HAIRLINE = "rgba(255, 255, 255, 0.10)";
export const HAIRLINE_STRONG = "rgba(255, 255, 255, 0.20)";

export const sitaaraTheme = defineTheme({
  name: "sitaara",
  tokens: {
    // --- the locked four ---
    "--color-accent": BLUE,
    "--color-accent-muted": "rgba(0, 153, 255, 0.18)",
    "--color-on-accent": GROUND,
    "--color-text-accent": BLUE,
    "--color-icon-accent": BLUE,

    "--color-success": GREEN,
    "--color-success-muted": "rgba(47, 187, 69, 0.18)",
    "--color-on-success": GROUND,

    "--color-warning": ORANGE,
    "--color-warning-muted": "rgba(220, 118, 45, 0.18)",
    "--color-on-warning": GROUND,

    "--color-error": RED,
    "--color-error-muted": "rgba(251, 44, 85, 0.18)",
    "--color-on-error": GROUND,

    // --- ground and surfaces ---
    "--color-background-body": GROUND,
    "--color-background-surface": PANEL,
    "--color-background-card": PANEL,
    "--color-background-popover": PANEL_LIFT,
    "--color-background-muted": GROUND_LIFT,
    "--color-background-inverted": TEXT,

    "--color-overlay": "rgba(10, 12, 14, 0.72)",
    "--color-overlay-hover": "rgba(255, 255, 255, 0.06)",
    "--color-overlay-pressed": "rgba(255, 255, 255, 0.12)",
    "--color-tint-hover": "rgba(255, 255, 255, 0.06)",

    // --- text and icons ---
    "--color-text-primary": TEXT,
    "--color-text-secondary": TEXT_DIM,
    "--color-text-disabled": TEXT_FAINT,
    "--color-icon-primary": TEXT,
    "--color-icon-secondary": TEXT_DIM,
    "--color-icon-disabled": TEXT_FAINT,

    // --- structure ---
    "--color-border": HAIRLINE,
    "--color-border-emphasized": HAIRLINE_STRONG,
    "--color-track": "rgba(255, 255, 255, 0.08)",
    "--color-skeleton": "rgba(255, 255, 255, 0.06)",
    "--color-shadow": "rgba(0, 0, 0, 0.55)",
    "--color-neutral": TEXT_DIM,

    // --- type: the fonts index.html already preloads ---
    "--font-family-body": '"Public Sans", -apple-system, "Segoe UI", Arial, sans-serif',
    "--font-family-heading": '"Public Sans", -apple-system, "Segoe UI", Arial, sans-serif',
    "--font-family-code": '"Roboto Mono", ui-monospace, "SF Mono", Menlo, monospace',

    // --- geometry: the floating-panel radius, one step softer than stock ---
    "--radius-container": "14px",
    "--radius-element": "8px",
    "--radius-inner": "6px",
  },
});
