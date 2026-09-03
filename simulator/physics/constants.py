"""Universal constants and fuel properties. All DERIVED (published, not fitted)."""

# --- Universal gas / atmosphere constants (Handbook 7.2) ---
R_AIR = 287.05          # J/(kg K), specific gas constant for dry air
G = 9.80665              # m/s^2, standard gravity
T0_ISA = 288.15          # K, ISA sea-level temperature
P0_ISA = 101325.0        # Pa, ISA sea-level pressure
L_ISA = 0.0065           # K/m, ISA tropospheric lapse rate
ISA_EXPONENT = G / (L_ISA * R_AIR)   # = 5.2559, matches Handbook 7.2 worked example

# --- Fuel properties (petrol / avgas-equivalent, Handbook 7.5) ---
Q_LHV = 43.5e6            # J/kg, lower heating value of petrol
AFR_STOICH = 14.7         # kg air / kg fuel, stoichiometric
FA_STOICH = 1.0 / AFR_STOICH  # kg fuel / kg air, stoichiometric

# --- Gas properties for the cycle model ---
GAMMA_COLD = 1.40         # ratio of specific heats, cold air (compression, Handbook 7.2 example)
GAMMA_HOT = 1.32           # ratio of specific heats, burned gas (Handbook 7.6/7.10: "1.30 to 1.35")
CV_AIR = R_AIR / (GAMMA_COLD - 1.0)   # J/(kg K), specific heat at constant volume, cold air
