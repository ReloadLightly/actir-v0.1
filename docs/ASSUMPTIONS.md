# ACTIR Assumptions & Switches (v0.1)

This document defines model assumptions as explicit configuration switches.
These are not "truth claims"—they are controlled experimental conditions
for ablations.

## Core modeling stance
- Environment: partially observable stochastic game (POSG) / Dec-POMDP-style.
- Regional system: Indo-Pacific cast with 1 extra-regional great power (XGP),
  1 regional great power (RGP), and multiple middle powers (MPs).
- Key task: infer spiral vs deterrence dynamics under uncertainty and choose
  signaling/arms/alliance/trade/calm actions to preserve stability.

## Configuration switches (initial)
### Information & perception
- `obs_noise` (float): observation noise scale (higher = more misperception).
- `intel_quality_by_role`: role-based modifier for observation quality.

### Offense–Defense & projection
- `offense_defense_balance` in {`defense_dominant`, `balanced`, `offense_dominant`}
- `maritime_friction_multiplier` (float): increases friction across "water."

### Commitment & credibility
- `commitment_reliability` (0..1): how reliably commitments/alliances hold.
- `signal_credibility_dynamics` (on/off): whether credibility updates from consistency.

### System structure / polarity
- `capability_preset` in {`bipolar`, `unipolar`, `multipolar`, `custom`}
- `power_shift_rate` (float): speed of long-run capability changes (econ/tech).

## Metrics (must be logged)
- Crisis level / escalation index
- War rate (terminal)
- Arms race slope (Δ arms/posture)
- Security dilemma intensity (SDI) metric (explicit)
- Middle-power autonomy / option value proxy (explicit)

## Development rule
We only label a mechanic as "faithful to X" after:
1) a traceability row exists (with pages),
2) the mechanic is implemented,
3) a stylized-fact test passes.
