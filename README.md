# CSTR RL Control

A self-study project exploring reinforcement-learning-based feedback control of a simplified nonlinear CSTR (continuous stirred-tank reactor).

A `gymnasium` environment models the reactor with first-principles mass and energy balances, and a Soft Actor-Critic (SAC) agent from `stable-baselines3` learns to manipulate the jacket temperature so that the product concentration tracks a setpoint.

## Problem formulation

| | |
|---|---|
| Reaction | A -> B (first order) |
| State | `[C_A, C_B, T]` (concentrations in mol/L, reactor temperature in K) |
| Action | Normalized jacket temperature command in [-1, 1], mapped linearly to `T_c` in [280, 360] K |
| Objective | Track `C_B` to the setpoint `C_B* = 0.6` mol/L |
| Episode | 300 steps, `dt = 0.1`, RK4 integration |

## Mathematical model

With residence time `tau`, feed `C_A,in = 1`, `C_B,in = 0`, `T_in = 330 K`:

```
dC_A/dt = (C_A,in - C_A) / tau - k(T) C_A
dC_B/dt = (C_B,in - C_B) / tau + k(T) C_A
dT/dt   = (T_in - T) / tau + beta k(T) C_A + gamma (T_c - T)

k(T) = k_ref exp( alpha (T - T_ref) )
```

Parameters: `tau = 5`, `k_ref = 0.25`, `T_ref = 330`, `alpha = 0.025`, `beta = 25`, `gamma = 0.2`.

## Reward and constraints

```
r = -(C_B - C_B*)^2 - 1e-5 (T_c - 320)^2
```

The episode terminates with an extra penalty of 10 if the reactor temperature leaves [275, 430] K or the state becomes invalid (negative or non-finite concentrations).

## Files

- `cstr.py` - environment, random-agent baseline, SAC training and evaluation
- `sac_cstr.zip` - trained SAC model

## Usage

```bash
pip install gymnasium stable-baselines3 numpy matplotlib

# Train (50,000 timesteps)
python -c "from cstr import train_sac; train_sac()"

# Evaluate the trained model and plot C_B tracking and the control action
python cstr.py
```

## Limitations

The reactor is a simplified benchmark: a first-order reaction with an exponential temperature dependence rather than full Arrhenius kinetics, no disturbances or measurement noise, and no comparison yet against classical controllers such as PID or MPC.
