# CSTR RL Control

A messy little project from my self-study of reinforcement learning. The code is rough, so feedback is welcome.

## What is this?

I wrote a simplified CSTR (Continuous Stirred-Tank Reactor) simulation as a `gymnasium` environment:

- Reaction: A -> B
- State: `[C_A, C_B, T]` (concentration of A, concentration of B, reactor temperature)
- Action: normalized jacket temperature command in [-1, 1]
- Goal: keep the concentration of B close to a target value (0.6 mol/L)

Then I trained a SAC agent from `stable-baselines3` to learn how to adjust the jacket temperature to hold that concentration.

## Files

- `cstr.py` - environment definition plus training/testing scripts
- `sac_cstr.zip` - trained SAC model weights

## Usage

```bash
pip install gymnasium stable-baselines3 numpy matplotlib

# Train
python -c "from cstr import train_sac; train_sac()"

# Test the trained model and plot the results
python cstr.py
```

## Notes

This is just a practice project for learning RL. The environment and reward function are simplified toy versions, with no rigorous process-control modeling, so don't treat it as a reference for real industrial systems.
