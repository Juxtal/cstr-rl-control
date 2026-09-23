import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt


class CSTREnv(gym.Env):
    """
    Simple nonlinear CSTR environment.

    Reaction:
        A -> B

    State:
        [C_A, C_B, T]

    Action:
        normalized jacket temperature command in [-1, 1]

    Control objective:
        keep C_B close to target_CB
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        # ==========================================
        # 1. Simulation settings
        # ==========================================

        self.dt = 0.1
        self.max_steps = 300

        # ==========================================
        # 2. CSTR parameters
        # ==========================================

        self.CA_in = 1.0       # mol/L
        self.CB_in = 0.0       # mol/L

        self.T_in = 330.0      # K

        self.tau = 5.0         # residence time

        # Reaction kinetics
        self.k_ref = 0.25
        self.T_ref = 330.0
        self.alpha = 0.025

        # Heat generation coefficient
        self.beta = 25.0

        # Heat transfer coefficient
        self.gamma = 0.20

        # Jacket temperature range
        self.Tc_min = 280.0
        self.Tc_max = 360.0

        # Desired product concentration
        self.target_CB = 0.60

        # ==========================================
        # 3. Observation space
        # ==========================================

        # state = [CA, CB, T]

        self.observation_space = spaces.Box(
            low=np.array(
                [0.0, 0.0, 270.0],
                dtype=np.float32
            ),
            high=np.array(
                [2.0, 2.0, 450.0],
                dtype=np.float32
            ),
            dtype=np.float32
        )

        # ==========================================
        # 4. Action space
        # ==========================================

        # RL outputs a normalized number:
        #
        # -1 -> Tc_min
        # +1 -> Tc_max

        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.state = None
        self.step_count = 0

    # ============================================================
    # Reaction rate constant
    # ============================================================

    def reaction_rate_constant(self, T):
        """
        Temperature-dependent reaction rate.

        This is a simplified Arrhenius-like expression.
        """

        k = self.k_ref * np.exp(
            self.alpha * (T - self.T_ref)
        )

        return k

    # ============================================================
    # CSTR model
    # ============================================================

    def cstr_ode(self, state, Tc):

        CA, CB, T = state

        k = self.reaction_rate_constant(T)

        # Reaction rate
        r = k * CA

        # ------------------------------------------
        # Mass balance A
        # ------------------------------------------

        dCA_dt = (
            (self.CA_in - CA) / self.tau
            - r
        )

        # ------------------------------------------
        # Mass balance B
        # ------------------------------------------

        dCB_dt = (
            (self.CB_in - CB) / self.tau
            + r
        )

        # ------------------------------------------
        # Energy balance
        # ------------------------------------------

        dT_dt = (
            (self.T_in - T) / self.tau
            + self.beta * r
            + self.gamma * (Tc - T)
        )

        return np.array(
            [dCA_dt, dCB_dt, dT_dt],
            dtype=np.float64
        )

    # ============================================================
    # RK4 numerical integration
    # ============================================================

    def rk4_step(self, state, Tc):

        dt = self.dt

        k1 = self.cstr_ode(state, Tc)

        k2 = self.cstr_ode(
            state + 0.5 * dt * k1,
            Tc
        )

        k3 = self.cstr_ode(
            state + 0.5 * dt * k2,
            Tc
        )

        k4 = self.cstr_ode(
            state + dt * k3,
            Tc
        )

        new_state = state + (
            dt / 6.0
        ) * (
            k1
            + 2.0 * k2
            + 2.0 * k3
            + k4
        )

        return new_state

    # ============================================================
    # Convert normalized action to real jacket temperature
    # ============================================================

    def action_to_temperature(self, action):

        action = float(
            np.clip(action[0], -1.0, 1.0)
        )

        Tc = (
            self.Tc_min
            + (action + 1.0)
            / 2.0
            * (self.Tc_max - self.Tc_min)
        )

        return Tc

    # ============================================================
    # Reset
    # ============================================================

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        # Initial reactor state
        self.state = np.array(
            [
                1.0,     # CA
                0.0,     # CB
                330.0    # T
            ],
            dtype=np.float64
        )

        self.step_count = 0

        observation = self.state.astype(
            np.float32
        )

        info = {}

        return observation, info

    # ============================================================
    # One environment step
    # ============================================================

    def step(self, action):

        # ------------------------------------------
        # 1. Convert RL action to Tc
        # ------------------------------------------

        Tc = self.action_to_temperature(action)

        # ------------------------------------------
        # 2. Integrate CSTR forward
        # ------------------------------------------

        next_state = self.rk4_step(
            self.state,
            Tc
        )

        self.state = next_state

        self.step_count += 1

        CA, CB, T = self.state

        # ------------------------------------------
        # 3. Reward
        # ------------------------------------------

        tracking_error = (
            CB - self.target_CB
        )

        # Main objective:
        # keep CB close to 0.6
        reward_tracking = -(
            tracking_error ** 2
        )

        # Small penalty for extreme jacket temperature
        reward_control = -1e-5 * (
            Tc - 320.0
        ) ** 2

        reward = (
            reward_tracking
            + reward_control
        )

        # ------------------------------------------
        # 4. Safety termination
        # ------------------------------------------

        unsafe_temperature = (
            T < 275.0
            or T > 430.0
        )

        invalid_state = (
            CA < 0
            or CB < 0
            or not np.all(
                np.isfinite(self.state)
            )
        )

        terminated = (
            unsafe_temperature
            or invalid_state
        )

        # Extra penalty for unsafe states
        if terminated:
            reward -= 10.0

        # ------------------------------------------
        # 5. Time limit
        # ------------------------------------------

        truncated = (
            self.step_count
            >= self.max_steps
        )

        observation = self.state.astype(
            np.float32
        )

        info = {
            "Tc": Tc,
            "tracking_error": tracking_error
        }

        return (
            observation,
            float(reward),
            terminated,
            truncated,
            info
        )


# ================================================================
# Random-agent test
# ================================================================

def run_random_agent():

    env = CSTREnv()

    state, info = env.reset(seed=42)

    # Save history
    times = []
    CA_history = []
    CB_history = []
    T_history = []
    Tc_history = []
    reward_history = []

    total_reward = 0.0

    for step in range(env.max_steps):

        # ------------------------------------------
        # Random action
        # ------------------------------------------

        action = env.action_space.sample()

        # ------------------------------------------
        # Environment step
        # ------------------------------------------

        (
            next_state,
            reward,
            terminated,
            truncated,
            info
        ) = env.step(action)

        CA, CB, T = next_state

        # ------------------------------------------
        # Save data
        # ------------------------------------------

        time = step * env.dt

        times.append(time)

        CA_history.append(CA)
        CB_history.append(CB)
        T_history.append(T)

        Tc_history.append(
            info["Tc"]
        )

        reward_history.append(
            reward
        )

        total_reward += reward

        # ------------------------------------------
        # Print occasionally
        # ------------------------------------------

        if step % 50 == 0:

            print(
                f"step={step:3d} | "
                f"CA={CA:.3f} | "
                f"CB={CB:.3f} | "
                f"T={T:.2f} K | "
                f"Tc={info['Tc']:.2f} K | "
                f"reward={reward:.4f}"
            )

        # ------------------------------------------
        # End episode
        # ------------------------------------------

        if terminated or truncated:
            break

        state = next_state

    print()
    print(
        f"Total reward: {total_reward:.3f}"
    )

    # ============================================================
    # Plot 1: concentrations
    # ============================================================

    plt.figure()

    plt.plot(
        times,
        CA_history,
        label="C_A"
    )

    plt.plot(
        times,
        CB_history,
        label="C_B"
    )

    plt.axhline(
        env.target_CB,
        linestyle="--",
        label="Target C_B"
    )

    plt.xlabel("Time")
    plt.ylabel("Concentration")
    plt.title(
        "CSTR Concentration Response"
    )

    plt.legend()
    plt.grid()

    plt.show()

    # ============================================================
    # Plot 2: reactor temperature
    # ============================================================

    plt.figure()

    plt.plot(
        times,
        T_history
    )

    plt.xlabel("Time")
    plt.ylabel("Reactor temperature [K]")
    plt.title(
        "Reactor Temperature"
    )

    plt.grid()

    plt.show()

    # ============================================================
    # Plot 3: control input
    # ============================================================

    plt.figure()

    plt.plot(
        times,
        Tc_history
    )

    plt.xlabel("Time")
    plt.ylabel(
        "Jacket temperature Tc [K]"
    )

    plt.title(
        "Random Control Input"
    )

    plt.grid()

    plt.show()

    # ============================================================
    # Plot 4: reward
    # ============================================================

    plt.figure()

    plt.plot(
        times,
        reward_history
    )

    plt.xlabel("Time")
    plt.ylabel("Reward")

    plt.title(
        "Reward During Episode"
    )

    plt.grid()

    plt.show()
def train_sac():

    # 1. Create the CSTR environment
    env = CSTREnv()

    # 2. Create the SAC agent
    from stable_baselines3 import SAC

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1
    )

    # 3. Start training
    model.learn(
        total_timesteps=50_000
    )

    # 4. Save the trained model
    model.save("sac_cstr")

    print("Training finished.")
def test_sac():

    from stable_baselines3 import SAC

    env = CSTREnv()

    # Load the model trained above
    model = SAC.load(
        "sac_cstr",
        env=env
    )

    state, info = env.reset(seed=42)

    total_reward = 0.0

    for step in range(env.max_steps):

        # SAC picks an action from the current state
        action, _ = model.predict(
            state,
            deterministic=True
        )

        state, reward, terminated, truncated, info = env.step(
            action
        )

        CA, CB, T = state

        total_reward += reward

        if step % 20 == 0:
            print(
                f"step={step:3d} | "
                f"CA={CA:.3f} | "
                f"CB={CB:.3f} | "
                f"T={T:.2f} | "
                f"Tc={info['Tc']:.2f} | "
                f"reward={reward:.4f}"
            )

        if terminated or truncated:
            break

    print("Total reward:", total_reward)
def test_sac():

    from stable_baselines3 import SAC
    import matplotlib.pyplot as plt

    # Create the environment
    env = CSTREnv()

    # Load the trained SAC model
    model = SAC.load("sac_cstr", env=env)

    # Reset the CSTR
    state, info = env.reset(seed=42)

    # Buffers for logging data
    times = []
    CA_history = []
    CB_history = []
    T_history = []
    Tc_history = []
    reward_history = []

    total_reward = 0.0

    for step in range(env.max_steps):

        # ============================
        # SAC chooses an action from the state
        # ============================

        action, _ = model.predict(
            state,
            deterministic=True
        )

        # Apply the action to the CSTR
        state, reward, terminated, truncated, info = env.step(action)

        CA, CB, T = state

        # Save data
        times.append(step * env.dt)
        CA_history.append(CA)
        CB_history.append(CB)
        T_history.append(T)
        Tc_history.append(info["Tc"])
        reward_history.append(reward)

        total_reward += reward

        if terminated or truncated:
            break

    print(f"Total reward: {total_reward:.3f}")

    # ============================
    # CB
    # ============================

    plt.figure()

    plt.plot(times, CB_history, label="C_B")

    plt.axhline(
        env.target_CB,
        linestyle="--",
        label="Target C_B"
    )

    plt.xlabel("Time")
    plt.ylabel("C_B")
    plt.title("SAC Control of C_B")
    plt.legend()
    plt.grid()

    plt.show()

    # ============================
    # Jacket temperature
    # ============================

    plt.figure()

    plt.plot(times, Tc_history)

    plt.xlabel("Time")
    plt.ylabel("Tc [K]")
    plt.title("SAC Control Action")
    plt.grid()

    plt.show()


if __name__ == "__main__":

    test_sac()