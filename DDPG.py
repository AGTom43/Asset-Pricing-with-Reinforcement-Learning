import pandas as pd
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gym import spaces
import gym

# Load your data
data = pd.read_csv("XOM_30_minute_6_month_data.csv")

# Display the loaded data
print("Loaded Data:")
print(data.head())


class StockTradingEnv(gym.Env):
    def __init__(self, df):
        super(StockTradingEnv, self).__init__()

        self.df = df
        self.current_step = 0
        self.max_steps = len(df) - 1

        # Action space: Discrete, representing Buy, Sell, and Hold
        # Additional continuous parameter for the amount
        self.action_space = spaces.Tuple(
            (
                spaces.Discrete(3),  # Buy, Sell, Hold
                spaces.Box(low=0, high=1, shape=(1,)),  # Amount (continuous)
            )
        )

        # Observation space: [last_price, volume, sma]
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,))

    def reset(self):
        self.current_step = 0
        obs = self._next_observation()
        print("Resetting Environment. Initial Observation:")
        print(obs)
        return obs

    def _next_observation(self):
        obs = self.df.loc[
            self.current_step, ["Last Price", "Volume", "SMAVG (15)"]
        ].values
        obs_normalized = obs / obs.max()
        print("Next Observation (Normalized):")
        print(obs_normalized)
        return obs_normalized

    def step(self, action):
        self.current_step += 1

        if self.current_step > self.max_steps:
            self.current_step = 0

        discrete_action, continuous_action = action

        obs = self._next_observation()
        reward = 0

        # Ensure continuous_action is a valid continuous value between 0 and 1
        continuous_action = np.clip(continuous_action, 0, 1)

        if discrete_action == 0:  # Buy
            reward = -self.df.loc[self.current_step, "Last Price"] * continuous_action
        elif discrete_action == 1:  # Sell
            reward = self.df.loc[self.current_step, "Last Price"] * continuous_action

        done = self.current_step == self.max_steps - 1

        print("Step:", self.current_step)
        print("Discrete Action:", discrete_action)
        print("Continuous Action (Amount):", continuous_action)
        print("Reward:", reward)
        print("Done:", done)

        return obs, reward, done, {}


# Create the environment
env = DummyVecEnv([lambda: StockTradingEnv(data)])
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

# Display the normalized observation space
print("Normalized Observation Space:")
print(env.observation_space)

# Define the DDPG model - Uses simple feedforward neural network
model = DDPG("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=10000)
print("Training Completed.")

# Save the model
model.save("ddpg_stock_trading")
print("Model Saved.")

# Evaluate the model
obs = env.reset()
for _ in range(len(data)):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()

env.close()
