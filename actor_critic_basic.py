import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import gymnasium
from gymnasium import spaces

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class DDPGTradingEnv(gymnasium.Env):

  def __init__(self, df, window_size, render_mode=None):
    super(DDPGTradingEnv, self).__init__()

    self.df = df
    self.window_size = window_size
    self.render_mode = render_mode

    self.prices, self.signal_features = self._process_data()

    self.shape = (window_size, self.signal_features.shape[1])

    # discrete_actions = spaces.Box(low=-1,
    #                               high=1,
    #                               shape=(1, ),
    #                               dtype=np.float32)
    # continuous_quantity = spaces.Box(low=-1,
    #                                  high=1,
    #                                  shape=(1, ),
    #                                  dtype=np.float32)
    # continuous_amount = spaces.Box(low=0,
    #                                high=1,
    #                                shape=(1, ),
    #                                dtype=np.float32)

    # Include the amount as part of the action space
    #         self.action_space = spaces.Tuple((discrete_actions, continuous_amount))
    self.action_space = spaces.Box(low=-1,
                                   high=1,
                                   shape=(1, ),
                                   dtype=np.float32)

    self.observation_space = spaces.Box(low=0,
                                        high=1,
                                        shape=self.shape,
                                        dtype=np.float32)

    self.profit_history = []
    self.reset()

  def reset(self, seed=None):
    super().reset(seed=1)
    self.current_step = self.window_size
    self.total_profit = 0
    self.history = []
    self.balance = 10000
    self.shares_held = 0
    self.portfolio_value = self.balance
    self.previous_portfolio_value = self.portfolio_value
    self.initial_balance = self.balance
    return self._get_observation()

  # Stores prices and features (for now just the price)
  def _process_data(self):

    # Extract prices and features
    prices = df['Last Price'].values
    #features = df[['Last Price', 'Volume', 'SMAVG (15)']].values
    features = df[['Last Price']].values

    return prices, features

  def _get_observation(self):
    start = self.current_step - self.window_size
    end = self.current_step
    observation = self.signal_features[start:end]
    return observation

  def _take_action(self, action_value):
    current_price = self.prices[self.current_step]
                   
    if buy_sell_decision > 0:
      self._buy_stock(current_price, action_value)
    elif buy_sell_decision < 0:
      self._sell_stock(current_price, action_value)

  def _update_portfolio(self, action_value):
    current_price = self.prices[self.current_step]
    if action_value > 0:
      self._buy_stock(current_price, action_value)
    elif action_value < 0:
      self._sell_stock(current_price, action_value)
    # Update the portfolio value
    self.portfolio_value = self.balance + self.shares_held * current_price

  def _buy_stock(self, action_value):
    # Determine the actual amount of stock to buy based on 'amount'
    # For example, you might interpret 'amount' as a percentage of your balance
    #         buy_amount = min(self.balance / current_price, amount)
    buy_amount = round(self.balance / current_price) * action_value
    self.balance -= buy_amount * current_price
    self.shares_held += buy_amount

  def _sell_stock(self, action_value):
    # Determine the actual amount of stock to sell based on 'amount'
    # Ensure that you don't sell more than you hold
    #         sell_amount = min(self.shares_held, amount)
    sell_amount = self.shares_held * action_value
    self.balance += sell_amount * current_price
    self.shares_held -= sell_amount

  def step(self, action, seed=None, options=None):
    super().reset(seed=seed)
    action_value = action
    self._take_action(action)
    self.current_step += 1
    reward = self._calculate_reward(action_value)
    done = self.current_step >= len(self.prices) - 1
    observation = self._get_observation()
    info = {
        'current_step': self.current_step,
        'total_profit': self.total_profit
    }
    self.profit_history.append(self.total_profit)
    print(
        f"Step: {self.current_step}, Portfolio value: {self.portfolio_value}")
    return observation, reward, done, info

  def render(self, mode='human'):
    # Simple text rendering
    if mode == 'human':
      print(f"Step: {self.current_step}, Total Profit: {self.total_profit}")

  def _calculate_reward(self, action):
    """
        Calculate the reward based on the action taken.
        Action can either be buying or selling a stock.
        The reward is the change in portfolio value as a result of the action.
        """
    # Assuming self.portfolio_value stores the current value of the portfolio
    previous_portfolio_value = self.portfolio_value

    # Update portfolio value based on the action
    self._update_portfolio(action)

    # New portfolio value
    current_portfolio_value = self.portfolio_value

    # Reward is the change in portfolio value
    reward = current_portfolio_value - previous_portfolio_value

    self._update_portfolio(
        action)  # Ensure this method updates the portfolio value
    current_portfolio_value = self.portfolio_value
    reward = current_portfolio_value - self.previous_portfolio_value
    self.previous_portfolio_value = current_portfolio_value
    self._update_profit(action)
    return reward

  def _update_profit(self, action):
    """
        Update the total profit based on the action taken.
        """
    current_price = self.prices[self.current_step]

    # Update the portfolio after the action
    self._update_portfolio(action)

    # Calculate total profit as the difference between current portfolio value and initial balance
    self.total_profit = self.portfolio_value - self.initial_balance
    # Print for debugging
    print(f"Updated total profit: {self.total_profit}")


# Load data
#C:\Users\rohit\OneDrive\Documents\Asset-Pricing-with-Reinforcement-Learning\XOM_30_minute_6_month_data.csv
df = pd.read_csv('XOM_30_minute_6_month_data.csv', parse_dates=['Date'])
df.sort_values('Date', inplace=True)

# Normalize
scaler = MinMaxScaler()
df[['Last Price', 'Volume', 'SMAVG (15)'
    ]] = scaler.fit_transform(df[['Last Price', 'Volume', 'SMAVG (15)']])

# Split into training and testing sets
train_size = int(len(df) * 0.8)
train_df = df[:train_size]
test_df = df[train_size:]

window_size = 60
env = DDPGTradingEnv(df, window_size=window_size, render_mode=None)


class Actor(nn.Module):

  def __init__(self, input_dim, output_dim, hidden_size=128):
    super(Actor, self).__init__()

    input_dim = input_dim[0]
    output_dim = output_dim[0]

    self.fc0 = nn.Linear(1, input_dim)
    self.fc1 = nn.Linear(input_dim, hidden_size)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(hidden_size, output_dim)
    self.tanh = nn.Tanh()

  def forward(self, state):
    x = self.relu(self.fc0(state))
    x = self.relu(self.fc1(x))
    x = self.tanh(self.fc2(x))
    return x


class Critic(nn.Module):

  def __init__(self, input_dim, output_dim, hidden_size=128):
    super(Critic, self).__init__()

    input_dim = input_dim[
        0]  # Assuming output_dim represents the action dimension
    action_dim = 2

    self.fc0 = nn.Linear(1, input_dim)
    self.fc1 = nn.Linear(input_dim + action_dim,
                         hidden_size)  # Combine state and action dimensions
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(hidden_size, output_dim)

  def forward(self, state, action):
    x_state = self.relu(self.fc0(state))
    x = torch.cat((x_state, action), dim=1)  # Concatenate state and action
    x = self.relu(self.fc1(x))
    x = self.fc2(x)
    return x


# Example usage in your DDPGTradingEnv
state_dim = env.observation_space.shape  # Adjust based on your environment's state shape
action_dim = env.action_space.shape  # Adjust based on your environment's action shape

actor_model = Actor(input_dim=state_dim, output_dim=action_dim)
critic_model = Critic(input_dim=state_dim, output_dim=1)

# Example forward pass
state = torch.FloatTensor(
    env._get_observation())  # Assuming _get_observation() returns the state
action = actor_model.forward(state)
value_estimate = critic_model.forward(state, action)

print(value_estimate)
