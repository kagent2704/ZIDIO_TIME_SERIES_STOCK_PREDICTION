import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

# -------------------------------
# Config
# -------------------------------
INITIAL_BALANCE = 10000
EPISODES = 50
BATCH_SIZE = 32
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
LR = 0.001

# -------------------------------
# DQN Network
# -------------------------------
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# -------------------------------
# Trading Environment
# -------------------------------
class TradingEnv:
    def __init__(self, df, initial_balance=INITIAL_BALANCE):
        self.df = df.reset_index(drop=True)
        self.n_days = len(df)
        self.initial_balance = initial_balance
        self.reset()

    def reset(self):
        self.day = 0
        self.balance = self.initial_balance
        self.shares = 0
        self.portfolio_value = self.balance
        return self._get_state()

    def _get_state(self):
        day_idx = min(self.day, self.n_days - 1)  # clamp to last row
        price = self.df.loc[day_idx, "Close"]
        return np.array([self.balance / 10000, self.shares / 10, price / 500])

    def step(self, action):
        done = False
        price = self.df.loc[self.day, "Close"]
        prev_value = self.portfolio_value

        # Action: 0=Hold, 1=Buy, 2=Sell
        if action == 1 and self.balance >= price:
            self.shares += 1
            self.balance -= price
        elif action == 2 and self.shares > 0:
            self.shares -= 1
            self.balance += price

        self.day += 1
        if self.day >= self.n_days:
            done = True

        self.portfolio_value = self.balance + self.shares * price
        reward = self.portfolio_value - prev_value
        return self._get_state(), reward, done

# -------------------------------
# Replay Memory
# -------------------------------
class ReplayMemory:
    def __init__(self, capacity=2000):
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# -------------------------------
# Training Loop
# -------------------------------
def train_rl_agent(input_csv="data/sp500.csv", output_path=None):
    """
    Train a DQN-based RL trading agent on historical stock prices.
    input_csv : path to CSV containing 'Close' prices
    output_path : path to save portfolio plot (png)
    """
    if output_path is None:
        output_path = os.path.join("outputs", "rl_trading_simulation.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.read_csv(input_csv)
    df = df.rename(columns={"S&P500": "Close"})
    env = TradingEnv(df)
    state_dim = 3
    action_dim = 3

    policy_net = DQN(state_dim, action_dim)
    target_net = DQN(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    criterion = nn.MSELoss()

    memory = ReplayMemory()
    epsilon = EPSILON_START
    portfolio_history = []

    for ep in range(EPISODES):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            if random.random() < epsilon:
                action = random.randint(0, action_dim-1)
            else:
                with torch.no_grad():
                    q_values = policy_net(torch.FloatTensor(state))
                    action = torch.argmax(q_values).item()

            next_state, reward, done = env.step(action)
            memory.push((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            # Experience Replay
            if len(memory) >= BATCH_SIZE:
                batch = memory.sample(BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions).unsqueeze(1)
                rewards = torch.FloatTensor(rewards).unsqueeze(1)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones).unsqueeze(1)

                q_values = policy_net(states).gather(1, actions)
                with torch.no_grad():
                    next_q = target_net(next_states).max(1)[0].unsqueeze(1)
                    target = rewards + (1 - dones) * GAMMA * next_q

                loss = criterion(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        portfolio_history.append(env.portfolio_value)
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        target_net.load_state_dict(policy_net.state_dict())
        print(f"Episode {ep+1}/{EPISODES} - Final Portfolio Value: {env.portfolio_value:.2f}")

    # -------------------------------
    # Save Portfolio Plot
    # -------------------------------
    plt.figure(figsize=(10,6))
    plt.plot(portfolio_history, color='blue')
    plt.xlabel("Episode")
    plt.ylabel("Portfolio Value")
    plt.title("RL Agent Trading Simulation")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"RL trading plot saved at: {output_path}")

    # Save Model
    model_path = os.path.join(os.path.dirname(output_path), "rl_agent.pth")
    torch.save(policy_net.state_dict(), model_path)
    print(f"RL agent model saved at: {model_path}")

if __name__ == "__main__":
    train_rl_agent()
