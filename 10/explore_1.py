"""
explore_1.py
功能：動態 ε 衰減策略
劇情比喻：蛇學會「先冒險探索，再穩健行動」
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
import copy
import matplotlib.pyplot as plt

STATE_SIZE = 10
ACTION_SIZE = 3

class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(STATE_SIZE, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, ACTION_SIZE)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def step(state, action):
    next_state = np.random.rand(STATE_SIZE)
    reward = random.choice([1, -0.01])
    done = random.random() < 0.1
    return next_state, reward, done

def train_dqn(episodes=50, batch_size=16, gamma=0.9, epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.95):
    policy_net = QNetwork()
    target_net = copy.deepcopy(policy_net)
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    buffer = deque(maxlen=1000)
    total_rewards = []
    
    epsilon = epsilon_start
    for ep in range(episodes):
        state = np.random.rand(STATE_SIZE)
        done = False
        ep_reward = 0
        while not done:
            if random.random() < epsilon:
                action = random.randint(0, ACTION_SIZE-1)
            else:
                with torch.no_grad():
                    action = torch.argmax(policy_net(torch.FloatTensor(state).unsqueeze(0))).item()
            next_state, reward, done = step(state, action)
            buffer.append((state, action, reward, next_state, done))
            ep_reward += reward

            if len(buffer) >= batch_size:
                batch = random.sample(buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions).unsqueeze(1)
                rewards = torch.FloatTensor(rewards).unsqueeze(1)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones).unsqueeze(1)

                q_values = policy_net(states).gather(1, actions)
                next_q = target_net(next_states).max(1)[0].unsqueeze(1)
                target = rewards + gamma * next_q * (1 - dones)

                loss = F.mse_loss(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            state = next_state

        target_net.load_state_dict(policy_net.state_dict())
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        total_rewards.append(ep_reward)
        print(f"Episode {ep+1}: Reward={ep_reward:.2f}, Epsilon={epsilon:.3f}")
    
    plt.plot(total_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("動態 ε 衰減策略 — 總分數曲線")
    plt.show()

if __name__=="__main__":
    train_dqn()
