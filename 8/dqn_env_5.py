"""
dqn_env_5.py
功能：封裝完整 DQN 訓練函式，支援多回合自動訓練
劇情比喻：阿偉建立「自我進化系統」，蛇能從多回合累積智慧
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from collections import deque
import copy

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

def train_dqn(episodes=10, batch_size=16, gamma=0.9, epsilon=0.2):
    policy_net = QNetwork()
    target_net = copy.deepcopy(policy_net)
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    buffer = deque(maxlen=1000)
    
    rewards_list = []
    for ep in range(episodes):
        state = np.random.rand(STATE_SIZE)
        done = False
        total_reward = 0
        while not done:
            if random.random() < epsilon:
                action = random.randint(0, ACTION_SIZE-1)
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    action = torch.argmax(policy_net(state_tensor)).item()
            next_state, reward, done = step(state, action)
            buffer.append((state, action, reward, next_state, done))
            total_reward += reward
            
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
        rewards_list.append(total_reward)
        print(f"Episode {ep+1}: Total Reward = {total_reward}")
    
    return rewards_list

if __name__=="__main__":
    rewards = train_dqn(episodes=5)
    print("Rewards per episode:", rewards)
