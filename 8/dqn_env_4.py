"""
dqn_env_4.py
功能：DQN 訓練迴圈，結合 ε-greedy
劇情比喻：蛇在回合中平衡「冒險與利用」，用神經網路選行動
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

# 簡單模擬環境 step
def step(state, action):
    next_state = np.random.rand(STATE_SIZE)
    reward = random.choice([1, -0.01])
    done = random.random() < 0.1
    return next_state, reward, done

if __name__=="__main__":
    policy_net = QNetwork()
    target_net = copy.deepcopy(policy_net)
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    buffer = deque(maxlen=1000)
    
    epsilon = 0.2
    gamma = 0.9
    batch_size = 16
    episodes = 5
    
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
                    q_values = policy_net(state_tensor)
                    action = torch.argmax(q_values).item()
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
        
        # 每回合更新 target_net
        target_net.load_state_dict(policy_net.state_dict())
        print(f"Episode {ep+1}: Total Reward = {total_reward}")
