"""
eval_env_3.py
功能：調整學習率 α，觀察策略收斂快慢
劇情比喻：蛇學習速度加快或減慢，像加油或休息
"""

import numpy as np
import matplotlib.pyplot as plt
import random

STATE_SIZE = 128
ACTIONS = ['LEFT','RIGHT','STRAIGHT']

def step(state, action):
    next_state = (state + action) % STATE_SIZE
    reward = 1 if next_state % 10 == 0 else -0.01
    done = next_state % 50 == 0
    return next_state, reward, done

def train(alpha):
    Q_table = np.zeros((STATE_SIZE, len(ACTIONS)))
    episodes = 50
    rewards = []
    gamma = 0.9
    epsilon = 0.2
    for ep in range(episodes):
        state = 0
        done = False
        total_reward = 0
        while not done:
            if random.random() < epsilon:
                action = random.choice(range(len(ACTIONS)))
            else:
                action = np.argmax(Q_table[state])
            next_state, reward, done = step(state, action)
            Q_table[state, action] += alpha * (reward + gamma*np.max(Q_table[next_state]) - Q_table[state, action])
            total_reward += reward
            state = next_state
        rewards.append(total_reward)
    return rewards

alphas = [0.05, 0.1, 0.2]
for a in alphas:
    scores = train(a)
    plt.plot(range(1, len(scores)+1), scores, label=f'alpha={a}')
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Effect of Learning Rate α on Convergence")
plt.legend()
plt.show()
