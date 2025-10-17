"""
eval_env_4.py
功能：調整折扣 γ，觀察遠期回報影響
劇情比喻：阿偉讓蛇更看重「未來收益」或「當前回報」
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

def train(gamma):
    Q_table = np.zeros((STATE_SIZE, len(ACTIONS)))
    episodes = 50
    rewards = []
    alpha = 0.1
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

gammas = [0.7, 0.9, 0.99]
for g in gammas:
    scores = train(g)
    plt.plot(range(1, len(scores)+1), scores, label=f'gamma={g}')
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Effect of Discount Factor γ on Convergence")
plt.legend()
plt.show()
