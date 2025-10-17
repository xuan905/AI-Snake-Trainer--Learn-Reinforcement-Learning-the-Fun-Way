"""
episode_env_4.py
功能：統計回合分數與學習進度
劇情比喻：阿偉記錄蛇每場遊戲的成績，評估成長
"""

import numpy as np
import random

STATE_SIZE = 128
ACTIONS = ['LEFT','RIGHT','STRAIGHT']
Q_table = np.zeros((STATE_SIZE, len(ACTIONS)))
alpha = 0.1
gamma = 0.9
epsilon = 0.2

def choose_action(state):
    if random.uniform(0,1) < epsilon:
        return random.choice(range(len(ACTIONS)))
    else:
        return np.argmax(Q_table[state])

def step(state, action):
    next_state = (state + action) % STATE_SIZE
    reward = 1 if next_state % 10 == 0 else -0.01
    done = next_state % 50 == 0
    return next_state, reward, done

def update_q(state, action, reward, next_state):
    Q_table[state, action] += alpha * (reward + gamma * np.max(Q_table[next_state]) - Q_table[state, action])

if __name__=="__main__":
    state = 0
    total_reward = 0
    done = False
    while not done:
        action = choose_action(state)
        next_state, reward, done = step(state, action)
        update_q(state, action, reward, next_state)
        total_reward += reward
        state = next_state
    print(f"Episode finished. Total Reward: {total_reward}")
