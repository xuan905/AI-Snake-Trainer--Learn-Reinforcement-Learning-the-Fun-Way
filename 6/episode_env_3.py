"""
episode_env_3.py
功能：更新 Q-table 每一步
劇情比喻：蛇把每一步的結果記入「行動手冊」，慢慢優化
"""

import numpy as np
import random

STATE_SIZE = 128
ACTIONS = ['LEFT','RIGHT','STRAIGHT']
Q_table = np.zeros((STATE_SIZE, len(ACTIONS)))
alpha = 0.1
gamma = 0.9

def choose_action(state, epsilon=0.2):
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
    done = False
    while not done:
        action = choose_action(state)
        next_state, reward, done = step(state, action)
        update_q(state, action, reward, next_state)
        print(f"State:{state}, Action:{ACTIONS[action]}, Reward:{reward}")
        state = next_state
    print("Single Episode with Q-update Finished")
