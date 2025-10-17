"""
qlearn_env_4.py
功能：將 Q-learning 與環境互動整合
劇情比喻：蛇邊移動邊學習，經驗逐步累積成最佳行動表
"""

import numpy as np
import random

ACTIONS = ['LEFT','RIGHT','STRAIGHT']
STATE_SIZE = 128
Q_table = np.zeros((STATE_SIZE, len(ACTIONS)))

def choose_action(state, epsilon=0.1):
    if random.uniform(0,1) < epsilon:
        return random.choice(range(len(ACTIONS)))
    else:
        return np.argmax(Q_table[state])

def update_q(Q_table, state, action, reward, next_state, alpha=0.1, gamma=0.9):
    Q_table[state, action] += alpha * (reward + gamma * np.max(Q_table[next_state]) - Q_table[state, action])
    return Q_table

# 假想簡單環境
def step(state, action):
    next_state = (state + action) % STATE_SIZE
    reward = 1 if next_state % 10 == 0 else -0.01
    done = next_state % 50 == 0
    return next_state, reward, done

if __name__=="__main__":
    state = 0
    for _ in range(20):
        action = choose_action(state, epsilon=0.2)
        next_state, reward, done = step(state, action)
        Q_table = update_q(Q_table, state, action, reward, next_state)
        print(f"State:{state}, Action:{ACTIONS[action]}, Reward:{reward}")
        state = next_state
        if done:
            state = 0
