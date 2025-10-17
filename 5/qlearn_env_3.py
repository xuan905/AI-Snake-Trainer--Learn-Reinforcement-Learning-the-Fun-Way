"""
qlearn_env_3.py
功能：更新 Q-table
劇情比喻：蛇把每次經驗記入「行動手冊」，慢慢變聰明
"""

import numpy as np

ACTIONS = ['LEFT','RIGHT','STRAIGHT']
STATE_SIZE = 128
Q_table = np.zeros((STATE_SIZE, len(ACTIONS)))

def update_q(Q_table, state, action, reward, next_state, alpha=0.1, gamma=0.9):
    Q_table[state, action] = Q_table[state, action] + alpha * (
        reward + gamma * np.max(Q_table[next_state]) - Q_table[state, action]
    )
    return Q_table

if __name__=="__main__":
    state = 0
    action = 1
    reward = 1
    next_state = 2
    Q_table = update_q(Q_table, state, action, reward, next_state)
    print(Q_table[state])
