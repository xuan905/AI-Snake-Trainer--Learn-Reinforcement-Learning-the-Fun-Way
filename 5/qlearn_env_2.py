"""
qlearn_env_2.py
功能：ε-greedy 探索策略選動作
劇情比喻：蛇學會「偶爾冒險」，既利用已知經驗，也探索未知領域
"""

import numpy as np
import random

ACTIONS = ['LEFT','RIGHT','STRAIGHT']
STATE_SIZE = 128
Q_table = np.zeros((STATE_SIZE, len(ACTIONS)))

def choose_action(state, epsilon=0.1):
    if random.uniform(0,1) < epsilon:
        return random.choice(range(len(ACTIONS)))  # 探索
    else:
        return np.argmax(Q_table[state])           # 利用

if __name__=="__main__":
    state = 10
    for _ in range(10):
        a = choose_action(state)
        print("Chosen action:", ACTIONS[a])
