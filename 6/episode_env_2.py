"""
episode_env_2.py
功能：回合中 ε-greedy 選動作
劇情比喻：蛇學會在回合中「冒險或保守」，平衡探索與利用
"""

import numpy as np
import random

STATE_SIZE = 128
ACTIONS = ['LEFT','RIGHT','STRAIGHT']
Q_table = np.zeros((STATE_SIZE, len(ACTIONS)))

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

if __name__=="__main__":
    state = 0
    done = False
    while not done:
        action = choose_action(state, epsilon=0.2)
        next_state, reward, done = step(state, action)
        print(f"State:{state}, Action:{ACTIONS[action]}, Reward:{reward}")
        state = next_state
    print("Single Episode with ε-greedy Finished")
