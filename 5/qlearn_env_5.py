"""
qlearn_env_5.py
功能：封裝成可訓練環境，AI 自動更新 Q-table
劇情比喻：蛇正式進入「自我學習模式」，能根據狀態決定最優行動
"""

import numpy as np
import random

ACTIONS = ['LEFT','RIGHT','STRAIGHT']
STATE_SIZE = 128

class QLearningSnakeEnv:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.Q_table = np.zeros((STATE_SIZE, len(ACTIONS)))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.state = 0

    def choose_action(self, state):
        if random.uniform(0,1) < self.epsilon:
            return random.choice(range(len(ACTIONS)))
        else:
            return np.argmax(self.Q_table[state])

    def step(self, state, action):
        next_state = (state + action) % STATE_SIZE
        reward = 1 if next_state % 10 == 0 else -0.01
        done = next_state % 50 == 0
        return next_state, reward, done

    def update_q(self, state, action, reward, next_state):
        self.Q_table[state, action] += self.alpha * (
            reward + self.gamma * np.max(self.Q_table[next_state]) - self.Q_table[state, action]
        )

if __name__=="__main__":
    env = QLearningSnakeEnv()
    for episode in range(5):
        state = 0
        done = False
        while not done:
            action = env.choose_action(state)
            next_state, reward, done = env.step(state, action)
            env.update_q(state, action, reward, next_state)
            print(f"Episode:{episode}, State:{state}, Action:{ACTIONS[action]}, Reward:{reward}")
            state = next_state
