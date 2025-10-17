"""
episode_env_5.py
功能：封裝完整訓練迴圈，支援多回合自動訓練
劇情比喻：蛇正式進入「多回合自我學習模式」，能從錯誤中累積智慧
"""

import numpy as np
import random

STATE_SIZE = 128
ACTIONS = ['LEFT','RIGHT','STRAIGHT']

class TrainingEnv:
    def __init__(self, episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.Q_table = np.zeros((STATE_SIZE, len(ACTIONS)))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes = episodes

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
        self.Q_table[state, action] += self.alpha * (reward + self.gamma * np.max(self.Q_table[next_state]) - self.Q_table[state, action])

    def train(self):
        for ep in range(self.episodes):
            state = 0
            done = False
            total_reward = 0
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.step(state, action)
                self.update_q(state, action, reward, next_state)
                total_reward += reward
                state = next_state
            print(f"Episode {ep+1}/{self.episodes}, Total Reward: {total_reward}")

if __name__=="__main__":
    env = TrainingEnv(episodes=10)
    env.train()
