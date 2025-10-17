"""
eval_env_5.py
功能：封裝完整評估函式，支援多組超參數測試
劇情比喻：阿偉建立「智慧檢測表」，一次比較多種設定
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

def train(alpha, gamma, epsilon=0.2, episodes=50):
    Q_table = np.zeros((STATE_SIZE, len(ACTIONS)))
    rewards = []
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

if __name__=="__main__":
    param_sets = [
        {'alpha':0.1,'gamma':0.9},
        {'alpha':0.2,'gamma':0.9},
        {'alpha':0.1,'gamma':0.99},
    ]
    for params in param_sets:
        scores = train(params['alpha'], params['gamma'])
        plt.plot(range(1,len(scores)+1), scores, label=f"α={params['alpha']}, γ={params['gamma']}")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Hyperparameter Evaluation")
    plt.legend()
    plt.show()
