"""
eval_env_2.py
功能：畫出收斂曲線 (Total Reward vs Episode)
劇情比喻：阿偉像畫成績曲線，觀察蛇的學習進度
"""

import matplotlib.pyplot as plt
import random

STATE_SIZE = 128
ACTIONS = ['LEFT','RIGHT','STRAIGHT']

def step(state, action):
    next_state = (state + action) % STATE_SIZE
    reward = 1 if next_state % 10 == 0 else -0.01
    done = next_state % 50 == 0
    return next_state, reward, done

episodes = 50
scores = []

for ep in range(episodes):
    state = 0
    done = False
    total_reward = 0
    while not done:
        action = random.choice(range(len(ACTIONS)))
        state, reward, done = step(state, action)
        total_reward += reward
    scores.append(total_reward)

plt.plot(range(1, episodes+1), scores)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Learning Convergence Curve")
plt.show()
