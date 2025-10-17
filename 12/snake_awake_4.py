"""
snake_awake_4.py
功能：分析策略收斂與穩定性
劇情比喻：蛇回顧歷史決策，找出最穩定路徑
"""

import matplotlib.pyplot as plt
import torch
from train_dqn_snake import DQN, SnakeGame
import random

state_dim = 11
action_dim = 3
model = DQN(state_dim, action_dim)
model.load_state_dict(torch.load("dqn_snake.pth"))
model.eval()

scores = []
episodes = 50

for ep in range(episodes):
    env = SnakeGame()
    state = env.reset()
    while not env.done:
        if random.random() < 0.05:
            action = random.randint(0,2)
        else:
            with torch.no_grad():
                action = torch.argmax(model(state)).item()
        state, reward, done = env.step(action)
    scores.append(env.score)

plt.plot(range(episodes), scores, label="Episode Score")
plt.xlabel("Episode")
plt.ylabel("Score")
plt.title("策略收斂與穩定性分析")
plt.legend()
plt.show()
