"""
snake_awake_5.py
功能：封裝完整自主訓練與展示流程
劇情比喻：阿偉建立「智慧實驗室」，蛇能自動訓練、學習、展示
"""

import torch
from train_dqn_snake import DQN, SnakeGame
import random
import matplotlib.pyplot as plt

state_dim = 11
action_dim = 3

# 載入或初始化模型
model = DQN(state_dim, action_dim)
try:
    model.load_state_dict(torch.load("dqn_snake.pth"))
    print("載入訓練模型")
except FileNotFoundError:
    print("找不到模型，將使用隨機初始化模型")

model.eval()

# 自動遊玩並紀錄分數
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
    print(f"Episode {ep+1} Score: {env.score}")

# 畫出分數曲線
plt.plot(range(episodes), scores)
plt.xlabel("Episode")
plt.ylabel("Score")
plt.title("自主訓練分數曲線")
plt.show()
