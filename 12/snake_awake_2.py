"""
snake_awake_2.py
功能：自動玩 Snake 並展示結果
劇情比喻：蛇開始自主巡邏迷宮，展示「生存智慧」
"""

import torch
import random
from train_dqn_snake import DQN, SnakeGame

state_dim = 11
action_dim = 3

# 初始化模型
model = DQN(state_dim, action_dim)
model.load_state_dict(torch.load("dqn_snake.pth"))
model.eval()

env = SnakeGame()
state = env.reset()
total_score = 0

while not env.done:
    # ε-greedy 探索
    epsilon = 0.05
    if random.random() < epsilon:
        action = random.randint(0, 2)
    else:
        with torch.no_grad():
            q_values = model(state)
            action = torch.argmax(q_values).item()
    state, reward, done = env.step(action)
    total_score += reward
    print(f"Action: {['Left','Straight','Right'][action]} | Score: {env.score}")

print(f"遊戲結束，最終得分：{env.score}")
