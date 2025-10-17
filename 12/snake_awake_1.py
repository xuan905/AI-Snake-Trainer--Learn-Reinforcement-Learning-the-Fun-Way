"""
snake_awake_1.py
功能：儲存與載入訓練模型
劇情比喻：阿偉給蛇「記憶庫」，蛇能記住過去學到的策略
"""

import torch
from train_dqn_snake import DQN

state_dim = 11
action_dim = 3

# 初始化模型
model = DQN(state_dim, action_dim)

# 載入訓練好的模型權重
try:
    model.load_state_dict(torch.load("dqn_snake.pth"))
    model.eval()
    print("模型成功載入 dqn_snake.pth")
except FileNotFoundError:
    print("找不到模型檔案，請先訓練 DQN 並存檔。")

# 儲存模型範例
torch.save(model.state_dict(), "dqn_snake_saved.pth")
print("模型已儲存為 dqn_snake_saved.pth")
