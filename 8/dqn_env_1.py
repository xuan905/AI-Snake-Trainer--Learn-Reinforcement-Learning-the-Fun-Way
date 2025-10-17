"""
dqn_env_1.py
功能：建立簡單神經網路 Q 模型
劇情比喻：阿偉給蛇一個「智慧大腦」，能記住過去經驗並預測回報
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

STATE_SIZE = 10  # 假設環境狀態向量長度
ACTION_SIZE = 3  # LEFT, RIGHT, STRAIGHT

class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(STATE_SIZE, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, ACTION_SIZE)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

if __name__=="__main__":
    model = QNetwork()
    sample_state = torch.rand((1, STATE_SIZE))
    q_values = model(sample_state)
    print("Sample Q-values:", q_values)
