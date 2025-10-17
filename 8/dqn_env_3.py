"""
dqn_env_3.py
功能：Target Network 與定期更新
劇情比喻：阿偉給蛇一個「穩定參考大腦」，避免策略跳動
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

STATE_SIZE = 10
ACTION_SIZE = 3

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
    policy_net = QNetwork()
    target_net = copy.deepcopy(policy_net)
    
    # 假設每隔 5 步更新 target network
    for step in range(1, 21):
        if step % 5 == 0:
            target_net.load_state_dict(policy_net.state_dict())
            print(f"Step {step}: Target network updated.")
