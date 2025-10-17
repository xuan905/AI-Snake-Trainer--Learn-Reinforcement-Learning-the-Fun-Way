"""
qlearn_env_1.py
功能：建立初始 Q-table
劇情比喻：阿偉給蛇一本「行動經驗手冊」，所有狀態與動作都先寫上 0 分
"""

import numpy as np

ACTIONS = ['LEFT','RIGHT','STRAIGHT']

# 假設狀態空間大小 (例如簡化 2^7 = 128)
STATE_SIZE = 128

# 初始化 Q-table
Q_table = np.zeros((STATE_SIZE, len(ACTIONS)))

if __name__=="__main__":
    print("初始 Q-table shape:", Q_table.shape)
    print(Q_table)
