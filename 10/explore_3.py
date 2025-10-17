"""
explore_3.py
功能：結合 ε-decay 與 Reward Shaping
劇情比喻：蛇在探索中獲得更合理的回饋
"""

# 核心概念：
# - ε 衰減同 explore_1.py
# - Reward Shaping: 修改 step() 回傳 reward
#   e.g., reward += 0.5 if靠近食物 else reward -=0.01
# 訓練流程類似 train_dqn()
# 可視化 total_rewards
