"""
explore_5.py
功能：封裝完整強化探索訓練流程
劇情比喻：阿偉打造「智慧試探室」，蛇自動探索並優化策略
"""

# 將 explore_1~4.py 功能封裝成函式:
# train(episodes, batch_size, gamma, epsilon_start, epsilon_min, epsilon_decay)
# 使用 Priority Replay 與 Reward Shaping
# 回傳 total_rewards 與 epsilon_history
# 最後自動畫圖
