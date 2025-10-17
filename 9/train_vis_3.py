"""
train_vis_3.py
功能：監控梯度更新與權重變化
劇情比喻：阿偉觀察蛇腦中神經連線的強弱，判斷學習效率
"""

# 訓練過程中加入梯度監控：
# for name, param in policy_net.named_parameters():
#     if param.grad is not None:
#         print(f"{name} grad mean: {param.grad.mean().item():.4f}")
