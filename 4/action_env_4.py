"""
action_env_4.py
功能：封裝 Policy Function，返回動作
劇情比喻：阿偉設計「策略大腦」，蛇能依狀態向量選擇動作
"""

import random

def simple_policy(state_vector):
    """
    簡單策略範例：
    - 若食物在前方，直行
    - 否則隨機選擇左右
    state_vector = [food_up, food_down, food_left, food_right, danger_front, danger_left, danger_right]
    """
    food_up, food_down, food_left, food_right, danger_front, danger_left, danger_right = state_vector
    if food_up and not danger_front:
        return 'STRAIGHT'
    else:
        return random.choice(['LEFT','RIGHT'])

if __name__ == "__main__":
    state = [1,0,0,0,0,0,0]
    for _ in range(5):
        action = simple_policy(state)
        print("Policy chooses action:", action)
