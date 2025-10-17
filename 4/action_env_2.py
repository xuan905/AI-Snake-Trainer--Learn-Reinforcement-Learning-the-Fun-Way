"""
action_env_2.py
功能：將方向映射到動作空間
劇情比喻：蛇學會「理解轉向指令」，把策略翻譯成實際方向
"""

def map_action(current_dir, action):
    dirs = ['UP','RIGHT','DOWN','LEFT']
    idx = dirs.index(current_dir)
    if action == 'STRAIGHT':
        new_dir = dirs[idx]
    elif action == 'RIGHT':
        new_dir = dirs[(idx + 1) % 4]
    elif action == 'LEFT':
        new_dir = dirs[(idx - 1) % 4]
    return new_dir

if __name__ == "__main__":
    print(map_action('UP','LEFT'))  # LEFT
    print(map_action('UP','RIGHT')) # RIGHT
    print(map_action('UP','STRAIGHT')) # UP
