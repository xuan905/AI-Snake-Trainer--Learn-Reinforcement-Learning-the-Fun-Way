"""
state_env_2.py
功能：食物方向編碼（上下左右）
劇情比喻：蛇能「看出食物方向」，像有指南針指引
"""
# 在 state_env_1.py 基礎上修改 get_state()
def get_state(self):
    snake_x, snake_y = self.snake_pos
    food_x, food_y = self.food_pos

    food_up = int(food_y < snake_y)
    food_down = int(food_y > snake_y)
    food_left = int(food_x < snake_x)
    food_right = int(food_x > snake_x)

    return [food_up, food_down, food_left, food_right]
