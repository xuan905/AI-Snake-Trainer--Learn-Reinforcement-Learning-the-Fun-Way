"""
state_env_4.py
功能：將位置與危險整合成完整狀態向量
劇情比喻：蛇開始綜合分析，決定最安全且最有效路徑
"""
def get_state(self):
    # 食物方向
    snake_x, snake_y = self.snake_pos
    food_x, food_y = self.food_pos
    food_up = int(food_y < snake_y)
    food_down = int(food_y > snake_y)
    food_left = int(food_x < snake_x)
    food_right = int(food_x > snake_x)

    # 危險
    size = self.snake_size
    dir = self.direction
    def danger(pos):
        x, y = pos
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return 1
        if [x, y] in self.snake_body[:-1]:
            return 1
        return 0

    if dir == 'UP':
        danger_front = danger([snake_x, snake_y - size])
        danger_left  = danger([snake_x - size, snake_y])
        danger_right = danger([snake_x + size, snake_y])
    elif dir == 'DOWN':
        danger_front = danger([snake_x, snake_y + size])
        danger_left  = danger([snake_x + size, snake_y])
        danger_right = danger([snake_x - size, snake_y])
    elif dir == 'LEFT':
        danger_front = danger([snake_x - size, snake_y])
        danger_left  = danger([snake_x, snake_y + size])
        danger_right = danger([snake_x, snake_y - size])
    else:  # RIGHT
        danger_front = danger([snake_x + size, snake_y])
        danger_left  = danger([snake_x, snake_y - size])
        danger_right = danger([snake_x, snake_y + size])

    # 整合狀態向量
    state_vector = [food_up, food_down, food_left, food_right,
                    danger_front, danger_left, danger_right]
    return state_vector
