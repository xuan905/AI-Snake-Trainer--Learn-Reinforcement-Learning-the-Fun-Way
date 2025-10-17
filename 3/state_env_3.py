"""
state_env_3.py
功能：偵測前/左/右危險（牆或自己）
劇情比喻：阿偉教蛇「辨識危險」，知道哪裡可能撞牆或撞自己
"""
# step 與 move 同前，get_state() 更新：
def get_state(self):
    # 計算蛇頭位置
    snake_x, snake_y = self.snake_pos
    size = self.snake_size

    # 前/左/右危險判斷（假設方向 self.direction）
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

    return [danger_front, danger_left, danger_right]
