"""
state_env_5.py
功能：完整 Snake 環境 + 狀態向量 + Reward Function，可互動
劇情比喻：阿偉完成「感知系統」，蛇能看到完整世界並做決策
"""

import pygame
import random

class StateSnakeEnv:
    def __init__(self, width=400, height=400, snake_size=20):
        pygame.init()
        self.width = width
        self.height = height
        self.snake_size = snake_size
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Snake AI State Environment")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.snake_pos = [100, 100]
        self.snake_body = [list(self.snake_pos)]
        self.direction = 'RIGHT'
        self.food_pos = [random.randrange(0, self.width, self.snake_size),
                         random.randrange(0, self.height, self.snake_size)]
        self.score = 0
        return self.get_state()

    def step(self, action):
        # 移動蛇
        if action == 'UP':
            self.direction = 'UP'
            self.snake_pos[1] -= self.snake_size
        elif action == 'DOWN':
            self.direction = 'DOWN'
            self.snake_pos[1] += self.snake_size
        elif action == 'LEFT':
            self.direction = 'LEFT'
            self.snake_pos[0] -= self.snake_size
        elif action == 'RIGHT':
            self.direction = 'RIGHT'
            self.snake_pos[0] += self.snake_size

        reward = -0.01  # 每步成本
        done = False

        # 撞牆
        if (self.snake_pos[0] < 0 or self.snake_pos[0] >= self.width or
            self.snake_pos[1] < 0 or self.snake_pos[1] >= self.height):
            reward = -1
            done = True

        # 撞自己
        if self.snake_pos in self.snake_body[:-1]:
            reward = -1
            done = True

        # 吃食物
        if self.snake_pos == self.food_pos:
            reward = 1
            self.score += 1
            self.food_pos = [random.randrange(0, self.width, self.snake_size),
                             random.randrange(0, self.height, self.snake_size)]
            self.snake_body.append(list(self.snake_pos))
        else:
            self.snake_body.append(list(self.snake_pos))
            self.snake_body.pop(0)

        return self.get_state(), reward, done

    def get_state(self):
        """
        回傳完整狀態向量：
        [食物方向(上,下,左,右), 前方危險, 左方危險, 右方危險]
        """
        snake_x, snake_y = self.snake_pos
        food_x, food_y = self.food_pos
        size = self.snake_size
        dir = self.direction

        # 食物方向
        food_up = int(food_y < snake_y)
        food_down = int(food_y > snake_y)
        food_left = int(food_x < snake_x)
        food_right = int(food_x > snake_x)

        # 危險判斷
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

        state_vector = [food_up, food_down, food_left, food_right,
                        danger_front, danger_left, danger_right]
        return state_vector

    def render(self):
        BLACK = (0, 0, 0)
        GREEN = (0, 255, 0)
        RED = (255, 0, 0)
        self.screen.fill(BLACK)
        for pos in self.snake_body:
            pygame.draw.rect(self.screen, GREEN, (*pos, self.snake_size, self.snake_size))
        pygame.draw.rect(self.screen, RED, (*self.food_pos, self.snake_size, self.snake_size))
        pygame.display.flip()
        self.clock.tick(10)

# 測試用
if __name__ == "__main__":
    env = StateSnakeEnv()
    running = True
    state = env.reset()
    import random
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action = random.choice(['UP','DOWN','LEFT','RIGHT'])
        state, reward, done = env.step(action)
        print("State:", state, "Reward:", reward)
        env.render()
        if done:
            print("Game Over! Score:", env.score)
            state = env.reset()
    pygame.quit()
