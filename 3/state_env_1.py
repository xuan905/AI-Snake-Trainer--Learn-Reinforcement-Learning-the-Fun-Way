"""
state_env_1.py
功能：定義基本狀態元素（蛇頭位置、食物位置）
劇情比喻：阿偉給蛇一雙眼睛，知道自己和食物在哪裡
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
        pygame.display.set_caption("Snake State Environment")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.snake_pos = [100, 100]
        self.snake_body = [list(self.snake_pos)]
        self.direction = 'RIGHT'
        self.food_pos = [random.randrange(0, self.width, self.snake_size),
                         random.randrange(0, self.height, self.snake_size)]
        return self.get_state()

    def get_state(self):
        # 狀態僅包含蛇頭位置與食物位置
        return {'snake_head': self.snake_pos, 'food_pos': self.food_pos}

    def step(self, action):
        # 移動蛇
        if action == 'UP':
            self.snake_pos[1] -= self.snake_size
        elif action == 'DOWN':
            self.snake_pos[1] += self.snake_size
        elif action == 'LEFT':
            self.snake_pos[0] -= self.snake_size
        elif action == 'RIGHT':
            self.snake_pos[0] += self.snake_size

        done = False
        reward = 0

        # 撞牆死亡
        if (self.snake_pos[0] < 0 or self.snake_pos[0] >= self.width or
            self.snake_pos[1] < 0 or self.snake_pos[1] >= self.height):
            reward = -1
            done = True

        # 吃食物
        if self.snake_pos == self.food_pos:
            reward = 1
            self.food_pos = [random.randrange(0, self.width, self.snake_size),
                             random.randrange(0, self.height, self.snake_size)]
            self.snake_body.append(list(self.snake_pos))
        else:
            self.snake_body.append(list(self.snake_pos))
            self.snake_body.pop(0)

        return self.get_state(), reward, done

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
