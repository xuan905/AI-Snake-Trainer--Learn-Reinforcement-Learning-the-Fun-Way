"""
snake_play_dqn.py
功能：載入 DQN 訓練模型，自動玩 Snake
劇情比喻：蛇使用過去學到的策略，自主巡邏迷宮，展示「生存智慧」
"""

import pygame
import random
import torch
import torch.nn as nn
import numpy as np
import time

# ----------------------------
# Snake 環境（簡化）
# ----------------------------
class SnakeGame:
    def __init__(self, width=200, height=200, block_size=20, render=True):
        self.width = width
        self.height = height
        self.block_size = block_size
        self.render = render
        if self.render:
            pygame.init()
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("DQN Snake Play")
            self.clock = pygame.time.Clock()
        self.reset()
    
    def reset(self):
        self.snake = [[self.width//2, self.height//2]]
        self.direction = [self.block_size, 0]
        self.food = [random.randrange(0, self.width, self.block_size),
                     random.randrange(0, self.height, self.block_size)]
        self.done = False
        self.score = 0
        return self.get_state()
    
    def step(self, action):  # action: 0-left,1-straight,2-right
        dx, dy = self.direction
        if action == 0:  # 左轉
            dx, dy = -dy, dx
        elif action == 2:  # 右轉
            dx, dy = dy, -dx
        self.direction = [dx, dy]

        head = [self.snake[0][0]+dx, self.snake[0][1]+dy]
        self.snake.insert(0, head)
        reward = 0

        if head == self.food:
            self.score += 1
            reward = 1
            self.food = [random.randrange(0, self.width, self.block_size),
                         random.randrange(0, self.height, self.block_size)]
        else:
            self.snake.pop()

        if head[0] < 0 or head[0] >= self.width or head[1] < 0 or head[1] >= self.height or head in self.snake[1:]:
            self.done = True
            reward = -1

        if self.render:
            self.render_game()
        
        return self.get_state(), reward, self.done
    
    def danger(self, dx, dy):
        head = [self.snake[0][0]+dx, self.snake[0][1]+dy]
        if head[0] < 0 or head[0] >= self.width or head[1] < 0 or head[1] >= self.height or head in self.snake[1:]:
            return 1.0
        return 0.0
    
    def get_state(self):
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        dir_l = self.direction == [-self.block_size, 0]
        dir_r = self.direction == [self.block_size, 0]
        dir_u = self.direction == [0, -self.block_size]
        dir_d = self.direction == [0, self.block_size]

        # 危險偵測
        dx, dy = self.direction
        danger_straight = self.danger(dx, dy)
        danger_left = self.danger(-dy, dx)
        danger_right = self.danger(dy, -dx)

        # 食物方向
        food_up = food_y < head_y
        food_down = food_y > head_y
        food_left = food_x < head_x
        food_right = food_x > head_x

        state = torch.tensor([
            danger_straight, danger_left, danger_right,
            dir_l, dir_r, dir_u, dir_d,
            food_up, food_down, food_left, food_right
        ], dtype=torch.float32)
        return state

    def render_game(self):
        self.screen.fill((0,0,0))
        # 畫蛇
        for block in self.snake:
            pygame.draw.rect(self.screen, (0,255,0), pygame.Rect(block[0], block[1], self.block_size, self.block_size))
        # 畫食物
        pygame.draw.rect(self.screen, (255,0,0), pygame.Rect(self.food[0], self.food[1], self.block_size, self.block_size))
        pygame.display.flip()
        self.clock.tick(10)  # 控制遊戲速度

# ----------------------------
# DQN 網路
# ----------------------------
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
    def forward(self, x):
        return self.fc(x)

# ----------------------------
# 主程式
# ----------------------------
def play(model_path="dqn_snake.pth"):
    env = SnakeGame(render=True)
    state_dim = 11
    action_dim = 3
    model = DQN(state_dim, action_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    state = env.reset()
    total_score = 0
    done = False

    while not done:
        with torch.no_grad():
            q_values = model(state)
            action = torch.argmax(q_values).item()
        state, reward, done = env.step(action)
        total_score += reward
        print(f"Action: {['Left','Straight','Right'][action]} | Score: {env.score}")

    print(f"遊戲結束，最終得分：{env.score}")
    if env.render:
        pygame.quit()

if __name__ == "__main__":
    play()
