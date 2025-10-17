"""
snake_env_3.py
功能：加入食物產生與隨機位置
劇情比喻：世界出現「資源」，蛇有了生存的動力
"""

import pygame
import sys
import random

pygame.init()
WIDTH, HEIGHT = 400, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake Game - Step 3")

BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

snake_pos = [100, 100]
snake_size = 20
speed = 20
direction = 'RIGHT'

# 食物初始位置
food_pos = [random.randrange(0, WIDTH, snake_size), random.randrange(0, HEIGHT, snake_size)]

clock = pygame.time.Clock()
running = True

while running:
    screen.fill(BLACK)
    
    # 事件控制
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                direction = 'UP'
            elif event.key == pygame.K_DOWN:
                direction = 'DOWN'
            elif event.key == pygame.K_LEFT:
                direction = 'LEFT'
            elif event.key == pygame.K_RIGHT:
                direction = 'RIGHT'

    # 移動蛇
    if direction == 'UP':
        snake_pos[1] -= speed
    elif direction == 'DOWN':
        snake_pos[1] += speed
    elif direction == 'LEFT':
        snake_pos[0] -= speed
    elif direction == 'RIGHT':
        snake_pos[0] += speed

    # 畫蛇和食物
    pygame.draw.rect(screen, GREEN, (*snake_pos, snake_size, snake_size))
    pygame.draw.rect(screen, RED, (*food_pos, snake_size, snake_size))
    
    pygame.display.flip()
    clock.tick(10)

pygame.quit()
sys.exit()
