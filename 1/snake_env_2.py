"""
snake_env_2.py
功能：加入蛇的繪製與移動控制
劇情比喻：阿偉賦予蛇「移動的意識」
"""

import pygame
import sys

pygame.init()
WIDTH, HEIGHT = 400, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake Game - Step 2")

BLACK = (0, 0, 0)
GREEN = (0, 255, 0)

# 蛇初始位置與速度
snake_pos = [100, 100]
snake_size = 20
speed = 20
direction = 'RIGHT'

clock = pygame.time.Clock()
running = True

while running:
    screen.fill(BLACK)
    
    # 處理事件
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

    pygame.draw.rect(screen, GREEN, (*snake_pos, snake_size, snake_size))
    pygame.display.flip()
    clock.tick(10)

pygame.quit()
sys.exit()
