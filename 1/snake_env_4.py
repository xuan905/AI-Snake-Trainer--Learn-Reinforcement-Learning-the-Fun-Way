"""
snake_env_4.py
功能：偵測蛇吃到食物、撞牆死亡
劇情比喻：阿偉創造「規則」，讓蛇懂得「後果」
"""

import pygame
import sys
import random

pygame.init()
WIDTH, HEIGHT = 400, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake Game - Step 4")

BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

snake_pos = [100, 100]
snake_body = [list(snake_pos)]
snake_size = 20
speed = 20
direction = 'RIGHT'
score = 0

food_pos = [random.randrange(0, WIDTH, snake_size), random.randrange(0, HEIGHT, snake_size)]

clock = pygame.time.Clock()
running = True

while running:
    screen.fill(BLACK)
    
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

    if direction == 'UP':
        snake_pos[1] -= speed
    elif direction == 'DOWN':
        snake_pos[1] += speed
    elif direction == 'LEFT':
        snake_pos[0] -= speed
    elif direction == 'RIGHT':
        snake_pos[0] += speed

    # 蛇吃到食物
    if snake_pos == food_pos:
        score += 1
        food_pos = [random.randrange(0, WIDTH, snake_size), random.randrange(0, HEIGHT, snake_size)]
        snake_body.append(list(snake_pos))
    else:
        snake_body.append(list(snake_pos))
        snake_body.pop(0)

    # 撞牆死亡
    if snake_pos[0] < 0 or snake_pos[0] >= WIDTH or snake_pos[1] < 0 or snake_pos[1] >= HEIGHT:
        print("Game Over! Score:", score)
        running = False

    # 畫蛇和食物
    for pos in snake_body:
        pygame.draw.rect(screen, GREEN, (*pos, snake_size, snake_size))
    pygame.draw.rect(screen, RED, (*food_pos, snake_size, snake_size))

    pygame.display.flip()
    clock.tick(10)

pygame.quit()
sys.exit()
