"""
snake_env_1.py
功能：建立基本 pygame 視窗與遊戲迴圈
劇情比喻：阿偉在空白畫布上誕生世界的第一道光
"""

import pygame
import sys

# 初始化 pygame
pygame.init()

# 遊戲視窗尺寸
WIDTH, HEIGHT = 400, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake Game - Step 1")

# 顏色定義
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# 遊戲主迴圈
running = True
while running:
    screen.fill(BLACK)  # 背景填黑
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.flip()

pygame.quit()
sys.exit()
