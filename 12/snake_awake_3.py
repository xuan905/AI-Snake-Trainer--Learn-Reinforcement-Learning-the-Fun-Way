"""
snake_awake_3.py
功能：增加視覺介面顯示策略決策過程
劇情比喻：阿偉打造「策略眼鏡」，能看到蛇內心決策
"""

import pygame
import torch
import random
from train_dqn_snake import DQN, SnakeGame

# 初始化 pygame
pygame.init()
cell_size = 20
cols, rows = 10, 10
screen = pygame.display.set_mode((cols*cell_size, rows*cell_size))
pygame.display.set_caption("Snake AI: 策略眼鏡")

state_dim = 11
action_dim = 3
model = DQN(state_dim, action_dim)
model.load_state_dict(torch.load("dqn_snake.pth"))
model.eval()

env = SnakeGame(width=cols*cell_size, height=rows*cell_size, block_size=cell_size)
state = env.reset()

running = True
while running and not env.done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 選擇動作
    epsilon = 0.05
    if random.random() < epsilon:
        action = random.randint(0, 2)
    else:
        with torch.no_grad():
            q_values = model(state)
            action = torch.argmax(q_values).item()
    
    state, reward, done = env.step(action)

    # 畫出蛇和食物
    screen.fill((0,0,0))
    for seg in env.snake:
        pygame.draw.rect(screen, (0,255,0), (*seg, cell_size, cell_size))
    pygame.draw.rect(screen, (255,0,0), (*env.food, cell_size, cell_size))
    
    pygame.display.flip()
    pygame.time.delay(100)
    print(f"Action: {['Left','Straight','Right'][action]} | Score: {env.score}")

pygame.quit()
print(f"遊戲結束，最終得分：{env.score}")
