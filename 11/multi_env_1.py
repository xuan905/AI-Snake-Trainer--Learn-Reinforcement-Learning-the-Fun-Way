"""
multi_env_1.py
功能：隨機化地圖尺寸
劇情比喻：蛇開始探索不同大小的迷宮，學會適應空間變化
"""

import numpy as np
import random

class SnakeEnv:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.reset()
    
    def reset(self):
        self.snake = [(self.width//2, self.height//2)]
        self.food = (random.randint(0,self.width-1), random.randint(0,self.height-1))
        self.done = False
        return self.get_state()
    
    def get_state(self):
        return (self.snake[0], self.food)
    
    def step(self, action):
        head_x, head_y = self.snake[0]
        if action == 0: head_y -= 1
        elif action == 1: head_x += 1
        elif action == 2: head_y += 1
        elif action == 3: head_x -= 1
        new_head = (head_x, head_y)
        reward = -0.01
        self.done = False
        if new_head == self.food:
            reward = 1
            self.snake.insert(0, new_head)
            self.food = (random.randint(0,self.width-1), random.randint(0,self.height-1))
        else:
            self.snake.insert(0,new_head)
            self.snake.pop()
        if not (0<=head_x<self.width and 0<=head_y<self.height) or new_head in self.snake[1:]:
            reward = -1
            self.done = True
        return self.get_state(), reward, self.done

# 測試不同地圖尺寸
if __name__=="__main__":
    for w,h in [(5,5),(10,10),(15,15)]:
        env = SnakeEnv(w,h)
        state = env.reset()
        print(f"Map size: {w}x{h}, Initial state: {state}")
