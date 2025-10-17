"""
action_env_3.py
功能：step() 接收動作空間輸入
劇情比喻：蛇學會「根據策略行動」，不再隨機亂走
"""

import random

class SnakeEnv:
    def __init__(self):
        self.directions = ['UP','RIGHT','DOWN','LEFT']
        self.direction = 'UP'
        self.pos = [100,100]

    def map_action(self, action):
        idx = self.directions.index(self.direction)
        if action == 'STRAIGHT':
            new_dir = self.directions[idx]
        elif action == 'RIGHT':
            new_dir = self.directions[(idx + 1) % 4]
        elif action == 'LEFT':
            new_dir = self.directions[(idx - 1) % 4]
        self.direction = new_dir

    def step(self, action):
        self.map_action(action)
        if self.direction == 'UP':
            self.pos[1] -= 1
        elif self.direction == 'DOWN':
            self.pos[1] += 1
        elif self.direction == 'LEFT':
            self.pos[0] -= 1
        elif self.direction == 'RIGHT':
            self.pos[0] += 1
        return self.pos

if __name__ == "__main__":
    env = SnakeEnv()
    for _ in range(5):
        action = random.choice(['LEFT','RIGHT','STRAIGHT'])
        print("Action:", action, "New Pos:", env.step(action))
