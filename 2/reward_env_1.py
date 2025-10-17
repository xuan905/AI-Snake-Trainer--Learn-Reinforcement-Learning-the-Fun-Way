"""
reward_env_1.py
功能：建立獎懲機制基礎框架
劇情比喻：阿偉開始教蛇理解「行為有意義」，每個動作都有價值
"""

import pygame
import random

class RewardSnakeEnv:
    def __init__(self, width=400, height=400, snake_size=20):
        pygame.init()
        self.width = width
        self.height = height
        self.snake_size = snake_size
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Snake Reward Environment")
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

    def get_state(self):
        return {'snake': self.snake_pos, 'food': self.food_pos}

    def step(self, action):
        self.direction = action
        if self.direction == 'UP':
            self.snake_pos[1] -= self.snake_size
        elif self.direction == 'DOWN':
            self.snake_pos[1] += self.snake_size
        elif self.direction == 'LEFT':
            self.snake_pos[0] -= self.snake_size
        elif self.direction == 'RIGHT':
            self.snake_pos[0] += self.snake_size

        reward = 0
        done = False

        # 簡單撞牆死亡
        if (self.snake_pos[0] < 0 or self.snake_pos[0] >= self.width or
            self.snake_pos[1] < 0 or self.snake_pos[1] >= self.height):
            reward = -1
            done = True

        # 蛇吃食物
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

# 測試
if __name__ == "__main__":
    env = RewardSnakeEnv()
    running = True
    state = env.reset()
    import random
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        action = random.choice(['UP','DOWN','LEFT','RIGHT'])
        state, reward, done = env.step(action)
        print("Reward:", reward)
        env.render()
        if done:
            print("Game Over! Score:", env.score)
            state = env.reset()
    pygame.quit()
