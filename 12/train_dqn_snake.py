"""
train_dqn_snake.py
功能：用 DQN 訓練 Snake，完成後存檔權重
劇情比喻：阿偉讓蛇反覆練習，學會自主尋找食物的策略
"""

import pygame
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

# ----------------------------
# Snake 環境
# ----------------------------
class SnakeGame:
    def __init__(self, width=200, height=200, block_size=20):
        self.width = width
        self.height = height
        self.block_size = block_size
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
# Replay Buffer
# ----------------------------
class ReplayMemory:
    def __init__(self, capacity=20000):
        self.memory = deque(maxlen=capacity)
    
    def push(self, transition):
        self.memory.append(transition)
    
    def sample(self, batch_size):
        idx = np.random.choice(len(self.memory), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.memory[i] for i in idx])
        return torch.stack(states), torch.tensor(actions), torch.tensor(rewards, dtype=torch.float32), torch.stack(next_states), torch.tensor(dones, dtype=torch.float32)
    
    def __len__(self):
        return len(self.memory)

# ----------------------------
# 訓練主程式
# ----------------------------
def train():
    env = SnakeGame()
    state_dim = 11
    action_dim = 3
    model = DQN(state_dim, action_dim)
    target_model = DQN(state_dim, action_dim)
    target_model.load_state_dict(model.state_dict())
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    memory = ReplayMemory()
    
    episodes = 10000
    batch_size = 128
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995
    target_update = 10
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        step_count = 0
        while not env.done:
            # ε-greedy 選擇動作
            if random.random() < epsilon:
                action = random.randint(0,2)
            else:
                with torch.no_grad():
                    q_values = model(state)
                    action = torch.argmax(q_values).item()
            
            next_state, reward, done = env.step(action)
            memory.push((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            step_count += 1
            
            # 更新 DQN
            if len(memory) >= batch_size:
                s, a, r, s_next, d = memory.sample(batch_size)
                q_pred = model(s).gather(1, a.unsqueeze(1)).squeeze()
                with torch.no_grad():
                    q_next = target_model(s_next).max(1)[0]
                q_target = r + gamma * q_next * (1-d)
                loss = criterion(q_pred, q_target)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # 更新 epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        
        # 更新 target model
        if episode % target_update == 0:
            target_model.load_state_dict(model.state_dict())
        
        print(f"Episode {episode+1}/{episodes} | Total Reward: {total_reward} | Steps: {step_count} | Epsilon: {epsilon:.3f}")
    
    # 存檔模型權重
    torch.save(model.state_dict(), "dqn_snake.pth")
    print("訓練完成，模型已存檔為 dqn_snake.pth")

if __name__ == "__main__":
    train()
