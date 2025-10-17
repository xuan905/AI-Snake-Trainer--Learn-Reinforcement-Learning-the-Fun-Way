"""
dqn_env_2.py
功能：Experience Replay 緩衝池
劇情比喻：蛇學會「回顧過去經驗」，像翻閱行動手冊
"""

import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

if __name__=="__main__":
    buffer = ReplayBuffer(100)
    for i in range(10):
        buffer.push([i]*10, i%3, i*0.1, [(i+1)*1.0]*10, False)
    print("Buffer length:", len(buffer))
    states, actions, rewards, next_states, dones = buffer.sample(5)
    print("Sampled actions:", actions)
