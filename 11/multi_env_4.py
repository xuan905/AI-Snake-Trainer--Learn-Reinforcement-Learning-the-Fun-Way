"""
multi_env_4.py
功能：監控不同環境下平均分數
劇情比喻：阿偉給蛇「環境觀察板」，看到每張地圖的表現
"""

import random
import matplotlib.pyplot as plt

MAP_SIZES=[(5,5),(8,8),(10,10)]
NUM_EPISODES=50

class SnakeEnv:
    def __init__(self,width,height):
        self.width=width
        self.height=height
        self.reset()
    def reset(self):
        self.snake=[(self.width//2,self.height//2)]
        self.food=(random.randint(0,self.width-1), random.randint(0,self.height-1))
        self.done=False
        return self.get_state()
    def get_state(self):
        return (self.snake[0], self.food)
    def step(self,action):
        x,y=self.snake[0]
        if action==0:y-=1
        elif action==1:x+=1
        elif action==2:y+=1
        elif action==3:x-=1
        new_head=(x,y)
        reward=-0.01
        self.done=False
        if new_head==self.food:
            reward=1
            self.snake.insert(0,new_head)
            self.food=(random.randint(0,self.width-1), random.randint(0,self.height-1))
        else:
            self.snake.insert(0,new_head)
            self.snake.pop()
        if not (0<=x<self.width and 0<=y<self.height) or new_head in self.snake[1:]:
            reward=-1
            self.done=True
        return self.get_state(),reward,self.done

# 訓練並計算平均分數
avg_rewards=[]
for w,h in MAP_SIZES:
    env=SnakeEnv(w,h)
    total=[]
    for ep in range(NUM_EPISODES):
        state=env.reset()
        ep_reward=0
        done=False
        while not done:
            action=random.randint(0,3)
            state,r,done=env.step(action)
            ep_reward+=r
        total.append(ep_reward)
    avg_rewards.append(sum(total)/NUM_EPISODES)

plt.bar([f"{w}x{h}" for w,h in MAP_SIZES], avg_rewards)
plt.xlabel("Map Size")
plt.ylabel("Average Reward")
plt.title("不同環境下平均分數")
plt.show()
