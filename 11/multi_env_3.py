"""
multi_env_3.py
功能：支援多個地圖場景
劇情比喻：阿偉建立多個「平行世界」，蛇能在任意世界生存
"""

import random

MAP_SIZES = [(5,5),(8,8),(10,10)]
NUM_ENV = len(MAP_SIZES)

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

# 測試多個地圖
if __name__=="__main__":
    for i,(w,h) in enumerate(MAP_SIZES):
        env = SnakeEnv(w,h)
        state=env.reset()
        print(f"Env {i+1}: Map {w}x{h}, Initial state: {state}")
