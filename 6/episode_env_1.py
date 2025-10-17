"""
episode_env_1.py
功能：單一遊戲回合迴圈
劇情比喻：阿偉讓蛇完成一場遊戲，理解「勝利或失敗」的意義
"""

import random

STATE_SIZE = 128
ACTIONS = ['LEFT','RIGHT','STRAIGHT']

def step(state, action):
    next_state = (state + action) % STATE_SIZE
    reward = 1 if next_state % 10 == 0 else -0.01
    done = next_state % 50 == 0
    return next_state, reward, done

if __name__=="__main__":
    state = 0
    done = False
    while not done:
        action = random.choice(range(len(ACTIONS)))
        next_state, reward, done = step(state, action)
        print(f"State:{state}, Action:{ACTIONS[action]}, Reward:{reward}")
        state = next_state
    print("Single Episode Finished")
