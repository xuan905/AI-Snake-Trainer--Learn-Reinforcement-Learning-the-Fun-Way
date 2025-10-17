"""
eval_env_1.py
功能：記錄每回合總分，計算平均分數
劇情比喻：阿偉開始統計蛇的每場表現，像為每場比賽打分
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
    episodes = 10
    total_scores = []
    for ep in range(episodes):
        state = 0
        done = False
        total_reward = 0
        while not done:
            action = random.choice(range(len(ACTIONS)))
            state, reward, done = step(state, action)
            total_reward += reward
        total_scores.append(total_reward)
        print(f"Episode {ep+1} Total Reward: {total_reward}")
    avg_score = sum(total_scores)/episodes
    print(f"Average Score: {avg_score}")
