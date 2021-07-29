#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -*- coding: utf-8 -*-

import gym
from gridworld import CliffWalkingWapper, FrozenLakeWapper, GridWorld
from agent import QLearningAgent
import time

# global total steps 
global_total_steps = 0

def run_episode(env, agent, render=False):
    total_steps = 0  # 记录每个episode走了多少step
    total_reward = 0
    global global_total_steps
    obs = env.reset()  # 重置环境, 重新开一局（即开始新的一个episode）

    while True:
        action = agent.sample(obs)  # 根据算法选择一个动作
        next_obs, reward, done, _ = env.step(action)  # 与环境进行一个交互
        
        #print("done =", done)
        #if done == True:
        #    global_total_steps += 1 # 统计到达终点的次数
            

        # 训练 Q-learning算法
        agent.learn(obs, action, reward, next_obs, done)

        obs = next_obs  # 存储上一个观察值
        total_reward += reward
        total_steps += 1  # 计算step数
        
        if render:
            env.render()  #渲染新的一帧图形
        if done:          
            break
    
    global_total_steps += total_steps
    return total_reward, total_steps


def test_episode(env, agent):
    total_reward = 0
    obs = env.reset()
    while True:
        action = agent.predict(obs)  # greedy
        next_obs, reward, done, _ = env.step(action)
        total_reward += reward
        obs = next_obs
        time.sleep(0.05)
        env.render()
        if done:
            print('test reward = %.1f' % (total_reward))
            break


def main():
    # env = gym.make("FrozenLake-v0", is_slippery=False)  # 0 left, 1 down, 2 right, 3 up
    # env = FrozenLakeWapper(env)

    env = gym.make("CliffWalking-v0")  # 0 up, 1 right, 2 down, 3 left
    env = CliffWalkingWapper(env)
    
    agent = QLearningAgent(
        obs_n=env.observation_space.n,
        act_n=env.action_space.n,
        learning_rate=0.1,
        gamma=0.9,
        e_greed=0.1)
        
    agent.restore()
    test_episode(env, agent)
    
    

    is_render = False
    for episode in range(10000):
        ep_reward, ep_steps = run_episode(env, agent, is_render)
        print('Episode %s: steps = %s , reward = %.1f, global_total_steps = %s' % (episode, ep_steps,
                                                          ep_reward, global_total_steps))
        
        # write data into file 
        file_name = 'data_intersection_9.txt'
        with open(file_name, 'a') as file:
            file.write('Episode %s: steps = %s, reward = %.1f, global_total_steps = %s\n' % (episode, ep_steps,
                                                          ep_reward, global_total_steps))
        
        # 每隔20个episode渲染一下看看效果
        
        if episode % 100 == 0:
            is_render = True
        else:
            is_render = False
        
        if episode % 1000 == 0: 
            pass
            agent.save()
            #test_episode(env, agent)
        
    # 训练结束，查看算法效果
    test_episode(env, agent)
    agent.save()



if __name__ == "__main__":
    main()
