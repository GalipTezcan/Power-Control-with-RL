#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 16:20:05 2017

@author: farismismar
"""

import random
import numpy as np
#from numpy import linalg as LA

import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.ticker as tick

from env import CustomEnv
from QLearningAgent import QLearningAgent as QLearner
#from DQNLearningAgent import DQNLearningAgent as QLearner
seed=0
agent = QLearner(seed=seed, state_size=3, action_size=5, batch_size=32)
env=CustomEnv(random_state=seed)
def run_agent(env, plotting = True,max_episodes_to_run=707):
    succ=[] #succsefull epsoides
    max_timesteps_per_episode = 20 # one AMR frame ms.

    for episode_index in np.arange(max_episodes_to_run):
        state,_ = env.reset()
        action = agent.begin_episode(state)
    
        # Recording arrays
        state_progress = ['start', 0]
        action_progress = ['start', 0]
        reward_progress = []
        alarm_reg = [0,0,0]
        pc_progress = []        
        network_progress = []
        reward=-100
        for timestep_index in range(max_timesteps_per_episode):
            # Perform the power control action and observe the new state.
            action = agent.act(state, reward)
            next_state, reward, terminated, turncated, _ = env.step(action)

            # make next_state the new current state for the next frame.
            state = next_state
            action_progress.append(action)
            state_progress.append(next_state)
                                
            if turncated == True:
                state_progress.append('ABORTED')
                
                
            if (terminated or turncated):
                state_progress.append('end')
                break                    

        if (terminated):
            succ.append(episode_index+1)
    print(env.retainability_score())
            

for i in range(10):
    
    env=CustomEnv(random_state=seed)
    run_agent(env, False,max_episodes_to_run=100*(i+1)) # %0.67

"""0.7599161156904436
0.7613640005243915
0.7610131858816576
0.7603599502792415
0.7602188193504223
0.7601465441968478
0.7599815405017718
0.7602481583817228
0.7603450721145284
0.760444021026806
PS C:\Users\galip\OneDrive\Masaüstü\Q-Learning-Power-Control-master\baseline> & C:/Users/galip/AppData/Local/Programs/Python/Python312/python.exe c:/Users/galip/OneDriv0.7016036655211912
0.7634807417974323
0.7602967011516689
0.7657109759629308
0.7662656231748628
0.7701015965166909
0.7677070328755722
0.7667568160222206
0.7661138211382114
0.7671385119326158
PS C:\Users\galip\OneDrive\Masaüstü\Q-Learning-Power-Control-master\baseline> & C:/Users/galip/AppData/Local/Programs/Python/Python312/python.exe c:/Users/galip/OneDriv0.7016036655211912
0.7634807417974323
0.7602967011516689
0.7657109759629308
0.7662656231748628
0.7701015965166909
0.7677070328755722
0.7667568160222206
0.7661138211382114
0.7671385119326158
PS C:\Users\galip\OneDrive\Masaüstü\Q-Learning-Power-Control-master\baseline> & C:/Users/galip/AppData/Local/Programs/Python/Python312/python.exe c:/Users/galip/OneDriv0.6770833333333333
0.7231481481481481
0.7888377445339471
0.7779735682819383
0.7680305131761442
0.766837899543379
0.7604685942173479
0.7600867678958785
0.7602967011516689
0.7642671292281006"""