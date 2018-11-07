"""
THIS IS A SIMPLE IMPL OF Q-LEARNING
"""

# ENVIRONMENT_NAME = 'CartPole-v0'

ENVIRONMENT_NAME = "FrozenLake-v0"

import numpy as np
from matplotlib import pyplot as plt
import sys
import gym
import os
import pickle

EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
#exploration
epsilon = 0.01
#discountin
gamma = 0.9
#learning rate
alpha = 0.01

#Create the game
env = gym.make(ENVIRONMENT_NAME)

# q_values = []
# for _ in range(env.observation_space.n):
#     q_values.append( np.zeros(env.action_space.n) )

# if os.path.exists("q_values.pkl"):
#     f = open("q_values.pkl", mode="rb")
#     q_values = pickle.load(f)
#     f.close()

q_values = np.zeros((env.observation_space.n, env.action_space.n))

"""
# FOR QUICK JUMP in q_values , u can try this for fun.
# THIS q_value table gives you 68% Prob that u will win the game.

q_values = [
 [-0.45281378, -0.57828233, -0.55344074, -0.58585993],
 [-3.23549769, -3.54655973 ,-3.78714103, -0.55897488],
 [-1.36752767, -1.59605439 ,-1.75030857, -0.60961639],
 [-3.2995912 , -3.09259603 ,-3.62097789, -0.54418721],
 [-0.46772898, -3.48882106 ,-3.43260058, -3.64545811],
 [ 0.         , 0.         , 0.        ,  0.        ],
 [-4.5646842  ,-7.30603213, -4.87979896, -6.20024309],
 [ 0.         , 0.         , 0.        ,  0.        ],
 [-3.60117131 ,-3.08206389, -3.56132852, -0.55471237],
 [-3.46191134 ,-0.76319961, -3.79161072, -4.39928466],
 [-1.71186727 ,-3.3657345 , -5.30873905, -4.30426594],
 [ 0.         , 0.          ,0.         , 0.        ],
 [ 0.          ,0.         , 0.         , 0.        ],
 [-3.27167827 ,-3.22469134, -0.22152574, -3.48593763],
 [-0.30887964 , 0.28778142 ,-0.07790995, -0.20299895],
 [ 0.          ,0.        ,  0.        ,  0.        ]]
"""
# q_values = np.zeros((env.observation_space.shape[0], env.action_space.n))


def Q(st, act=None):
    if act is None:
        return q_values[st]
    return q_values[st][act]

def policy(st):
   if np.random.rand() > epsilon:
      return np.argmax(Q(st))
   return np.random.randint(0, env.action_space.n)

win, loss = 0, 0

for e in range(EPISODES):

    #initialize
    state = env.reset()
    action = policy(state)

    terminate = False

    while not terminate:

        #env.render()

        new_state , reward , terminate , _ = env.step(action)

        if terminate:
            if reward == 0:
                loss += 1
                reward = -10
            else:
                win += 1
            #...
            td_target = reward
        else:
            new_action = policy(new_state)
            td_target = reward + gamma * Q(new_state, new_action)
        
        td_error = td_target - Q(state, action)

        q_values[state][action] = q_values[state][action] + (alpha * td_error)

        state = new_state
        action = new_action

    #sys.stdout.flush()
    c = 'Episode: %d, WIN: %d, LOSE: %d PERCENT = %.2f' %(e, win, loss, win/ ( win + loss ))
    print(c,  end='\r')

print (c)
print(q_values)
# f = open("q_values.pkl", "wb")
# pickle.dump(q_values , f)
# f.close()