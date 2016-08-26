import gym
from pylab import *
import numpy as np
import random, math
from tiles import *
from ACRL2 import *

#Initialize Hashing Tile Coder
numTilings = 8
cTableSize = 8192
cTable = CollisionTable(cTableSize, 'safe') 
print cTable
F = np.zeros(numTilings)
n = cTableSize

#Initialize Cartpole environemt
env = gym.make('CartPole-v0')
# print env.action_space
# print env.observation_space
# print(env.observation_space.high)

#Initialize Actor - Critic parameters
logging = 1
cart = ACRL2(0.97, 0, 0.1/numTilings, 0.01/numTilings, 0.7, n, logging)    #ACRL(gamma, alphaSig, alphaV, alphaU, lambda, n)
#note: having given alphaSig = 0 means that the Normal function from which action is derived, shall always have a default variance = 1

def loadFeatures(stateVars, featureVector):
	stateVars = list(stateVars)
	stateVars[0] += 1.2
	stateVars[1] += 0.07
	stateVars[0] *= 10
	stateVars[1] *= 100
	
	loadtiles(featureVector, 0, numTilings, cTable, stateVars)
	return featureVector
	""" 
	As provided in Rich's explanation
		   tiles                   ; a provided array for the tile indices to go into
		   starting-element        ; first element of "tiles" to be changed (typically 0)
		   num-tilings             ; the number of tilings desired
		   memory-size             ; the number of possible tile indices
		   floats                  ; a list of real values making up the input vector
		   ints)                   ; list of optional inputs to get different hashings
	"""

if __name__ == '__main__':
	print '==========================TEST=========================='
	numEpisodes = 0
	numRuns = 10
	avgReward = 0
	for i_episode in range(numEpisodes):
		tmp_reward = 0   #calculate reward for this particular episode
		prev_observation = env.reset() #x, x_dot, theta, theta_dot = state (position, velocity, angular position, angular velocity)
		for t in range(100):
			env.render()
			# action = env.action_space.sample()
			action = cart.getAction(prev_observation)
			new_observation, reward, done, info = env.step(action)   #this will form part of tile coding: (new_observation, reward), (X,y)
			# if logging: print '--->',action, new_observation, reward, done, info
			if done:
				reward = 0   #its weird that inspite of done == 1, the reward is still passed as 1??. Hence had to manually make it zero.
				cart.updates(reward, np.array(prev_observation), np.array(new_observation))
				print 'Episode finished(pole dropped) after ',t+1, ' timesteps. (Mean:',cart.mean, ' Variance:',cart.sigma,' Reward:',reward,')'
				print '------------------------------------------------------------------------------------------------------------------------------------'
				avgReward += tmp_reward   #check the last line to understand this
				break
			else:
				cart.updates(reward, np.array(prev_observation), np.array(new_observation))
				tmp_reward += reward
			prev_observation = new_observation
				

	print '==========================TEST=========================='
	print 'Avg Reward:', float(avgReward)/(numEpisodes*1.0)

"""
Issues Faced
0. This environment requires action that are [0,1]. But ACRL gives us a normal distro value. How to convert that to binary value?? 
1. The Avg reward does not go above 20. 
2. The delta values increase over time, due to which my mean for normal-distro becomes really large and I only get one kind of action. 
"""