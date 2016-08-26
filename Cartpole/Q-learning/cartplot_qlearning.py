"""
https://github.com/Skalextric/openai/blob/master/test.py
"""

import gym
from qlearn.qlearningNN import qneuralagent
import numpy as np

env = gym.make('CartPole-v1')

print env.observation_space
agent = qneuralagent(env, done_reward=-1, epsilon=0.01, epsilon_decay=1)

rend = False
learning = True
i_episode = 0
last_episodes = [0]
episode_steps = 200

while 1:
    x = np.average(last_episodes)
    print "Average reward: {} \n".format(x)
    print agent.epsilon
    if x > 200:
        learning = False

    if not learning:
        print "Finis learning in episode #{}".format(i_episode)
        # break

    i_episode += 1

    observation = env.reset()
    print "Episode #{}".format(i_episode)

    if i_episode % 20 == 0:
        rend = True
    else:
        rend = False

    total_reward = 0
    for t in range(episode_steps):
        if not learning or rend:
            env.render()
        action = agent.getAction(observation)
        newObservation, reward, done, info = env.step(action)
        total_reward += reward
        if learning:
            agent.update(observation, action, reward, newObservation, done)
        else:
            pass
            # print "Stopped learning"
        observation = newObservation
        if done or (t + 1) == episode_steps:
            print("Episode finished after {} timesteps".format(t + 1))
            print "Last reward was {} ".format(total_reward)
            if len(last_episodes) < 100:
                last_episodes.append(total_reward)
            else:
                last_episodes.pop(0)
                last_episodes.append(total_reward)
            break