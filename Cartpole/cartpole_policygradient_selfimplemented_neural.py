"""
https://github.com/dnddnjs/OpenAI/blob/master/PG_DNN.py
"""

import numpy as np
import pickle
import gym
OUT_DIR = 'cartpole-experiment2'
render = False
hidden = 10  
in_size = 4
out_size = 1
batch_size = 10  
resume = False
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
    
class RL:
    def __init__(self, in_size, out_size):
        np.random.seed(1)
        self.gamma = 0.995
        self.decay_rate = 0.99
        self.learning_rate = 0.002
       
        if resume: self.model = pickle.load(open('Cart.p', 'rb'))
        else:
            self.model = {}
            self.model['W1'] = np.random.randn(hidden, in_size) / np.sqrt(in_size)
            self.model['W2'] = np.random.randn(hidden, hidden) / np.sqrt(hidden)
            self.model['W3'] = np.random.randn(hidden) / np.sqrt(hidden)
        self.grad_buffer = []
        for i in range(batch_size):    
            self.grad_buffer.append({k: np.zeros_like(v) for k,v in self.model.iteritems()})
        self.rmsprop_cache = {k : np.zeros_like(v) for k,v in self.model.iteritems()}
        
        
    def discount_rewards(self, r):
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(xrange(0, r.size)):
            running_add = running_add * self.gamma + r[t]
            discounted_r[t] = running_add
        
        return discounted_r
            
    def policy_forward(self, x):
        h1 = np.dot(self.model['W1'], x)
        h1 = sigmoid(h1)
        h2 = np.dot(self.model['W2'], h1)
        h2 = sigmoid(h2)
        logp = np.dot(self.model['W3'], h2)
        p = sigmoid(logp)
        return p, h1, h2
    
    def policy_backward(self, eph1, eph2, epdlogp, epx, ep_num):
        dW3 = np.dot(eph2.T, epdlogp).ravel()
        dh2 = np.outer(epdlogp, self.model['W3'])
        eph2_dot = eph2*(1-eph2)
        dW2 = dh2 * eph2_dot
        dh1 = np.dot(dW2, self.model['W2'])
        eph1_dot = eph1*(1-eph1)
        dW1 = dh1 * eph1_dot
        dW2 = np.dot(dW2.T, eph1)
        dW1 = np.dot(dW1.T, epx)        
        self.grad_buffer[ep_num%batch_size] = {'W1':dW1, 'W2':dW2, 'W3':dW3}
        
    def learning(self):
        tmp = self.grad_buffer[0]
        for i in range(1,batch_size):
            for k,v in self.model.iteritems():
                tmp[k] += self.grad_buffer[i][k]
           
        for k,v in self.model.iteritems():
            g = tmp[k] 
            self.rmsprop_cache[k] = self.decay_rate * self.rmsprop_cache[k] + (1 - self.decay_rate)*g**2
            self.model[k] += self.learning_rate * g / (np.sqrt(self.rmsprop_cache[k]) + 1e-5) 
        
        
        
def train(env):
    
    rl = RL(in_size, out_size)
    running_reward = None
    reward_sum, episode_num = 0,0
    xs,h1s,h2s,dlogps,drs = [],[],[],[],[]
    
    for i_episode in range(2000):
        done = False
        observation = env.reset()
        
        while not done:
            x = observation    
            if render: env.render()
            act_prob, h1, h2 = rl.policy_forward(x)
            action = 1 if np.random.uniform() < act_prob else 0
            xs.append(x)
            h1s.append(h1)
            h2s.append(h2)
            y = action
            dlogps.append(y - act_prob)
    
            observation, reward, done, info = env.step(action)
            reward_sum += reward
            drs.append(reward)
        
        if done:
            episode_num += 1
            print "episode : " + str(episode_num) + ", reward : " + str(reward_sum) + " " + str(act_prob) 
        
            epx = np.vstack(xs)
            eph1 = np.vstack(h1s)
            eph2 = np.vstack(h2s)
            epdlogp = np.vstack(dlogps)
            epr = np.vstack(drs)
            xs,h1s,h2s,dlogps,drs = [],[],[],[],[]
        
            discounted_epr = rl.discount_rewards(epr)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)
        
            epdlogp *= discounted_epr
        
            rl.policy_backward(eph1,eph2,epdlogp,epx,episode_num)
            rl.learning()
        
            reward_sum = 0
    
    
    
if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    env.monitor.start(OUT_DIR, force=True)
    train(env)
    env.monitor.close()