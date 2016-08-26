from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import RMSprop
from random import uniform
import numpy as np
import random


class qneuralagent():
    def __init__(self, env, minibatch=40, replays=2000, learning_rate=0.00025, reward_discount=0.99, epsilon=0.9,
                 epsilon_decay=0.99, done_reward=None):
        self.env = env
        self.done_reward = done_reward
        self.minibatch = minibatch
        self.exp_buffer = replays
        self.exp_replay = []

        # Learning rate
        self.learning_rate = learning_rate

        # Reward Discount
        self.discount = reward_discount

        # Probability of random action
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        self.model = model = Sequential()

        model.add(Dense(128, init='lecun_uniform', bias=True, input_shape=self.env.observation_space.shape))
        model.add(Activation('relu'))

        model.add(Dense(128, init='lecun_uniform', bias=True))
        model.add(Activation('relu'))

        model.add(Dense(self.env.action_space.n, init='lecun_uniform'))
        model.add(Activation('linear'))

        rms = RMSprop(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=rms)

    def getQValues(self, state):
        return self.model.predict(state.reshape(1, -1), batch_size=1)

    def computeActionFromQValues(self, state):
        return np.argmax(self.getQValues(state))   #are we sure this will always give a -1/+1??

    def getAction(self, state):
        random = uniform(0, 1)
        if random <= self.epsilon:   #epsilon exploration in q-learning
            action = self.env.action_space.sample()
        else:
            action = self.computeActionFromQValues(state)
        self.epsilon *= self.epsilon_decay          #reduce random exploration over time
        return action

    def update(self, oldState, action, reward, newState, done):

        oldState = oldState.reshape(1, -1)
        newState = newState.reshape(1, -1)

        if len(self.exp_replay) < self.exp_buffer:
            self.exp_replay.append((oldState, action, reward, newState, done))
            qval = self.getQValues(oldState)  #returns a vector wih q-values of all possible actions from that state

            if self.done_reward is None and done:
                update_values = reward
            elif self.done_reward is not None and done:
                update_values = self.done_reward
            else:
                update_values = reward + (self.discount * np.max(self.getQValues(newState))) #td-learning

            qval[0][action] = update_values
            self.model.fit(oldState, qval, batch_size=1, nb_epoch=1, verbose=0)  #it may be fine if we don't do this, and only do it in the upcoming else loop

        else:
            self.exp_replay.pop(0)
            self.exp_replay.append((oldState, action, reward, newState, done))

            minibatch = random.sample(self.exp_replay, self.minibatch)
            X_train = []
            Y_train = []

            for example in minibatch:
                old_state, action, reward, new_state, done = example
                old_qval = self.getQValues(old_state)
                if self.done_reward is None and done:
                    update_values = reward
                elif self.done_reward is not None and done:
                    update_values = self.done_reward
                else:
                    update_values = reward + (self.discount * np.max(self.getQValues(new_state)))

                old_qval[0][action] = update_values
                X_train.append(old_state.reshape(oldState.shape[1], ))
                Y_train.append(old_qval.reshape(self.env.action_space.n, ))

            X_train = np.array(X_train)
            Y_train = np.array(Y_train)
            self.model.fit(X_train, Y_train, batch_size=self.minibatch, nb_epoch=1, verbose=0)