import random

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, MaxPooling2D, LSTM
from keras.optimizers import Adam
from collections import deque
import keras.backend as K
import numpy as np

from her_replay_buffer import HerReplayBuffer
from algorithms import DQN


class HRL(DQN):
    def __init__(self, state_size, action_size, max_tau=10):
        self.max_tau = max_tau
        self.iterations = 20
        DQN.__init__(self, state_size=state_size, action_size=action_size)
        self.memory = HerReplayBuffer(2000)

    def replay(self, batch_size):
        """Experience replay with a heuristic"""
        batch_size = min(self.batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)
        minibatch = self.relabel(minibatch)
        for i in range(self.iterations):
            for step in minibatch:
                state = step['state']
                next_state = step['nextState']
                goal = step['goal']
                tau = None
                actions = self.model.predict(state)[0]
                y = -1*self.H(next_state,goal)*(tau==0) + np.argmax(actions)*(tau!=0)
                self.model.fit(state,y,epochs=1,verbose=0)
        
    def _relabel(self, batch):
        """Given a batch, attach a goal state"""
        k = random.randrange(len(batch))
        goal = batch[k]['state']
        

    def H(self, state, goal):
        """Heuristic for reach goal from state"""
        pass
