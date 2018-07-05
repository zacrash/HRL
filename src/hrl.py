import random

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from collections import deque
import keras.backend as K
import numpy as np

from algorithms.DQN import DQN


"""
1. Get batch()
2. extract rewards, terminals, obs, actions, next_obs, goals, steps_left(tau)
3. Get action from policy
4. M is minibatch size
5. Q(s,sg,tau) -> (actions,rewards)
6. q_target = H + target_q_values, where tqv = max(Q'(o,g,tau-1)[0])
7. q_pred = max(Q(o,g,tau)[0])
8. Loss = MSE(q_pred - q_target)
"""

class HRL(DQN):
    """Subclass of DQN to use heuristics to improve sample efficiency"""
    def __init__(self, state_size, action_size, max_tau=10):
        self.max_tau = max_tau
        self.iterations = 20
        super().__init__(state_size, action_size)
        self.batch_size=32
        self.episodes=1000
        # Dont want any epsilon
        self.epsilon=0.0
        self.epsilon_min=0.0
        self.epsilon_decay=0.0

    def replay(self, batch_size):
        """Experience replay with a heuristic"""
        minibatch = self._get_minibatch()

        for i in range(self.iterations):
            for step in minibatch:
                state = step['state']
                next_state = step['nextState']
                goal = step['goal']
                tau = None
                actions = self.model.predict(state)[0]
                y = -1*self.H(next_state,goal)*(tau==0) + np.argmax(actions)*(tau!=0)
                self.model.fit(state,y,epochs=1,verbose=1)

    def _get_minibatch(self):
        """Grab minibatch from memory"""
        batch_size = min(self.batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)
        return minibatch
        
    def _relabel(self, batch):
        """Given a batch, attach a goal state"""
        k = random.randrange(len(batch))
        goal = batch[k]['state']
        

    def H(self, state, goal):
        """Heuristic for reach goal from state"""
        return 1
