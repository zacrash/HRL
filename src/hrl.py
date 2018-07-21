import random
import math

from collections import deque
import numpy as np

import torch
from model import Q
import torch.optim as optim
import torch.nn as nn

torch.set_default_tensor_type('torch.FloatTensor')


"""TODO:
1. Action options are both [0.0, 0.0]. Not a whole lot to choose from there...
2. Need to sample differently based on tau. Write function to grab samples (paths)
3. Optimize storage and random access with numpy
4. Create heuristics
5. Parallelize code by running heuristic on separate process (or thread with cython)
"""

class HRL:
    """Subclass of DQN to use heuristics to improve sample efficiency"""
    # Just 1 because we are using an int
    tau_size = 1
    
    def __init__(self, state_size, action_size, max_tau=10):
        self.max_tau = max_tau
        self.iterations = 1
        self.state_size = state_size
        self.action_size = action_size
        self.goal_size = state_size
        self.lr = 0.001
        self.batch_size=128
        self.episodes=1000
        self.epsilon=1.0
        self.epsilon_min=0.01
        self.epsilon_decay=0.995
        self.model = Q(self.state_size + self.goal_size + self.tau_size, action_size, self.lr)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.model.lr)
        #self.loss = nn.SmoothL1Loss()
        self.loss = nn.MSELoss()
        self.memory = deque(maxlen=2000)
        self.tau = None


    def load(file, training):
        """Load pretrained model from file"""
        # TODO (@zac): Implement this for PyTorch
        self.model = load_model()
        

    def remember(self, state, action, reward, next_state, done, goal, tau):
        """Append step to memory"""
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        goal = torch.tensor(goal, dtype=torch.float)
        tau = torch.tensor([tau], dtype=torch.float)
        self.memory.append([state, action, reward, next_state, done, goal, tau])


    def act(self, state, goal, tau):
        """Take best action"""
        state = torch.tensor(state, dtype=torch.float)
        goal = torch.tensor(goal, dtype=torch.float)
        tau = torch.tensor(tau, dtype=torch.float)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        action_values = self.model(state, goal, tau).detach().numpy()
        return np.argmax(action_values[0])

    
    def replay(self, batch_size):
        """Experience replay with a heuristic"""
        minibatch = self._get_minibatch()
        for i in range(self.iterations):
            for state, action, reward, next_state, done, goal, tau in minibatch:
                q_pred = self.model(next_state, goal, tau-1)
                y_pred, _ = torch.tensor(q_pred.max(-1), dtype=torch.float, requires_grad=True)
                if done:
                    y_target = reward
                else:
                    y_target = -1*self.H(next_state,goal)*(int(tau)==0) + \
                                torch.max(y_pred)*(int(tau)!=0)
                    #print(self.H(next_state, goal))

                target_f = self.model(state, goal, tau-1)
                target_f[int(action)] = y_target
                target_f = target_f.detach()
                
                # Update network
                self.optimizer.zero_grad()
                output = self.loss(self.model(state, goal, tau-1), target_f)
                output.backward()
                self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def get_model():
        """Accessor for model parameters"""
        return self.model
                
  
    def _get_minibatch(self):
        """Grab minibatch from memory"""
        tau = np.random.randint(0, self.max_tau)
        # TODO (@zac): This is O(n) just to get a random number...
        try:
            # current observation is at len(self.memory) -1 so look past that
            k = np.random.randint(0, len(self.memory)-2)
        except ValueError:
            k = 0
        goal = self.memory[k][0]
        batch_size = min(self.batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)
        # TODO (@zac): This is absurdly inefficient (but easy :) )
        for i, (state, _, reward, _, _, _, _) in enumerate(minibatch):
            minibatch[i][2] = self.H(state, goal)
            minibatch[i][6] = torch.tensor([tau], dtype=torch.float)
            #print(minibatch[i][2].detach().numpy())

        return minibatch
        

    def H(self, state, goal):
        """Heuristic for reach goal from state"""
        return abs(state[2]*180/(math.pi))
