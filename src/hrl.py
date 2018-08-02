import random
import math

from collections import deque
import numpy as np

import torch
from model import Q
import torch.optim as optim
import torch.nn as nn
torch.set_default_tensor_type('torch.FloatTensor')


class HRL:
    """Subclass of DQN to use heuristics to improve sample efficiency"""
    # Just 1 because we are using an int
    tau_size = 1
    
    def __init__(self, state_size, action_size, max_tau=5):
        self.max_tau = max_tau
        self.iterations = 3
        self.state_size = state_size
        self.action_size = action_size
        self.goal_size = state_size
        self.lr = 0.001
        self.batch_size = 128
        self.episodes = 1000
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = Q(self.state_size + self.goal_size + self.tau_size, action_size, self.lr)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.model.lr)
        # self.loss = nn.SmoothL1Loss()
        self.loss = nn.MSELoss()
        self.memory = deque(maxlen=2000)
        self.tau = self.max_tau
        self.k = 4
        self.n = 2

    @staticmethod
    def load(file, training):
        """Load pretrained model from file"""
        # TODO (@zac): Implement this for PyTorch
        self.model = load_model()

    def remember(self, state, action, next_state, goal, tau, done):
        """Append step to memory"""
        state = torch.tensor(state)
        action = torch.tensor(action)
        next_state = torch.tensor(next_state)
        goal = torch.tensor(goal)
        tau = torch.tensor(tau)
        self.memory.append([state, action, next_state, goal, tau, done])

    def act(self, state, goal, tau):
        """Take best action"""
        state = torch.tensor(state, dtype=torch.float)
        goal = torch.tensor(goal, dtype=torch.float)
        # Epsilon-greedy
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        action_values = self.model(state, goal, tau).detach().numpy()
        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return np.argmax(action_values[0])

    # TODO (@zac): Make this replay function according to your algorithm
    def replay(self):
        """Experience replay with a heuristic"""
        minibatch = self._get_minibatch()
        for state, action, next_state, goal, tau, done in minibatch:
            target = -1 * self.h(state, goal) * (tau==0) + self.model(next_state, goal, tau-1) * (tau!=0)
            predicted = self.model(state, goal, tau)

            # Update network
            self.optimizer.zero_grad()
            output = self.loss(predicted, target)
            output.backward()
            self.optimizer.step()

    def augment_goals(self, state, action, next_state, done):
        """Sample a goal and tau"""
        tau = np.random.randint(0, self.max_tau)
        # Goal is sampled from a state a random number of steps after present (note, present in deque is right-most)
        goal = self.memory[np.random.randint(len(self.memory)-1, 0)][0]
        self.remember(state, action, next_state, goal, tau, done)

    def get_model(self):
        """Accessor for model parameters"""
        return self.model

    def _get_minibatch(self):
        """Grab minibatch from memory"""
        batch_size = min(self.batch_size, len(self.memory))
        # TODO (@zac): This is absurdly inefficient (but easy :) )
        minibatch = random.sample(self.memory, batch_size)
        return minibatch

    @staticmethod
    def h(state, goal):
        """Heuristic for reach goal from state"""
        return abs(state[2]*180/math.pi - goal[2]*180/math.pi)
