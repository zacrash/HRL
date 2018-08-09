import random
import math

from collections import deque
import numpy as np

import torch
torch.set_default_tensor_type('torch.FloatTensor')
import torch.optim as optim
import torch.nn as nn

from model import Q

import time


class HRL:

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
        self.epsilon_decay = 0.999
        self.model = Q(self.state_size + self.goal_size + self.tau_size, action_size, self.lr)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.model.lr)
        self.loss = nn.SmoothL1Loss()
        #self.loss = nn.MSELoss()
        self.memory = deque(maxlen=2000)
        self.k = 4
        self.n = 2

    def load(self, model_file):
        """Load pre-trained model from file"""
        # TODO (@zac): Implement this for PyTorch
        self.model.load_state_dict(torch.load(model_file))

    def save(self, model_file):
        """Save model parameters"""
        torch.save(self.model.state_dict(), model_file)

    def remember(self, state, action, next_state, goal, tau, done):
        """Append step to memory"""
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        if goal is None and tau is None:
            if len(self.memory) <= 1:
                goal = next_state
                tau = 0
            else:
                goal = self.memory[np.random.randint(0, len(self.memory) - 1)][0]
                tau = np.random.randint(0, self.max_tau)
        goal = torch.tensor(goal, dtype=torch.float)
        tau = torch.tensor([tau], dtype=torch.float)
        self.memory.append([state, action, next_state, goal, tau, done])

    def act(self, state, goal, tau):
        """Take best action"""
        state = torch.tensor(state, dtype=torch.float)
        goal = torch.tensor(goal, dtype=torch.float)
        # Epsilon-greedy
        if np.random.rand() <= self.epsilon:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            return random.randrange(self.action_size)
        action_values = self.model(state, goal, tau).detach().numpy()
        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        print np.argmax(action_values)
        return np.argmax(action_values)

    # TODO (@zac): state_action_values = policy_net(state_batch).gather(1, action_batch) -> CLEAN!
    def replay(self):
        """Experience replay with a heuristic"""
        minibatch = self._get_minibatch()
        for state, action, next_state, goal, tau, done in minibatch:
            # target = -1 * self.h(state, goal) * (int(tau)==0) + torch.max(self.model(next_state, goal, tau-1)) * (int(tau)!=0)
            if int(tau) == 0:
                target = -1 * self.h(state, goal)
            else:
                target = torch.max(self.model(next_state, goal, tau-1))
            predicted = torch.max(self.model(state, goal, tau))

            # Update network
            self.optimizer.zero_grad()
            output = self.loss(predicted, target.detach())
            output.backward()
            self.optimizer.step()

    def augment_goals(self, state, action, next_state, done):
        """Sample a goal and tau"""
        if len(self.memory) <= 1:
            return
        tau = np.random.randint(0, self.max_tau)
        # Goal is sampled from a state a random number of steps after present (note, present in deque is right-most)
        goal = self.memory[np.random.randint(0, len(self.memory)-1)][0]
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
        # print abs(state[2]*180/math.pi - goal[2]*180/math.pi)**2
        # print state[2], goal[2]
        # print
        return torch.tensor(abs(state[2]*180/math.pi - goal[2]*180/math.pi), dtype=torch.float)
