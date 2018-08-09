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
        self.state_size = state_size
        self.action_size = action_size
        self.lr = 0.001
        self.batch_size = 32
        self.episodes = 1000
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = Q(self.state_size, action_size, self.lr)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.model.lr)
        # self.loss = nn.SmoothL1Loss()
        self.loss = nn.MSELoss()
        self.memory = deque(maxlen=2000)
        self.k = 4
        self.n = 1
        self.gamma = 0.95

    def load(self, model_file):
        """Load pre-trained model from file"""
        # TODO (@zac): Implement this for PyTorch
        self.model.load_state_dict(torch.load(model_file))

    def save(self, model_file):
        """Save model parameters"""
        torch.save(self.model.state_dict(), model_file)

    def remember(self, state, action, reward, next_state, done):
        """Append step to memory"""
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        self.memory.append([state, action, reward, next_state, done])

    def act(self, state):
        """Take best action"""
        state = torch.tensor(state, dtype=torch.float)
        # Epsilon-greedy
        if np.random.rand() <= self.epsilon:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            return random.randrange(self.action_size)
        action_values = self.model(state).detach().numpy()
        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return np.argmax(action_values)

    # TODO (@zac): state_action_values = policy_net(state_batch).gather(1, action_batch) -> CLEAN!
    def replay(self):
        """Experience replay with a heuristic"""
        minibatch = self._get_minibatch()
        for state, action, reward, next_state, done in minibatch:
            target = torch.tensor(0, dtype=torch.float)
            if not done:
                target = reward + self.gamma * torch.max(self.model(next_state))

            inp = self.model(state)[int(action)]

            # Update network
            self.optimizer.zero_grad()
            output = self.loss(inp, target.detach())
            output.backward()
            # for param in self.model.parameters():
            #     print param.grad
            #     param.grad.data.clamp_(-1, 1)
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
