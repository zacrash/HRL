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
        # Dont want any epsilon
        self.epsilon=0.0
        self.epsilon_min=0.0
        self.epsilon_decay=0.0
        self.model = Q(self.state_size + self.goal_size + self.tau_size, action_size, self.lr)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.model.lr)
        #self.loss = nn.SmoothL1Loss()
        self.loss = nn.MSELoss()
        self.memory = deque(maxlen=2000)


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
        #print(state)
        #print(goal)
        #print(tau)
        action_values = self.model(state, goal, tau).detach().numpy()
        print (action_values)
        return np.argmax(action_values[0])

    
    def replay(self, batch_size):
        """Experience replay with a heuristic"""
        minibatch = self._get_minibatch()
        for i in range(self.iterations):
            for state, action, reward, next_state, done, goal, tau in minibatch:
                actions = self.model(state, goal, tau-1)
                #print(actions)
                #_, y_pred = torch.tensor(actions.max(0), dtype=torch.float, requires_grad=True)
                y_pred, _ = actions.max(0)
                #print("y: {}".format(y_pred))
                tau = int(tau[0])
                y_target = -1*self.H(next_state,goal)*(tau==0) + y_pred*(tau!=0)
                y_target = y_target.detach()
                
                # Update network
                self.optimizer.zero_grad()
                output = self.loss(y_pred, y_target)
                output.backward()
                self.optimizer.step()


    def get_model():
        """Accessor for model parameters"""
        return self.model
                

    def _get_minibatch(self):
        """Grab minibatch from memory"""
        tau = np.random.randint(0, self.max_tau)
        # TODO (@zac): This is O(n) just to get a random number...
        try:
            k = np.random.randint(1, len(self.memory))
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
        
    def _relabel(self, batch):
        """Given a batch, attach a goal state"""
        k = random.randrange(len(batch))
        goal = batch[k]['state']
        

    def H(self, state, goal):
        """Heuristic for reach goal from state"""
        #print(state[2]*180/(math.pi)**(0.1))
        return state[2]*180/(math.pi)**(0.1)
