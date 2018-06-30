import numpy as np
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, MaxPooling2D, LSTM
from keras.optimizers import Adam
from collections import deque
import random


# Deep Q-learning Agent
class DQN(object):
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma=0.95
        self.epsilon=1.0
        self.epsilon_min=0.01
        self.epsilon_decay=0.995
        self.learning_rate=0.001

        self.batch_size=32
        self.episodes=1000

        self.memory = deque(maxlen=2000)

        self.model = self._build_model()


    def _build_model(self):
        """ From Mnih et al. "Playing atari with deep reinforcement learning." 2013. """
        model = Sequential()

        model.add(Convolution2D(32,(8,8), strides=(4,4), activation='relu',
                                input_shape=self.state_size))
        model.add(Convolution2D(64,(4,4), strides=(2,2), activation='relu'))
        model.add(Convolution2D(64,(3,3), activation='relu'))

        model.add(Flatten())
        model.add(Dense(512, activation='relu', input_dim=self.state_size))
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        return model

    def _build_model_linear(self):
        model = Sequential()
        model.add(Dense(24, activation='relu', input_dim=self.state_size))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def load(file, training):
        # Load model from file
        self.model = load_model('drqn_model.h5')        
        if not training:
            # Only want greedy decisions
            self.epsilon = 0.0

    """ Save values to memory """
    def remember(self, state, action, reward, nextState, done):
        self.memory.append((state, action, reward, nextState, done))

    def act(self, state):
        """ Carry out best or random action """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        action_values = self.model.predict(state)
        return np.argmax(action_values[0])

    def replay(self, batch_size):
        """ Replay minibatches """
        batch_size = min(self.batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                target[0][action] = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))

            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_model():
        return self.model

class DRQN(DQN):
    """ Deep Recurrent Q-Learning Network """
    """ https://arxiv.org/pdf/1507.06527.pdf """
    def __init__(self, state_size, action_size):
        super().__init__(state_size, action_size)
        self.model = self._build_model()

    def _build_model(self):
        # Input = [pedx-x, pedy-y, orthogonalVelocitiesDifference, pedTheta-Theta, headTheta]^T
        model = Sequential()

        # Convolution Layers
        model.add(Convolution2D(32,(8,8), strides=(4,4), activation='relu',
                                input_shape=self.state_size))

        model.add(Convolution2D(64,(4,4), strides=(2,2), activation='relu'))
        model.add(Convolution2D(64,(3,3), activation='relu'))

        # Fully Connected Layers
        model.add(Flatten())
        model.add(LSTM(units=32))

        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        return model

class DDQN(DQN):
    """DDQN class based on DQN base class above"""
    def __init__(self, state_size, action_size):
        print(self)
        super(DDQN, self).__init__(state_size, action_size)
        self.target_model = self._build_model_linear()
        self.model = self._build_model_linear()

    def update_target_model(self):
        """Update target network after episdoe"""
        self.target_model.set_weights(self.model.get_weights())

    def _huber_loss(self, target, prediction):
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)

    def _build_model_linear(self):
        model = Sequential()
        model.add(Dense(24, activation='relu', input_dim=self.state_size))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=self._huber_loss, optimizer=Adam(lr=self.learning_rate))
        return model

    def replay(self, batch_size):
        """ Replay a minibatches """
        batch_size = min(self.batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                target[0][action] = (reward + self.gamma * np.amax(self.target_model.predict(next_state)[0]))

            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
