import gym
import numpy as np

from hrl import HRL


def run():
    env = gym.make('CartPole-v0')
    env._max_episode_steps = None

    state_size = env.observation_space.shape[0]
    print(env.observation_space)
    action_size = env.action_space.n
    agent = HRL(state_size, action_size)
    # agent.load("./save/cartpole-dqn.h5")

    for e in range(agent.episodes):
        state = env.reset()
        #state = np.reshape(state, (1, state_size))
        score = 0
        done = False
        while not done:
            score += 1
            #env.render()
            goal = [0.0, 0.0, 0.0, 0.0]
            action = agent.act(state, goal, [100-score])
            next_state, reward, done, info = env.step(action)
            reward = reward if not done else 0
            #next_state = np.reshape(next_state, (1, state_size))
            agent.remember(state, action, reward, next_state, done, state, 1)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                    .format(e, agent.episodes, score, agent.epsilon))
            if len(agent.memory) > agent.batch_size:
                agent.replay(agent.batch_size)


if __name__ == '__main__':
    run()

