import gym

import torch

from hrl import HRL


def run():
    env = gym.make('CartPole-v0')
    env._max_episode_steps = None

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = HRL(state_size, action_size)

    timesteps = 10

    for e in range(agent.episodes):
        state = env.reset()
        goal = [0.0, 0.0, 0.0, 0.0]
        score = 0

        # Rollout
        for t in range(timesteps):
            score += 1
            # env.render()
            action = agent.act(state, goal, timesteps-t)
            next_state, _, done, _ = env.step(action)
            agent.remember(state, action, next_state, goal, timesteps-t, done)
            agent.augment_goals(state, action, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}".format(
                    e, agent.episodes, score, agent.epsilon))
                break

        # Perform optimization
        for _ in range(agent.n):
            agent.replay()


if __name__ == '__main__':
    run()

