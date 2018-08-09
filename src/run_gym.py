import gym

import torch

from hrl import HRL


def run():
    env = gym.make('CartPole-v0')
    env._max_episode_steps = None

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = HRL(state_size, action_size)

    for e in range(agent.episodes):
        state = torch.tensor(env.reset(), dtype=torch.float)
        score = 0
        done = False
        # Rollout
        for t in range(400):
            score += 1
            env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else 0
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            # agent.augment_goals(state, action, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}".format(
                    e, agent.episodes, score, agent.epsilon))
                break


        # Perform optimization
        #for _ in range(agent.n):



if __name__ == '__main__':
    run()

