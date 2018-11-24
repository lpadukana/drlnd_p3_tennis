from collections import deque
import torch
from ddpg_agent import Agent
from unityagents import UnityEnvironment
import numpy as np
from contextlib import closing, contextmanager
import matplotlib.pyplot as plt


@contextmanager
def muted_logs(log_name):
    import logging
    logger = logging.getLogger(log_name)
    old_level = logger.level
    old_propagate = logger.propagate
    logger.setLevel(logging.CRITICAL)
    logger.propagate = False
    try:
        yield
    finally:
        logger.setLevel(old_level)
        logger.propagate = old_propagate


class UnityEnvWrapper:
    def __init__(self, no_graphics=False):
        self.env = UnityEnvironment(
            file_name="Tennis_Linux/Tennis.x86_64",
            no_graphics=no_graphics
        )
        self.reset()

    def brain_name(self):
        return self.env.brain_names[0]

    def brain(self):
        return self.env.brains[self.brain_name()]

    def reset(self, train_mode=True):
        self.env_info = self.env.reset(train_mode=train_mode)[
            self.brain_name()]
        return self.env_info.vector_observations

    def n_agents(self):
        # self.states().shape[0]
        return len(self.env_info.agents)

    def n_actions(self):
        return self.brain().vector_action_space_size

    def states(self):
        return self.env_info.vector_observations

    def step(self, actions):
        self.env_info = self.env.step(actions)[self.brain_name()]
        return self.env_info.vector_observations, self.env_info.rewards, self.env_info.local_done, None

    def n_states(self):
        return self.states().shape[1]

    def info(self):
        print('Number of agents:', self.n_agents())
        print('Size of each action:', self.n_actions())
        print('Size of observations: {}'.format(self.n_states()))
        print('Example state:', self.states()[0])

    @classmethod
    def random_play(cls, n_episodes):
        env = cls()
        with closing(env):
            for i in range(1, n_episodes):
                states = env.reset(train_mode=False)
                scores = np.zeros(env.n_agents())
                while True:
                    actions = np.clip(np.random.randn(
                        env.n_agents(), env.n_actions()), -1, 1)
                    next_states, rewards, dones, _ = env.step(actions)
                    scores += rewards
                    states = next_states
                    if np.any(dones):
                        break

                print('Score (max over agents) from episode {}: {}'.format(
                    i, np.max(scores)))

    def close(self):
        self.env.close()


# with muted_logs('unityagents'), muted_logs('root'):
#     UnityEnvWrapper.random_play(n_episodes=15)


env = UnityEnvWrapper(no_graphics=False)
agent1 = Agent(state_size=env.n_states() + 1,
               action_size=env.n_actions(), random_seed=2)
agent2 = Agent(state_size=env.n_states() + 1,
               action_size=env.n_actions(), random_seed=2)

agent2.critic_local = agent1.critic_local
agent2.critic_target = agent1.critic_target
agent2.critic_optimizer = agent1.critic_optimizer

agent2.actor_local = agent1.actor_local
agent2.actor_target = agent1.actor_target
agent2.actor_optimizer = agent1.actor_optimizer

agent2.memory = agent1.memory

print(env.n_agents(), env.n_states(), env.n_actions())


def save(i_episode, scores1, scores2, mean_scores):
    print("Saving checkpoints...")
    torch.save(agent1.actor_local.state_dict(),
               'checkpoint_actor_1.pth')
    torch.save(agent2.actor_local.state_dict(),
               'checkpoint_actor_2.pth')
    torch.save(agent1.critic_local.state_dict(),
               'checkpoint_critic_1.pth')
    torch.save(agent2.critic_local.state_dict(),
               'checkpoint_critic_2.pth')
    torch.save(dict(episode=i_episode,
                    scores1=scores1,
                    scores2=scores2,
                    mean_scores=mean_scores),
               'scores.pth')


def train_agent(n_episodes=10000, print_every=100, target_score=0.5):
    scores1 = []
    scores2 = []
    mean_scores = []
    for i_episode in range(0, n_episodes + 1):
        state = env.reset(train_mode=True)
        agent1.reset()
        agent2.reset()
        score1 = 0
        score2 = 0
        while True:
            state1 = np.concatenate([state[0], [1]])
            state2 = np.concatenate([state[1], [-1]])
            action1 = agent1.act(state1)
            action2 = agent2.act(state2)
            next_state, reward, done, _ = env.step([action1, action2])
            next_state1 = np.concatenate([next_state[0], [1]])
            next_state2 = np.concatenate([next_state[1], [-1]])
            agent1.step(state1, action1, np.mean(reward),
                        next_state1, done[0])
            agent2.step(state2, action2, np.mean(reward),
                        next_state2, done[1])
            state = next_state
            score1 += reward[0]
            score2 += reward[1]
            if np.all(done):
                break

        scores1.append(score1)
        scores2.append(score2)
        mean_scores.append(np.mean([score1, score2]))
        mean1 = np.mean(scores1[-100:])
        mean2 = np.mean(scores2[-100:])
        mean_score = np.mean(mean_scores[-100:])

        print('Episode {}\t mean-score-1: {:.3f} mean-score-2: {:.3f} mean-score: {:.3f}'.format(
            i_episode, mean1, mean2, mean_score))

        if i_episode % print_every == 0:
            save(i_episode, scores1, scores2, mean_scores)

        if mean_score > target_score:
            save(i_episode, scores1, scores2, mean_scores)
            break

    return scores1


def main():
    scores = train_agent(target_score=1.0)

    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores) + 1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig('score.png')

    play()


def play():
    agent1.actor_local.load_state_dict(torch.load('checkpoint_actor_1.pth'))
    agent1.actor_local.eval()
    agent2.critic_local.load_state_dict(torch.load('checkpoint_critic_2.pth'))
    agent2.actor_local.eval()

    state = env.reset(train_mode=False)
    while True:
        state1 = np.concatenate([state[0], [1]])
        state2 = np.concatenate([state[1], [-1]])
        action1 = agent1.act(state1, add_noise=False)
        action2 = agent2.act(state2, add_noise=False)
        state, _, _, _ = env.step([action1, action2])
        # if done:
        #     break

#     env.close()


main()
