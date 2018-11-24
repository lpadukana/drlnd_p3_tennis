from collections import deque
import torch
from ddpg_agent import Agent
from unityagents import UnityEnvironment
import numpy as np
from contextlib import closing, contextmanager
import matplotlib.pyplot as plt


class UnityEnvWrapper:
    def __init__(self, no_graphics=False):
        self.env = UnityEnvironment(
            file_name="Tennis_Linux/Tennis.x86_64",
            no_graphics=no_graphics
        )
        self.env_info = None
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

    def close(self):
        self.env.close()


env = UnityEnvWrapper(no_graphics=False)
agent1 = Agent(state_size=env.n_states() + 1,
               action_size=env.n_actions(), random_seed=2)
agent2 = Agent(state_size=env.n_states() + 1,
               action_size=env.n_actions(), random_seed=2)

agent2.actor_local = agent1.actor_local
agent2.actor_target = agent1.actor_target
agent2.actor_optimizer = agent1.actor_optimizer

print(env.n_agents(), env.n_states(), env.n_actions())


def play():
    agent1.actor_local.load_state_dict(torch.load('checkpoint_actor_1.pth'))
    agent1.actor_local.eval()
    # agent2.actor_local.load_state_dict(torch.load('checkpoint_actor_2.pth'))
    # agent2.actor_local.eval()

    state = env.reset(train_mode=False)
    while True:
        state1 = np.concatenate([state[0], [1]])
        state2 = np.concatenate([state[1], [-1]])
        action1 = agent1.act(state1, add_noise=False)
        action2 = agent2.act(state2, add_noise=False)
        state, _, _, _ = env.step([action1, action2])


play()
