import random
from collections import namedtuple
import numpy as np
import torch

Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])


class ReplayBuffer:
    """
    Reinforcement learning replay buffer for training data
    """

    def __init__(self):
        self.memory = []

    def push(self, state, action, reward, next_state, done):
        """saving an experience tuple"""
        experience = Experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self, batch_size):
        """randomly sampling a batch of experiences"""
        tem = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*tem)
        states, actions, rewards, next_states, dones = np.stack(states), np.stack(actions), np.stack(rewards), np.stack(
            next_states), np.stack(dones)
        states, actions, rewards, next_states, dones = torch.FloatTensor(states), torch.FloatTensor(
            actions), torch.FloatTensor(rewards), torch.FloatTensor(next_states), torch.FloatTensor(dones)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        """return the length of replay buffer"""
        return len(self.memory)


if __name__ == '__main__':
    buffer = ReplayBuffer()
    for i in range(1000):
        buffer.push(np.array([1, 2, 3]), np.array(4), np.array(5), np.array([6, 7, 8]), np.array(0))
    print(buffer.sample(20))
