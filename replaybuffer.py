"""Replay Buffer"""
from collections import namedtuple, deque
import random
import numpy as np
import torch


class ReplayBuffer:
    """Fixed-size buffer to store expereince tuples.
    """

    def __init__(self, action_size, buffer_size, batch_size, device=torch.device('cpu')):
        """A ReplayBuffer

        Args:
          action_size (int): dimension of each action
          buffer_size (int): maximum size of buffer
          batch_size (int): size of each training batch
          seed (int): random seed
        """
        self.action_size = action_size
        self.device = device
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            'Experience',
            field_names=['state', 'action', 'reward', 'next_state', 'done'])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory"""
        self.memory.append(
            self.experience(state, action, reward, next_state, done))

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])
        ).float().to(self.device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])
        ).long().to(self.device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])
        ).float().to(self.device)
        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in experiences if e is not None])
        ).float().to(self.device)
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)
        ).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)
