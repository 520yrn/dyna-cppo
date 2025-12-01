# utils/replay_buffer.py
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self.storage = deque(maxlen=capacity)

    def add_batch(self, obs, act, next_obs):
        """
        obs: [T, obs_dim]
        act: [T, act_dim]
        next_obs: [T, obs_dim]
        """
        for o, a, n in zip(obs, act, next_obs):
            self.storage.append((o.copy(), a.copy(), n.copy()))

    def sample(self, batch_size: int):
        idx = np.random.choice(len(self.storage), size=batch_size, replace=False)
        obses, acts, next_obses = [], [], []
        for i in idx:
            o, a, n = self.storage[i]
            obses.append(o); acts.append(a); next_obses.append(n)
        return (
            np.asarray(obses, dtype=np.float32),
            np.asarray(acts, dtype=np.float32),
            np.asarray(next_obses, dtype=np.float32),
        )

    def __len__(self):
        return len(self.storage)
