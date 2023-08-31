from DQN import DQN
from utils import train_agent_rollin

from fourrooms import FourRoomsMatrix
import random
import numpy as np
import torch
# torch.use_deterministic_algorithms(True)

for seed in [4, 3, 2, 1, 0]:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(0)

    for exploration_factor in [1e0, 1e1, 1e2, 1e3, 1e4]:
    # for exploration_factor in [1e4]:
        _, eval_dqn = train_agent_rollin(
            agent=DQN(explore=False, exploration_factor=exploration_factor), 
            explore_agent=DQN(explore=True, exploration_factor=exploration_factor), 
            env=FourRoomsMatrix(layout='4rooms', goal=80), 
            exploration_steps=int(1e3), 
            batch_size=10,
            filename=f"rollin_dqn_{exploration_factor}_{seed}"
        )