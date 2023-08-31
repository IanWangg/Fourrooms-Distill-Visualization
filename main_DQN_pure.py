from DQN import DQN
from utils import train_agent

from fourrooms import FourRoomsMatrix

import random
import numpy as np
import torch
# torch.use_deterministic_algorithms(True)

for seed in [4, 3, 2, 1, 0]:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(0)
    _, eval_dqn = train_agent(
        DQN(explore=False, exploration_factor=1e0), 
        FourRoomsMatrix(layout='4rooms', goal=80), 
        exploration_steps=int(1e3), 
        batch_size=10,
        filename=f"pure_dqn_{seed}"
    )
# plt.plot(eval_dqn)
# plt.title('DQN agent performance under end-to-end training process')