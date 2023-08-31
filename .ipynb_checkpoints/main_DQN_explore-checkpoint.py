from DQN import DQN
from utils import train_agent

from fourrooms import FourRoomsMatrix


for exploration_factor in [1e0, 1e1, 1e2, 1e3, 1e4]:
# for exploration_factor in [1e4]:
    _, eval_dqn = train_agent(
        DQN(explore=True, exploration_factor=exploration_factor), 
        FourRoomsMatrix(layout='4rooms', goal=80), 
        exploration_steps=int(1e3), 
        batch_size=10,
        filename=f"explore_dqn_{exploration_factor}",
    )