from DQN import DQN
from utils import train_agent

from fourrooms import FourRoomsMatrix


_, eval_dqn = train_agent(
    DQN(explore=False, exploration_factor=1e0), 
    FourRoomsMatrix(layout='4rooms', goal=80), 
    exploration_steps=int(1e3), 
    batch_size=10,
    filename="pure_dqn"
)
# plt.plot(eval_dqn)
# plt.title('DQN agent performance under end-to-end training process')