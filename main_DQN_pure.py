from DQN import DQN
from utils import train_agent

from fourrooms import FourRoomsMatrix


dqn, eval_dqn = train_agent(DQN(explore=False, exploration_factor=1e0), FourRoomsMatrix(layout='4rooms'), exploration_steps=int(1e3), batch_size=10)
plt.plot(eval_dqn)
plt.title('DQN agent performance under end-to-end training process')