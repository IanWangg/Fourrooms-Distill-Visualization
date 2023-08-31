import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer

class QNet(nn.Module):
    def __init__(self, output_feature=1):
        super(QNet, self).__init__()

        self.output_feature = output_feature

        self.fc1 = layer_init(nn.Linear(169, 256))
        self.fc2 = layer_init(nn.Linear(256, 256))
        self.fc3 = layer_init(nn.Linear(256, output_feature * 4))
        # self.fc = layer_init(nn.Linear(169, output_feature * 4))

        self.to(device)


    def forward(self, state):
        batch_size = state.shape[0]
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        return self.fc3(x).view(batch_size, 4, self.output_feature).squeeze(-1)

class ExplorationModule(nn.Module):
    def __init__(self):
        super(ExplorationModule, self).__init__()
        self.fc1 = nn.Linear(169+1, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 169)
    
        self.to(device)
    
    def forward(self, state, action, next_state):
        cat = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        pred_next_state = self.fc3(x)

        estimation_residule = (pred_next_state - next_state).pow(2).mean(1) / 2
        # print(estimation_residule.shape)
        return estimation_residule.unsqueeze(-1)

class ExplorationModule(object):
    def __init__(self, factor=1.0):
        self.table = {}
    
    def __call__(self, state, action, next_state):
        pass


class DQN(object):
    def __init__(
        self,
        tau=5e-3,
        discount=0.99,
        initial_eps=1,
        end_eps=0.1,
        eps_decay_period=2e4,
        eval_eps=0.1,
        explore=False,
        exploration_factor=1.0,
    ):
        self.tau = tau
        self.discount = discount

        self.Q = QNet(output_feature=1)
        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer = torch.optim.RMSprop(self.Q.parameters(), lr=3e-4)

        self.initial_eps = initial_eps
        self.end_eps = end_eps
        self.slope = (self.end_eps - self.initial_eps) / eps_decay_period
        self.eval_eps = eval_eps

        self.iterations = 0

        # Exploration Module
        self.explore = explore
        if explore:
            self.exploration_module = ExplorationModule()
            self.exploration_factor = exploration_factor
            self.exploration_module_optimizer = torch.optim.Adam(self.exploration_module.parameters(), lr=1e-4)

    def select_action(self, state, eval=False):
        eps = self.eval_eps if eval \
            else max(self.slope * self.iterations + self.initial_eps, self.end_eps)

        if np.random.uniform(0,1) > eps:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(device)
                return int(self.Q(state).argmax(1))
        else:
            return np.random.randint(4)
    
    def train(self, replay_buffer, batch_size=32):
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # Compute the target Q value
        with torch.no_grad():
            next_action = self.Q(next_state).argmax(1, keepdim=True)
            target_Q = (
                reward + not_done * self.discount *
                self.Q_target(next_state).gather(1, next_action).reshape(-1, 1)
            )

            if self.explore:
                # compute bonus and add to target
                int_bonus = self.exploration_module(state, action, next_state) * self.exploration_factor
                target_Q += int_bonus

        current_Q = self.Q(state).gather(1, action)

        Q_loss = F.smooth_l1_loss(current_Q, target_Q)

        self.Q_optimizer.zero_grad()
        Q_loss.backward()
        nn.utils.clip_grad_norm_(self.Q.parameters(), 5)
        self.Q_optimizer.step()
        
        if self.explore:
            uncertainty = self.exploration_module(state, action, next_state).sum()
            self.exploration_module_optimizer.zero_grad()
            uncertainty.backward()
            self.exploration_module_optimizer.step()

        self.iterations += 1
        self.soft_update()

    def soft_update(self):
        for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def copy_update(self):
        if self.iterations % self.target_update_frequency == 0:
                self.Q_target.load_state_dict(self.Q.state_dict())