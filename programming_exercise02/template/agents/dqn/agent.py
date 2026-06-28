from copy import deepcopy

import torch
import numpy as np
import torch.nn.functional as F
from omegaconf import DictConfig
from tensordict import TensorDictBase
from torchrl.envs.utils import exploration_type, ExplorationType


class DeepQNetwork(torch.nn.Module):
    def __init__(self, cfg: DictConfig, action_space_spec, observation_space_spec):
        super().__init__()

        self.cfg = cfg
        self.action_space_spec = action_space_spec
        self.observation_space_spec = observation_space_spec

        self.net_output_dim = self.action_space_spec.space.n
        self.net_input_dim = self.observation_space_spec["observation"].shape[0]

        self.q_net = torch.nn.Sequential(
            torch.nn.Linear(self.net_input_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512), # single hidden layer with 512 units
            torch.nn.ReLU(),
            torch.nn.Linear(512, self.net_output_dim)
        )

        self.target_net = deepcopy(self.q_net)

        for param in self.target_net.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.Adam(
            self.q_net.parameters(), 
            lr=self.cfg.optim.lr
        )

        self._eps = self.cfg.exploration.eps_start_value

    def forward(self, tensordict: TensorDictBase):

        mode = exploration_type()
        obs = tensordict["observation"]
        batch_size = obs.shape[0]

        if mode == ExplorationType.RANDOM:
            actions = torch.randint(0, self.net_output_dim, (batch_size,), device=obs.device)
        
        else:
            q_values = self.q_net(obs)
            actions = torch.argmax(q_values, dim=-1)

        tensordict["action"] = actions

        return tensordict

    def update(self, dqn_replay_buffer, steps: int):
        metrics = {}

        data = dqn_replay_buffer.sample()

        obs = data["observation"]
        actions = data["action"]
        rewards = data["next"]["reward"]
        nxtobs = data["next"]["observation"]
        dones = data["next"]["done"].float()

        # ------------------------------------------------------------ #
        # - TODO: Implement DQN temporal-difference update
        # - TODO: Implement update of target networks
        # - TODO: Implement monitoring of loss values and average q values
        # ------------------------------------------------------------ #

        # ------------------------------------------------------------ #
        # - Implement DQN temporal-difference update
        # ------------------------------------------------------------ #
        
        # 1. PREDICTION: Get Q(s, a) for the specific actions that were taken in the buffer.
        # Ensure actions tensor is shaped [Batch, 1] for .gather()
        actions_idx = actions.argmax(dim=-1, keepdim=True)
        current_q = self.q_net(obs).gather(dim=1, index=actions_idx)

        # 2. TARGET: Calculate 1-step lookahead ground truth using frozen Target Net
        with torch.no_grad():
            max_next_q, _ = self.target_net(nxtobs).max(dim=-1, keepdim=True)
            td_target = rewards + (self.cfg.loss.gamma * (1.0 - dones) * max_next_q)

        # 3. LOSS & BACKPROPAGATION
        loss = F.mse_loss(current_q, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------------------------------------------------ #
        # - Implement update of target networks (Polyak Averaging)
        # ------------------------------------------------------------ #
        tau = self.cfg.optim.tau
        with torch.no_grad():
            for online_param, target_param in zip(self.q_net.parameters(), self.target_net.parameters()):
                target_param.data.lerp_(online_param.data, tau)

        metrics["epsilon"] = self._eps
        metrics["avg_dqn_loss"] = loss.item()
        metrics["avg_q_values"] = current_q.mean().item()

        return metrics
