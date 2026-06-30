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

        # ------------------------------------------------------------ #
        # Q-network: 2-layer MLP mapping obs → Q-values per action
        self.q_network = torch.nn.Sequential(
            torch.nn.Linear(self.net_input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, self.net_output_dim),
        )

        # Target network: frozen deep copy of Q-network, updated via Polyak averaging
        self.target_network = deepcopy(self.q_network)
        for param in self.target_network.parameters():
            param.requires_grad = False

        # Adam optimizer for the online Q-network only
        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(), lr=self.cfg.optim.lr
        )
        # ------------------------------------------------------------ #

        self._eps = self.cfg.exploration.eps_start_value

    def forward(self, tensordict: TensorDictBase):

        mode = exploration_type()
        obs = tensordict["observation"]

        # ------------------------------------------------------------ #
        # Compute Q-values from the online network (no gradient needed here)
        with torch.no_grad():
            q_values = self.q_network(obs)

        greedy_action = q_values.argmax(dim=-1)  # shape: () or (B,)

        if mode == ExplorationType.DETERMINISTIC or mode == ExplorationType.MEAN:
            # Pure greedy action selection during evaluation
            action = greedy_action
        else:
            # Epsilon-greedy during training.
            # Works for both scalar obs (shape (obs_dim,)) and batched obs (shape (B, obs_dim))
            # by deriving the sample shape from greedy_action.
            mask = torch.rand(greedy_action.shape, device=obs.device) < self._eps
            rand_actions = torch.randint(
                self.net_output_dim, greedy_action.shape, device=obs.device
            )
            action = torch.where(mask, rand_actions, greedy_action)

        # TorchRL's GymEnv expects discrete actions as one-hot vectors for GymEnv.
        tensordict["action"] = F.one_hot(action, num_classes=self.net_output_dim)
        # ------------------------------------------------------------ #

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
        # Current Q-values: Q(s, a) for each action taken in the batch
        q_values = self.q_network(obs)                                    # (B, num_actions)

        # TorchRL's GymEnv stores discrete actions as one-hot vectors (B, num_actions).
        # Convert to integer indices (B, 1) before gathering.
        if actions.dim() > 1 and actions.shape[-1] == self.net_output_dim:
            action_indices = actions.argmax(dim=-1, keepdim=True).long()  # one-hot → index
        else:
            action_indices = actions.long().view(-1, 1)                   # already integer

        q_values_selected = q_values.gather(1, action_indices).view(-1)  # (B,)

        # Temporal-difference target using the frozen target network
        # y = r + γ * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            target_q_values = self.target_network(nxtobs)                 # (B, num_actions)
            max_target_q = target_q_values.max(dim=-1).values            # (B,)
            td_target = (
                rewards.view(-1)
                + self.cfg.loss.gamma * max_target_q * (1.0 - dones.view(-1))
            )                                                             # (B,)

        # MSE loss between predicted Q-values and TD targets
        loss = F.mse_loss(q_values_selected, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update target network via Polyak averaging: θ⁻ ← τθ + (1−τ)θ⁻
        tau = self.cfg.optim.tau
        for param, target_param in zip(
            self.q_network.parameters(), self.target_network.parameters()
        ):
            target_param.data.copy_(
                tau * param.data + (1.0 - tau) * target_param.data
            )

        # Linear epsilon decay from eps_start to eps_end over eps_anneal_steps env steps
        eps_start = self.cfg.exploration.eps_start_value
        eps_end = self.cfg.exploration.eps_end_value
        eps_anneal = self.cfg.exploration.eps_anneal_steps
        self._eps = max(
            eps_end,
            eps_start - (eps_start - eps_end) * steps / eps_anneal
        )
        # ------------------------------------------------------------ #

        metrics["epsilon"] = self._eps
        metrics["avg_dqn_loss"] = loss.item()
        metrics["avg_q_values"] = q_values_selected.mean().item()

        return metrics