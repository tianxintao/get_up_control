import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rlkit_distributions import GaussianMixture

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StudentPolicy(nn.Module):
    """MLP actor network."""

    def __init__(self, state_dim, action_dim, args=None):
        super().__init__()
        self.args = args

        self.trunk = nn.Sequential(
            nn.Linear(state_dim, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, action_dim),
            # torch.nn.Tanh()
        ).to(device)

        # self.trunk = nn.ModuleList(
        #     [
        #         nn.Sequential(nn.Linear(state_dim, 255), nn.ReLU()),
        #         nn.Sequential(nn.Linear(255 + 1, 511), nn.ReLU()),
        #         # nn.Sequential(nn.Linear(510 + 2, 1022), nn.BatchNorm1d(1022), nn.ReLU()),
        #         # nn.Sequential(nn.Linear(1022 + 2, 510), nn.BatchNorm1d(510), nn.ReLU()),
        #         nn.Sequential(nn.Linear(511 + 1, 255), nn.ReLU()),
        #         nn.Sequential(nn.Linear(255 + 1, 63), nn.ReLU()),
        #         nn.Linear(63 + 1, action_dim)
        #     ]
        # ).to(device)

        self.optimizer = torch.optim.Adam(
            self.trunk.parameters(), lr=self.args.actor_lr, betas=(0.9, 0.999)
        )

        self.reset_record()

    def forward(self, state):
        return self.trunk(state)

    def select_action(self, state):
        self.trunk.eval()
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.forward(state).cpu().data.numpy().flatten()
        self.trunk.train()
        return action

    def reset_record(self):
        self.loss_dict = {
            "action": []
        }

    def train(self, distill_buffer, batch_size=100):
        state, action, next_state, reward, not_done = distill_buffer.sample(
            batch_size)

        predicted_action = self.forward(state)
        loss = F.mse_loss(action, predicted_action)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.loss_dict["action"].append(loss.item())

    def save(self, filename):
        torch.save(self.trunk.state_dict(), filename + "_policy.pt")
        torch.save(self.optimizer.state_dict(), filename + "_optimizer.pt")

    def load(self, filename, load_optimizer=False):
        self.trunk.load_state_dict(torch.load(filename + "_policy.pt"))
        if load_optimizer:
            self.optimizer.load_state_dict(torch.load(filename + "_optimizer.pt"))
