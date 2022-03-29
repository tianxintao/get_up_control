import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi, power):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    scalar = power[:, None]
    mu = torch.tanh(mu) * scalar  # broadcast power along the second dimension for multiplication
    if pi is not None:
        pi = torch.tanh(pi) * scalar
    if log_pi is not None:
        log_pi -= torch.log(scalar * (F.relu(1 - pi.pow(2)) + 1e-6)).sum(-1, keepdim=True)
    return mu, pi, log_pi


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class Actor(nn.Module):
    """MLP actor network."""

    def __init__(self, state_dim, action_dim, log_std_min, log_std_max):
        super().__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.trunk = nn.Sequential(
            nn.Linear(state_dim, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, 2 * action_dim)
        )

        self.outputs = dict()

    def forward(self, state, compute_pi=True, compute_log_pi=True):

        power = state[:, -1]

        mu, log_std = self.trunk(state).chunk(2, dim=-1)

        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
                self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi, power)

        return mu, pi, log_pi, log_std


class Critic(nn.Module):
    """Critic network, employes two q-functions."""

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.Q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, 1)
        )

        self.Q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, 1)
        )

    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=1)
        q1 = self.Q1(state_action)
        q2 = self.Q2(state_action)

        return q1, q2


class SAC(nn.Module):
    def __init__(
            self,
            state_dim,
            action_dim,
            discount=0.99,
            init_temperature=0.1,
            alpha_lr=1e-4,
            alpha_beta=0.9,
            actor_lr=1e-4,
            actor_beta=0.9,
            actor_log_std_min=-5,
            actor_log_std_max=2,
            policy_freq=1,
            critic_lr=1e-4,
            critic_beta=0.9,
            tau=0.005,
            critic_target_update_freq=2,
            args=None
    ):
        super().__init__()
        self.args = args
        self.discount = discount
        self.tau = tau
        self.policy_freq = policy_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.init_temperature = init_temperature
        self.alpha_lr = alpha_lr
        self.alpha_beta = alpha_beta

        self.actor = Actor(state_dim, action_dim, actor_log_std_min, actor_log_std_max).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_dim)

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.reset_alpha()
        self.total_it = 0
        self.reset_record()

    def reset_alpha(self):
        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=self.alpha_lr, betas=(self.alpha_beta, 0.999)
        )

    def reset_record(self):
        self.loss_dict = {
            "critic": [],
            "actor": [],
            "temperature": [],
            "temperature_value": []
        }

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            mu, _, _, _ = self.actor(
                state, compute_pi=False, compute_log_pi=False
            )
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, state, terrain=None):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            mu, pi, _, _ = self.actor(state, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=100):
        self.total_it += 1

        state, action, next_state, reward, not_done = replay_buffer.sample(
            batch_size)

        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_state)
            target_Q1, target_Q2 = self.critic_target(next_state, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            _, pi, log_pi, log_std = self.actor(state)
            actor_Q1, actor_Q2 = self.critic(state, pi)

            actor_Q = torch.min(actor_Q1, actor_Q2)
            actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

            # optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()

            alpha_loss.backward()
            self.log_alpha_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.loss_dict["critic"].append(critic_loss.item())
        if self.total_it % self.policy_freq == 0:
            self.loss_dict["actor"].append(actor_loss.item())
            self.loss_dict["temperature"].append(alpha_loss.item())
        self.loss_dict["temperature_value"].append(self.alpha.item())

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic.pt")
        torch.save(self.critic_target.state_dict(), filename + "_critic_target.pt")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer.pt")

        torch.save(self.actor.state_dict(), filename + "_actor.pt")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer.pt")

        torch.save(self.log_alpha, filename + "_log_alpha.pt")
        torch.save(self.log_alpha_optimizer.state_dict(), filename + "_log_alpha_optimizer.pt")

    def load(self, filename, load_optimizer=False):
        self.critic.load_state_dict(torch.load(filename + "_critic.pt"))
        self.critic_target.load_state_dict(torch.load(filename + "_critic_target.pt"))
        self.actor.load_state_dict(torch.load(filename + "_actor.pt"))
        self.log_alpha = torch.load(filename + "_log_alpha.pt")
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=1e-4, betas=(0.9, 0.999)
        )

        if load_optimizer:
            self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer.pt"))
            self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer.pt"))
            self.log_alpha_optimizer.load_state_dict(torch.load(filename + "_log_alpha_optimizer.pt"))
