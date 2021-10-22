import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi, power, original):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    scalar = 1.0 if original else power[:,None]
    mu = torch.tanh(mu) * scalar # broadcast power along the second dimension for multiplication
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
    def __init__(self, state_dim, action_dim, log_std_min, log_std_max, original, encoder):
        super().__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.encoder = encoder
        self.original = original

        self.trunk = nn.Sequential(
            nn.Linear(state_dim, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, 2 * action_dim)
        )

        # self.outputs = dict()
        # self.apply(weight_init)

    def forward(self, state, compute_pi=True, compute_log_pi=True, terrain=None):

        power = state[:,-1]
        if terrain is not None:
            terrain_encoded = self.encoder(terrain, detach=True)
            state = torch.cat([state, terrain_encoded], dim=1)
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

        mu, pi, log_pi = squash(mu, pi, log_pi, power, self.original)

        return mu, pi, log_pi, log_std

class ContactPredictor(nn.Module):
    
    def __init__(self, state_dim, action_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def predict_force(self, state, action, detach):
        state_action = torch.cat([state, action], dim=1)
        logits = self.layers(state_action)
		
        if detach:
            f1 = logits.detach()

        return logits
    
class Critic(nn.Module):
    """Critic network, employes two q-functions."""
    def __init__(self, state_dim, action_dim, encoder):
        super().__init__()
        self.encoder = encoder
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

        # self.apply(weight_init)

    def forward(self, state, action, terrain=None, detach=False):
        # detach_encoder allows to stop gradient propogation to encoder
        if terrain is not None:
            terrain_encoded = self.encoder(terrain, detach=detach)
            state = torch.cat([state, terrain_encoded], dim=1)
        state_action = torch.cat([state, action], dim=1)
        q1 = self.Q1(state_action)
        q2 = self.Q2(state_action)

        return q1, q2

# class Critic(nn.Module):
#     """Critic network, employes two q-functions."""
#     def __init__(self, state_dim, action_dim, encoder):
#         super().__init__()
#         self.encoder = encoder
#         self.feature1 = nn.Sequential(
#             nn.Linear(state_dim + action_dim, 1024), nn.ReLU(),
#             nn.Linear(1024, 1024), nn.ReLU()
#         )
#         self.ln1 = nn.Linear(1024, 1)
#
#         self.feature2 = nn.Sequential(
#             nn.Linear(state_dim + action_dim, 1024), nn.ReLU(),
#             nn.Linear(1024, 1024), nn.ReLU(),
#         )
#         self.ln2 = nn.Linear(1024, 1)
#
#         self.apply(weight_init)
#
#     # def predict_force(self, state, action, detach):
#     #     state_action = torch.cat([state, action], dim=1)
#     #     f1 = self.f1(self.feature1(state_action))
#     #     f2 = self.f2(self.feature2(state_action))
#
#     #     if detach:
#     #         f1 = f1.detach()
#     #         f2 = f2.detach()
#
#     #     return f1, f2
#
#     def forward(self, state, action, terrain=None, detach=False):
#         # detach_encoder allows to stop gradient propogation to encoder
#         if terrain is not None:
#             terrain_encoded = self.encoder(terrain, detach=detach)
#             state = torch.cat([state, terrain_encoded], dim=1)
#         state_action = torch.cat([state, action], dim=1)
#         q1 = self.ln1(self.feature1(state_action))
#         q2 = self.ln2(self.feature2(state_action))
#
#         return q1, q2


class Encoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim):
        super().__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(1, 4, 3, stride=1), nn.ReLU(),
            nn.Conv2d(4, 4, 3, stride=1), nn.ReLU(),
        )

        # Debug needed
        output_size = self.convs(torch.randn(1,1,obs_shape,obs_shape)).shape.numel()

        self.ln = nn.Sequential(
            nn.Linear(output_size, 16), nn.ReLU(),
            nn.Linear(16, feature_dim)
        )

        self.apply(weight_init)

    def forward(self, obs, detach=False):
        h = self.convs(obs)
        h = h.view(h.size(0), -1)

        if detach:
            h = h.detach()

        return self.ln(h)


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
        self.encoder = None
        self.reward_avg = 0.0
        self.avg_force_loss = 0.0
        self.init_temperature = init_temperature
        self.alpha_lr = alpha_lr
        self.alpha_beta = alpha_beta

        if self.args.add_terrain:
            state_dim = state_dim + self.args.terrain_dim
            self.encoder = Encoder(self.args.heightfield_dim, self.args.terrain_dim).to(device)

        self.actor = Actor(state_dim, action_dim, actor_log_std_min, actor_log_std_max, args.original, self.encoder).to(device)

        self.critic = Critic(state_dim, action_dim, self.encoder).to(device)

        if self.args.predict_force:
            self.force_network = ContactPredictor(state_dim, action_dim, self.args.force_dim).to(device)
            self.force_optimizer = torch.optim.Adam(
                self.force_network.parameters(), lr=1e-4
            )

        self.critic_target = copy.deepcopy(self.critic)

        # self.critic_target.load_state_dict(self.critic.state_dict())

        # tie encoders between actor and critic, and CURL and critic
        # self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        
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

        if self.args.scheduler:
            milestones = [100000,600000,1000000,1600000]
            self.actor_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.actor_optimizer,
                milestones=milestones,
                gamma=0.5
            )
            self.critic_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.critic_optimizer,
                milestones=milestones,
                gamma=0.5
            )

        if self.args.debug:
            self.reset_record()
    
    def reset_alpha(self):
        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=self.alpha_lr, betas=(self.alpha_beta, 0.999)
        )

    def reset_record(self):
        self.critic_loss = []
        self.actor_loss = []
        self.temperature_loss = []
        self.temperature = []
        self.grad = []
        self.reaction_force_loss = []

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, terrain=None):
        if self.args.add_terrain: self.encoder.eval()
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            if terrain is not None:
                terrain = torch.FloatTensor(terrain).unsqueeze(0).unsqueeze(0).to(device)
            mu, _, _, _ = self.actor(
                state, compute_pi=False, compute_log_pi=False, terrain=terrain
            )
            if self.args.add_terrain: self.encoder.train()
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, state, terrain=None):
        if self.args.add_terrain: self.encoder.eval()
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            if terrain is not None:
                terrain = torch.FloatTensor(terrain).unsqueeze(0).unsqueeze(0).to(device)
            mu, pi, _, _ = self.actor(state, compute_log_pi=False, terrain=terrain)
            if self.args.add_terrain: self.encoder.train()
            return pi.cpu().data.numpy().flatten()
    
    # def policy_cloning(self, replay_buffer, batch_size=100):

    #     for _ in range(5):
    #         state, action, next_state, reward, not_done, terrain, next_terrain, reaction_force = replay_buffer.sample(batch_size)

    #         selected_action, sampled_action, _, _ = self.actor(next_state, terrain=next_terrain)
    #         actor_loss = F.mse_loss(sampled_action, action)
    #         self.actor_optimizer.zero_grad()
    #         actor_loss.backward()
    #         self.actor_optimizer.step()

    #         if self.args.debug:
    #             self.actor_loss.append(actor_loss.item())

    def train(self, replay_buffer, curriculum_finished, batch_size=100):
        self.total_it += 1

        state, action, next_state, reward, not_done, terrain, next_terrain, reaction_force, behavior_action = replay_buffer.sample(batch_size)

        if self.args.predict_force:
            predicted_f = self.force_network.predict_force(state, action, detach=False)
            force_loss = F.mse_loss(predicted_f, reaction_force)
            # weights = torch.tensor([4.0, 4.0, 1.0, 1.0], device=torch.device('cuda:0'))
            # force_loss = (((predicted_f - reaction_force) ** 2) * weights).sum(axis=1).mean()
            self.force_optimizer.zero_grad()
            force_loss.backward()
            self.force_optimizer.step()
            self.avg_force_loss = self.avg_force_loss * 0.99 + force_loss.item() * 0.01

        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_state, terrain=next_terrain)
            target_Q1, target_Q2 = self.critic_target(next_state, policy_action, terrain=next_terrain)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)
            self.reward_avg = self.reward_avg * 0.99 + reward.mean() * 0.01
            if self.args.predict_force:
                target_Q += (force_loss.detach()/self.avg_force_loss) * self.reward_avg * 0.2

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action, terrain=terrain)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # if self.args.debug:
        #     for param in self.critic.parameters():
        #         self.grad.append(torch.norm(param.grad.detach().cpu().flatten()))
        self.critic_optimizer.step()


        if self.total_it % self.policy_freq == 0:
            _, pi, log_pi, log_std = self.actor(state, terrain=terrain)
            actor_Q1, actor_Q2 = self.critic(state, pi, terrain=terrain, detach=True)

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
            # print(self.alpha)

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        if self.args.scheduler and curriculum_finished:
            self.actor_scheduler.step()
            self.critic_scheduler.step()
            # self.log_alpha_scheduler.step()

        if self.args.debug:
            self.critic_loss.append(critic_loss.item())
            if self.total_it % self.policy_freq == 0: 
                self.actor_loss.append(actor_loss.item()) 
                self.temperature_loss.append(alpha_loss.item())
            if self.args.predict_force:
                self.reaction_force_loss.append(force_loss.item())
            self.temperature.append(self.alpha.item())
            

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic.pt")
        torch.save(self.critic_target.state_dict(), filename + "_critic_target.pt")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer.pt")

        torch.save(self.actor.state_dict(), filename + "_actor.pt")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer.pt")

        torch.save(self.log_alpha, filename + "_log_alpha.pt")
        torch.save(self.log_alpha_optimizer.state_dict(), filename + "_log_alpha_optimizer.pt")

        if self.args.predict_force:
            torch.save(self.force_network.state_dict(), filename + "_force.pt")
            torch.save(self.force_optimizer.state_dict(), filename + "_force_optimizer.pt")

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
        
 