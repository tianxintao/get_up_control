import numpy as np
import torch
import copy
from torch.nn import Module, Linear
from torch.distributions import Distribution, Normal
from torch.nn.functional import relu, logsigmoid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_STD_MIN_MAX = (-20, 2)

def quantile_huber_loss_f(quantiles, samples):
    pairwise_delta = samples[:, None, None, :] - quantiles[:, :, :, None]  # batch x nets x quantiles x samples
    abs_pairwise_delta = torch.abs(pairwise_delta)
    huber_loss = torch.where(abs_pairwise_delta > 1,
                                abs_pairwise_delta - 0.5,
                                pairwise_delta ** 2 * 0.5)

    n_quantiles = quantiles.shape[2]
    tau = torch.arange(n_quantiles, device=device).float() / n_quantiles + 1 / 2 / n_quantiles
    loss = (torch.abs(tau[None, None, :, None] - (pairwise_delta < 0).float()) * huber_loss).mean()
    return loss

class Mlp(Module):
    def __init__(
            self,
            input_size,
            hidden_sizes,
            output_size
    ):
        super().__init__()
        # TODO: initialization
        self.fcs = []
        in_size = input_size
        for i, next_size in enumerate(hidden_sizes):
            fc = Linear(in_size, next_size)
            self.add_module(f'fc{i}', fc)
            self.fcs.append(fc)
            in_size = next_size
        self.last_fc = Linear(in_size, output_size)

    def forward(self, input):
        h = input
        for fc in self.fcs:
            h = relu(fc(h))
        output = self.last_fc(h)
        return output


class Critic(Module):
    def __init__(self, state_dim, action_dim, n_quantiles, n_nets):
        super().__init__()
        self.nets = []
        self.n_quantiles = n_quantiles
        self.n_nets = n_nets
        for i in range(n_nets):
            net = Mlp(state_dim + action_dim, [512, 512, 512], n_quantiles)
            self.add_module(f'qf{i}', net)
            self.nets.append(net)

    def forward(self, state, action):
        sa = torch.cat((state, action), dim=1)
        quantiles = torch.stack(tuple(net(sa) for net in self.nets), dim=1)
        return quantiles


class Actor(Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.action_dim = action_dim
        self.net = Mlp(state_dim, [256, 256], 2 * action_dim)

    def forward(self, obs):
        mean, log_std = self.net(obs).split([self.action_dim, self.action_dim], dim=1)
        log_std = log_std.clamp(*LOG_STD_MIN_MAX)

        if self.training:
            std = torch.exp(log_std)
            tanh_normal = TanhNormal(mean, std)
            action, pre_tanh = tanh_normal.rsample()
            log_prob = tanh_normal.log_prob(pre_tanh)
            log_prob = log_prob.sum(dim=1, keepdim=True)
        else:  # deterministic eval without log_prob computation
            action = torch.tanh(mean)
            log_prob = None
        return action, log_prob


class TanhNormal(Distribution):
    def __init__(self, normal_mean, normal_std):
        super().__init__()
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.standard_normal = Normal(torch.zeros_like(self.normal_mean, device=device),
                                        torch.ones_like(self.normal_std, device=device))
        self.normal = Normal(normal_mean, normal_std)

    def log_prob(self, pre_tanh):
        log_det = 2 * np.log(2) + logsigmoid(2 * pre_tanh) + logsigmoid(-2 * pre_tanh)
        result = self.normal.log_prob(pre_tanh) - log_det
        return result

    def rsample(self):
        pretanh = self.normal_mean + self.normal_std * self.standard_normal.sample()
        return torch.tanh(pretanh), pretanh


class TQC(Module):
    def __init__(
        self,
		state_dim,
        action_dim,
		discount,
		tau,
		top_quantiles_to_drop,
        args
    ):
        super().__init__()
        self.args = args
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim, action_dim, args.n_quantiles, args.n_nets).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.log_alpha = torch.zeros((1,), requires_grad=True, device=device)

        # TODO: check hyperparams
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-5)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=1e-4)

        self.discount = discount
        self.tau = tau
        self.top_quantiles_to_drop = top_quantiles_to_drop
        self.target_entropy = -np.prod(action_dim)

        self.quantiles_total = self.critic.n_quantiles * self.critic.n_nets

        self.total_it = 0
        self.reset_record()

    def reset_record(self):
        self.critic_loss = []
        self.actor_loss = []
        self.temperature_loss = []
        self.temperature = []

    def train(self, replay_buffer, curriculum_finished, batch_size=256):
        state, action, next_state, reward, not_done, terrain, next_terrain, reaction_force, behavior_action = replay_buffer.sample(batch_size)
        alpha = torch.exp(self.log_alpha)

        # --- Q loss ---
        with torch.no_grad():
            # get policy action
            new_next_action, next_log_pi = self.actor(next_state)

            # compute and cut quantiles at the next state
            next_z = self.critic_target(next_state, new_next_action)  # batch x nets x quantiles
            sorted_z, _ = torch.sort(next_z.reshape(batch_size, -1))
            sorted_z_part = sorted_z[:, :self.quantiles_total-self.top_quantiles_to_drop]

            # compute target
            target = reward + not_done * self.discount * (sorted_z_part - alpha * next_log_pi)

        cur_z = self.critic(state, action)
        critic_loss = quantile_huber_loss_f(cur_z, target)

        # --- Update ---
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Policy and alpha loss ---
        new_action, log_pi = self.actor(state)
        alpha_loss = -self.log_alpha * (log_pi + self.target_entropy).detach().mean()
        actor_loss = (alpha * log_pi - self.critic(state, new_action).mean(2).mean(1, keepdim=True)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.total_it += 1

        if self.args.debug:
            self.critic_loss.append(critic_loss.item())
            self.actor_loss.append(actor_loss.item()) 
            self.temperature_loss.append(alpha_loss.item())
            self.temperature.append(self.log_alpha.exp().item())

    def select_action(self, state, terrain=None):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            if terrain is not None:
                terrain = torch.FloatTensor(terrain).unsqueeze(0).unsqueeze(0).to(device)
            mu = self.actor(state)[0]
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, state, terrain=None):
        return self.select_action(state, terrain=terrain)

    def save(self, filename):
        filename = str(filename)
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_target.state_dict(), filename + "_critic_target")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
        torch.save(self.log_alpha, filename + '_log_alpha')
        torch.save(self.alpha_optimizer.state_dict(), filename + "_alpha_optimizer")

    def load(self, filename):
        filename = str(filename)
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_target.load_state_dict(torch.load(filename + "_critic_target"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.log_alpha = torch.load(filename + '_log_alpha')
        self.alpha_optimizer.load_state_dict(torch.load(filename + "_alpha_optimizer"))