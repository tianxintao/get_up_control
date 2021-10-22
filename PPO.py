import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Network(nn.Module):
    """Network definition to be used for actor and critic networks"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # NOTE: feel free to experiment with this network
        self.linin = nn.Linear(in_dim, 300)
        self.hidden1 = nn.Linear(300, 300)
        self.linout = nn.Linear(300, out_dim)

        # initialize weights and bias to 0 in the last layer.
        # this ensures the actors starts out completely random in the beginning, and that the value function starts at 0
        # this can help training.  you can experiment with turning it off.
        self.linout.bias.data.fill_(0.0)
        self.linout.weight.data.fill_(0.0)

    def forward(self, inputs):
        """
        Args:
            inputs (torch.Tensor):  (BS, in_dim)
        Returns:
            torch.Tensor:  (BS, out_dim)
        """
        x = self.linin(inputs)
        x = torch.relu(x)
        x = torch.relu(self.hidden1(x))
        x = self.linout(x)
        return x


# NOTE: polcy gradient methods can handle discrete or continuous actions.
# we include definitions for both cases below.

class DiscreteActor(nn.Module):
    """Actor network that chooses 1 discrete action by sampling from a Categorical distribution of N actions"""
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.logits_net = Network(obs_dim, act_dim)

    def forward(self, obs, taken_act=None):
        logits = self.logits_net(obs)
        pi = Categorical(logits=logits)
        logp_a = None
        if taken_act is not None:
            logp_a = pi.log_prob(taken_act)
        return pi, logp_a

class GaussianActor(nn.Module):
    """Actor network that chooses N continuous actions by sampling from N parameterized independent Normal distributions"""
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.mu_net = Network(obs_dim, act_dim)
        # make the std learnable, but not dependent on the current observation
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

    def forward(self, obs, taken_act=None):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        pi = Normal(mu, std)
        logp_a = None
        if taken_act is not None:
            logp_a = pi.log_prob(taken_act).sum(axis=-1)
        return pi, logp_a

class PPO(nn.Module):
    """
    Object to hold Actor and Critic network objects

    See Sutton book (http://www.incompleteideas.net/book/RLbook2018.pdf) Chapter 13 for discussion of Actor Critic methods.
    Basically they are just policy gradients methods where you also learn a value function and use that to aid in learning.
    Not all options in this class use a critic, for example psi_mode='future_return' just uses the rewards in a REINFORCE fashion.
    """
    def __init__(
            self,
            obs_dim,
            act_dim,
            discrete,
            target_kl=0.01,
            lr_pi=1e-4,
            lr_v = 1e-4,
            clip_ratio=0.1,
            train_pi_iters=80,
            train_v_iters=320
    ):
        super().__init__()
        # bulid actor networt
        self.discrete = discrete
        if self.discrete:
            self.pi = DiscreteActor(obs_dim, act_dim)
        else:
            self.pi = GaussianActor(obs_dim, act_dim)
        # build value function
        self.v  = Network(obs_dim, 1)

        self.pi_optimizer = torch.optim.Adam(self.pi.parameters(), lr=lr_pi)
        self.vf_optimizer = torch.optim.Adam(self.v.parameters(), lr=lr_v)
        self.clip_ratio = clip_ratio
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.target_kl = target_kl

        self.reset_record()
    
    def reset_record(self):
        self.pi_loss = []
        self.v_loss = []

    def sample_action(self, state, terrain=None):
        """Run a single forward step of the ActorCritic networks.  Used during rollouts, but not during optimization"""
        # no_grad, since we don't need to do any backprop while we collect data.
        # this means we will have to recompute forward passes later. (this is standard)
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        with torch.no_grad():
            pi, _ = self.pi(state)
            a = pi.sample()
            logp_a = pi.log_prob(a) if self.discrete else pi.log_prob(a).sum(axis=-1)
            v = self.v(state)
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()

    def select_action(self, state, terrain=None):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        pi, _ = self.pi(state)
        return pi.mean.cpu().detach().numpy()

    # Set up update function
    def update(self, buf):
        batch = buf.get()

        # Get loss and info values before update
        pi_l_old, pi_info_old = self.compute_loss_pi(batch)
        pi_l_old = pi_l_old.item()
        v_l_old = self.compute_loss_v(batch).item()

        # Policy learning
        for i in range(self.train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(batch)
            if pi_info["kl"] > self.target_kl:
                print("Early stopping")
                break
            loss_pi.backward()
            self.pi_loss.append(loss_pi.item())
            self.pi_optimizer.step()

        # Value function learning
        for i in range(self.train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v = self.compute_loss_v(batch)
            loss_v.backward()
            self.v_loss.append(loss_v.item())
            self.vf_optimizer.step()
        
        # # # Log changes from update
        # kl, ent = pi_info['kl'], pi_info_old['ent']
        # logs['kl'] += [kl]
        # logs['ent'] += [ent]
        # logs['loss_v'] += [loss_v.item()]
        # logs['loss_pi'] += [loss_pi.item()]

    # Set up function for computing policy loss
    def compute_loss_pi(self, batch):
        obs, act, psi, logp_old = batch['obs'], batch['act'], batch['psi'], batch['logp']
        pi, logp = self.pi(obs, act)

        # Policy loss
        g = psi.clone()
        index_positive = (g >= 0)
        g[index_positive] *= (1 + self.clip_ratio)
        g[~(index_positive)] *= (1 - self.clip_ratio)
        loss_pi = -torch.mean(torch.min(torch.exp(logp - logp_old) * psi, g))


        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(self, batch):
        obs, ret = batch['obs'], batch['ret']
        v = self.v(obs)
        loss_v = nn.functional.mse_loss(v.squeeze(-1), ret)
        return loss_v

    def save(self, filename):
        torch.save(self.v.state_dict(), filename + "_critic.pt")
        torch.save(self.vf_optimizer.state_dict(), filename + "_critic_optimizer.pt")

        torch.save(self.pi.state_dict(), filename + "_actor.pt")
        torch.save(self.pi_optimizer.state_dict(), filename + "_actor_optimizer.pt")


if __name__=="__main__":

    logits = torch.nn.Parameter(torch.tensor([-2.5, 0.9, 2.4, 3.7]))  # imagine these came from the output of the network
    c = Categorical(logits=logits)
    a_t = torch.tensor(2)  # imagine this came from c.sample()
    logp = c.log_prob(a_t)
    logp.backward()
    print(logits.grad)

    std = torch.exp(-0.5 * torch.as_tensor(np.ones(2, dtype=np.float32)))
    mu = torch.nn.Parameter(torch.tensor([0.0,1.2]))
    n = Normal(mu, std)
    a_t = torch.tensor([0.1,1.0])
    logp = n.log_prob(a_t).sum(axis=-1)
    logp.backward()
    print(mu.grad)