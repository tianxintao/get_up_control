import torch
import numpy as np
import scipy.signal
import os
import logging
import imageio
from dm_control.mujoco.wrapper import mjbindings
        

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, args, max_size=int(1e6)):
        self.max_size = max_size
        self.args = args
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        if self.args.add_terrain:
            self.terrain = np.zeros((max_size, self.args.heightfield_dim, self.args.heightfield_dim))
            self.next_terrain = np.zeros_like(self.terrain)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state["scalar"]
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state["scalar"]
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        if self.args.add_terrain:
            self.terrain[self.ptr] = state["terrain"]
            self.next_terrain[self.ptr] = state["terrain"]

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, curriculum):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.FloatTensor(self.terrain[ind]).unsqueeze(1).to(self.device)\
                 if self.args.add_terrain else None,
            torch.FloatTensor(self.next_terrain[ind]).unsqueeze(1).to(self.device)\
                 if self.args.add_terrain else None,
        )
    
    def save(self, filename):
        data = {
            'state': self.state,
            'action': self.action,
            'next_state': self.next_state,
            'reward': self.reward,
            'not_done': self.not_done,
            'ptr': self.ptr,
            'size': self.size
        }
        np.savez(filename,**data)

    def load(self, filename):
        # with np.load(filename) as data:
        data = np.load(filename)
        self.state = data['state']
        self.action = data['action']
        self.next_state = data['next_state']
        self.reward = data['reward']
        self.not_done = data['not_done']
        self.ptr = data['ptr']
        self.size = data['size']
        

class PGBuffer:
    """
    A buffer for storing trajectories experienced by a Policy Gradient agent interacting
    with the environment.  Even though this is on-policy, we collect several episodes in a
    row our policy so we can batch process them more efficiently and take better gradient steps.
    """

    def __init__(self, obs_dim, act_dim, discrete, size, args, device):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        if discrete:
            self.act_buf = np.zeros((size,), dtype=np.float32)
        else:
            self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = args.gamma, args.lam
        self.psi_mode = args.psi_mode
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.device = device

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0

        psi = self.adv_buf


        # normalize psi.  this really helps.  you can try without
        psi_mean, psi_std = np.mean(psi), np.std(psi)
        psi = (psi - psi_mean) / (psi_std + 1e-5)

        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf, psi=psi, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32).to(self.device) for k, v in data.items()}

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def create_logger(output_path):
    logger = logging.getLogger()
    logger.handlers.clear()
    logger_name = os.path.join(output_path, 'session.log')
    file_handler = logging.FileHandler(logger_name)
    console_handler = logging.StreamHandler()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    return logger

def randomize_limited_and_rotational_joints(physics, k=0.1):
    random = np.random

    hinge = mjbindings.enums.mjtJoint.mjJNT_HINGE
    slide = mjbindings.enums.mjtJoint.mjJNT_SLIDE
    ball = mjbindings.enums.mjtJoint.mjJNT_BALL
    free = mjbindings.enums.mjtJoint.mjJNT_FREE

    qpos = physics.named.data.qpos

    for joint_id in range(physics.model.njnt):
        joint_name = physics.model.id2name(joint_id, 'joint')
        joint_type = physics.model.jnt_type[joint_id]
        is_limited = physics.model.jnt_limited[joint_id]
        range_min, range_max = physics.model.jnt_range[joint_id]

        if is_limited:
            if joint_type == hinge or joint_type == slide:
                qpos[joint_name] = random.uniform(k * range_min, k * range_max)
