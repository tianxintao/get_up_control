import torch
import numpy as np
import os
import logging
from dm_control.mujoco.wrapper import mjbindings
from scipy import interpolate
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
        

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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
            
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
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


def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path

class RLLogger(object):

    def __init__(self, output_path):
        self.logger = logging.getLogger()
        self.logger.handlers.clear()
        logger_name = os.path.join(output_path, 'session.log')
        file_handler = logging.FileHandler(logger_name)
        console_handler = logging.StreamHandler()
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))

    def info(self, string):
        self.logger.info(string)

    def log_start(self, argv, args):
        self.logger.info(str(argv))
        self.logger.info("---------------------------------------")
        self.logger.info(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
        self.logger.info("---------------------------------------")

    def log_train_episode(self, t, episode_num, episode_timesteps, episode_reward, loss_dict, env, args):
        console_output = "Total T: {t} Episode Num: {episode_num} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}" \
            .format(t=t + 1, episode_num=episode_num + 1, episode_timesteps=episode_timesteps,
                    episode_reward=episode_reward)
        if (t >= args.start_timesteps):
            console_output += "|A_Loss: {:.3f}".format(np.array(loss_dict["actor"]).mean())
            console_output += "|C_Loss: {:.3f}".format(np.array(loss_dict["critic"]).mean())
            console_output += "|T_Loss: {:.3f}".format(np.array(loss_dict["temperature"]).mean())
            console_output += "|T: {:.3f}".format(np.array(loss_dict["temperature_value"]).mean())
            if args.changing_velocity:
                console_output += "|Velocity: {:.3f}".format(env.speed)

        self.logger.info(console_output)

    def log_test(self, test_reward, min_test_reward, curriculum, power):
        self.logger.info("-------------------------------------------------")
        self.logger.info("Evaluation over 10 episodes: {:.3f}, minimum reward: {:.3f}, Curriculum: {}". \
                    format(test_reward, min_test_reward, curriculum))
        self.logger.info("-------------------------------------------------")
        self.logger.info("Current power: {:.3f}".format(power))

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


def interpolate_motion(data, interpolation_coeff=0.25):
    # data = np.load(trajectory_file)
    xquat = data["xquat"].reshape(data["xquat"].shape[0],-1,4).copy()
    extremities = data["extremities"].copy()
    com_dist = data["com"].copy()
    com_vel = data["com_vel"][:, 2].copy()
    qpos = data["qpos"].copy()
    qvel = data["qvel"].copy()
    first_index = np.argwhere(com_vel>0)[0][0]
    l = xquat.shape[0]

    timestamp_ori = np.linspace(0, l-first_index, num=l-first_index, endpoint=True)
    timestamp_new = interpolation_coeff * timestamp_ori

    for j in range(qpos.shape[1]):
        curve = interpolate.splrep(timestamp_ori, qpos[first_index:,j])
        qpos[first_index:,j] = interpolate.splev(timestamp_new, curve, der=0)

    for j in range(qvel.shape[1]):
        curve = interpolate.splrep(timestamp_ori, qvel[first_index:,j])
        qvel[first_index:,j] = interpolate.splev(timestamp_new, curve, der=0) * interpolation_coeff

    for j in range(com_dist.shape[1]):
        curve = interpolate.splrep(timestamp_ori, com_dist[first_index:,j])
        com_dist[first_index:,j] = interpolate.splev(timestamp_new, curve, der=0)
    
    for j in range(extremities.shape[1]):
        curve = interpolate.splrep(timestamp_ori, extremities[first_index:,j])
        extremities[first_index:,j] = interpolate.splev(timestamp_new, curve, der=0)

    for j in range(xquat.shape[1]):
        rotations = xquat[first_index:,j,:]
        rotations = R.from_quat(rotations[:,[1,2,3,0]])
        slerp = Slerp(timestamp_ori, rotations)
        # timestamp_new[0] = 1.0
        interp_rots = slerp(timestamp_new).as_quat()
        interp_rots[first_index:,[0,1,2,3]] = interp_rots[first_index:,[3,0,1,2]]
        xquat[first_index:,j,:] = interp_rots

    interpolated_data = {
        "xquat": xquat.reshape(xquat.shape[0],-1),
        "extremities": extremities,
        "com": com_dist,
        "qpos": qpos,
        "qvel": qvel
    }

    # np.savez('./data/trajectory3d_interpolated.npz', **interpolated_data)
    return interpolated_data






if __name__ == "__main__":
    interpolate_motion('./data/trajectory3d_52_collected.npz')
