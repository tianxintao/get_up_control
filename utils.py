import json
import logging
import math
import os

import numpy as np
import torch
from scipy import interpolate
from scipy.spatial.transform import Rotation as R


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
        print("Got here")
        data = {
            'state': self.state,
            'action': self.action,
            'next_state': self.next_state,
            'reward': self.reward,
            'not_done': self.not_done,
            'ptr': self.ptr,
            'size': self.size
        }
        np.savez(filename, **data)

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
        self.logger.info(f"Env: {args.env}, Seed: {args.seed}, imitation: {args.teacher_student}")
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
            if args.teacher_student:
                console_output += "|Velocity: {:.3f}".format(env.speed)

        self.logger.info(console_output)

    def log_episode_collection(self, t, episode_num, episode_timesteps, episode_reward, env):
        console_output = "Total T: {t} Episode Num: {episode_num} Episode T: {episode_timesteps} Reward: {episode_reward:.3f} | Velocity: {speed:.3f}" \
            .format(t=t + 1, episode_num=episode_num + 1, episode_timesteps=episode_timesteps,
                    episode_reward=episode_reward, speed=env.speed)
        self.logger.info(console_output)

    def log_test(self, test_reward, min_test_reward, curriculum, power):
        self.logger.info("-------------------------------------------------")
        self.logger.info("Evaluation over 10 episodes: {:.3f}, minimum reward: {:.3f}, Curriculum: {}". \
                         format(test_reward, min_test_reward, curriculum))
        self.logger.info("-------------------------------------------------")
        self.logger.info("Current power: {:.3f}".format(power))

    def log_data_collection(self, iteration):
        self.logger.info("-------------------------------------------------")
        self.logger.info("Starting #{} data collection procedure". \
                         format(iteration))
        self.logger.info("-------------------------------------------------")

    def log_policy_training(self, iteration):
        self.logger.info("-------------------------------------------------")
        self.logger.info("Starting #{} policy training procedure". \
                         format(iteration))
        self.logger.info("-------------------------------------------------")


def randomize_limited_and_rotational_joints(physics, k=0.1):
    random = np.random

    hinge = 3
    slide = 2
    ball = 1
    free = 0

    qpos = physics.named.data.qpos

    for joint_id in range(physics.model.njnt):
        joint_name = physics.model.id2name(joint_id, 'joint')
        joint_type = physics.model.jnt_type[joint_id]
        is_limited = physics.model.jnt_limited[joint_id]
        range_min, range_max = physics.model.jnt_range[joint_id]

        if is_limited:
            if joint_type == hinge or joint_type == slide:
                qpos[joint_name] = random.uniform(k * range_min, k * range_max)


def quaternion_multiply(quaternion1, quaternion2):
    a1, b1, c1, d1 = quaternion1
    a2, b2, c2, d2 = quaternion2
    return [a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2,
            a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
            a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2,
            a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2]


def rotmatrix_to_quat(mat):
    return R.from_matrix(mat).as_quat()


def rotmatrix_to_angleaxis(mat):
    ep = 1e-12
    s = R.from_matrix(mat).as_rotvec()
    degrees = np.linalg.norm(s)
    return np.array([degrees * 180 / np.pi, s[0] / (degrees + ep), s[1] / (degrees + ep), s[2] / (degrees + ep)])

def organize_args(args):
    args.curr_power = args.power
    if args.teacher_student:
        args.power = 1.0
        args.power_end = 1.0
        with open(os.path.join(args.teacher_dir, 'args.json'), 'r') as f:
            teacher_args = json.load(f)
        args.teacher_power = teacher_args["curr_power"]
        if args.test_policy:
            args.fast_speed = args.target_speed
            args.slow_speed = args.target_speed
    return args

def interpolate_motion(data, interpolation_coeff=0.25, uneven=False):
    com_dist = data["com"].copy()
    qpos = data["qpos"].copy()
    qvel = data["qvel"].copy()
    com_ori = data["com_ori"].copy()
    l = com_dist.shape[0]

    timestamp_ori = np.linspace(0, l, num=l, endpoint=True)
    if uneven:
        cut_off = l - 12
        timestamp_part1 = np.linspace(0, cut_off, num=math.ceil(cut_off / 0.05), endpoint=False)
        timestamp_part2 = np.linspace(cut_off, l, num=math.ceil((l - cut_off) / interpolation_coeff), endpoint=True)
        timestamp_new = np.concatenate((timestamp_part1, timestamp_part2))
    else:
        timestamp_new = np.linspace(0, l, num=math.ceil(l / interpolation_coeff), endpoint=True)

    com_ori_new = []
    for j in range(com_ori.shape[1]):
        curve = interpolate.splrep(timestamp_ori, com_ori[:, j])
        com_ori_new.append(interpolate.splev(timestamp_new, curve, der=0)[:, None])

    qpos_new = []
    for j in range(qpos.shape[1]):
        curve = interpolate.splrep(timestamp_ori, qpos[:, j])
        qpos_new.append(interpolate.splev(timestamp_new, curve, der=0)[:, None])

    qvel_new = []
    for j in range(qvel.shape[1]):
        curve = interpolate.splrep(timestamp_ori, qvel[:, j])
        qvel_new.append(interpolate.splev(timestamp_new, curve, der=0)[:, None] * interpolation_coeff)

    com_new = []
    for j in range(com_dist.shape[1]):
        curve = interpolate.splrep(timestamp_ori, com_dist[:, j])
        com_new.append(interpolate.splev(timestamp_new, curve, der=0)[:, None])

    interpolated_data = {
        "com": np.concatenate(com_new, axis=1),
        "qpos": np.concatenate(qpos_new, axis=1),
        "qvel": np.concatenate(qvel_new, axis=1),
        "com_ori": np.concatenate(com_ori_new, axis=1)
    }

    return interpolated_data


if __name__ == "__main__":
    print("done")
