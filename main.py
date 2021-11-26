import time
import imageio
import numpy as np
import cv2
import os
import utils
import json
import sys
import copy
import torch
import copy
from tensorboardX import SummaryWriter
from PPO import PPO
from SAC import SAC
from TQC import TQC
from torch.utils.tensorboard import SummaryWriter
from env import HumanoidStandupEnv, HumanoidStandingEnv, HumanoidStandupVelocityEnv, HumanoidStandupHybridEnv
from utils import RLLogger, ReplayBuffer
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ArgParserTrain(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument('--env', type=str, default='HumanoidStandup',
                          choices=['Cheetah', 'HumanoidStandup', 'HumanoidRandom', 'HumanoidHybrid', 'HumanoidBench',
                                   'HumanoidChair', 'HumanoidStanding', 'HumanoidStandupVelocityEnv',
                                   'MountainCarContinuous-v0'])
        self.add_argument("--policy", default="SAC", choices=['SAC', 'PPO', 'TQC'])
        self.add_argument('--debug', default=False, action='store_true')
        self.add_argument('--scheduler', default=False, action='store_true')
        self.add_argument('--imitation_reward', default=False, action='store_true')
        self.add_argument('--test_policy', default=False, action='store_true')
        self.add_argument('--teacher_student', default=False, action='store_true')
        self.add_argument('--changing_velocity', default=False, action='store_true')
        self.add_argument("--seed", default=0, type=int)
        self.add_argument("--power", default=1.0, type=float)
        self.add_argument("--power_end", default=0.4, type=float)
        self.add_argument("--teacher_power", default=0.4, type=float)
        self.add_argument('--max_timesteps', type=int, default=10000000, help='Number of simulation steps to run')
        self.add_argument('--test_interval', type=int, default=20000, help='Number of simulation steps between tests')
        self.add_argument('--test_iterations', type=int, default=10, help='Number of simulation steps between tests')
        self.add_argument('--replay_buffer_size', type=int, default=1e6, help='Number of simulation steps to run')
        self.add_argument('--gamma', type=float, default=0.999, help='discount factor')
        self.add_argument('--max_ep_len', type=int, default=1000)
        self.add_argument('--custom_reset', default=False, action='store_true')
        self.add_argument('--avg_reward', default=False, action='store_true')
        self.add_argument("--imitation_data", default='./data/imitation_data_sample.npz', type=str)
        self.add_argument("--work_dir", default='./experiment/')
        self.add_argument("--load_dir", default=None, type=str)
        self.add_argument("--standing_policy", default=None, type=str)
        self.add_argument("--standup_policy", default=None, type=str)

        # Terrain hyperparameters
        self.add_argument('--max_height', default=0.1, type=float)
        self.add_argument('--add_terrain', default=False, action='store_true')
        self.add_argument('--terrain_dim', default=6, type=int)
        self.add_argument('--heightfield_dim', default=9, type=int)

        # Force/contact hypeparameters
        self.add_argument("--predict_force", action="store_true")
        self.add_argument("--force_dim", type=int, default=1)

        # SAC hyperparameters
        self.add_argument("--batch_size", default=1024, type=int)
        self.add_argument("--discount", default=0.99, type=float)
        self.add_argument("--init_temperature", default=0.1, type=float)
        self.add_argument("--critic_target_update_freq", default=2, type=int)
        self.add_argument("--alpha_lr", default=1e-4, type=float)
        self.add_argument("--actor_lr", default=1e-4, type=float)
        self.add_argument("--critic_lr", default=1e-4, type=float)
        self.add_argument("--tau", default=0.005)
        self.add_argument("--start_timesteps", default=10000, type=int)

        self.add_argument('--render_interval', type=int, default=100, help='render every N')
        self.add_argument('--log_interval', type=int, default=100, help='log every N')
        self.add_argument("--tag", default="")


def env_function(args):
    if args.env == 'HumanoidStandup':
        if args.changing_velocity:
            return HumanoidStandupVelocityEnv
        return HumanoidStandupEnv
    elif args.env == "HumanoidHybrid":
        return HumanoidStandupHybridEnv
    elif args.env == "HumanoidStanding":
        return HumanoidStandingEnv


def main():
    args = ArgParserTrain().parse_args()
    Trainer(args)


class Trainer():
    def __init__(self, args):
        self.setup(args)
        self.logger.log_start(sys.argv, args)
        env = self.create_env(args)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        obs_dim = env.obs_shape[0]
        if args.env == "HumanoidHybrid":
            act_dim = 1
        else:
            act_dim = env.action_space.shape[0]

        self.buf = ReplayBuffer(obs_dim, act_dim, args, max_size=int(args.replay_buffer_size))
        env.buf = self.buf
        self.args = args

        self.policy = SAC(obs_dim, act_dim,
                     init_temperature=args.init_temperature,
                     alpha_lr=args.alpha_lr,
                     actor_lr=args.actor_lr,
                     critic_lr=args.critic_lr,
                     tau=args.tau,
                     discount=args.discount,
                     critic_target_update_freq=args.critic_target_update_freq,
                     args=args)
        ts_info = {}

        if args.teacher_student:
            teacher_policy = SAC(68,
                                 1,
                                 init_temperature=args.init_temperature,
                                 alpha_lr=args.alpha_lr,
                                 actor_lr=args.actor_lr,
                                 critic_lr=args.critic_lr,
                                 tau=args.tau,
                                 discount=args.discount,
                                 critic_target_update_freq=args.critic_target_update_freq,
                                 args=args)
            teacher_policy.load(os.path.join(args.load_dir + '/model', 'best_model'), load_optimizer=False)
            for param in teacher_policy.parameters():
                param.requires_grad = False
            env.teacher_policy = teacher_policy

        self.train_sac(env,act_dim,ts_info)


    def setup(self, args):
        ts = time.gmtime()
        ts = time.strftime("%m-%d-%H-%M", ts)
        exp_name = args.env + '_' + ts + '_' + 'seed_' + str(args.seed)
        exp_name = exp_name + '_' + args.tag if args.tag != '' else exp_name
        experiment_dir = os.path.join(args.work_dir, exp_name)

        utils.make_dir(experiment_dir)
        self.video_dir = utils.make_dir(os.path.join(experiment_dir, 'video'))
        self.model_dir = utils.make_dir(os.path.join(experiment_dir, 'model'))
        self.buffer_dir = utils.make_dir(os.path.join(experiment_dir, 'buffer'))
        self.logger = RLLogger(experiment_dir)

        with open(os.path.join(experiment_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, sort_keys=True, indent=4)

    def create_env(self, args):
        env_generator = env_function(args)
        return env_generator(args, args.seed)

    def train_sac(self, env, act_dim, ts_info):
        state, done = env.reset(store_buf=True), False
        t = 0
        self.last_power_update = -np.inf
        self.last_duration = np.inf
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0
        self.curriculum = True
        best_reward = -np.inf

        while t < int(self.args.max_timesteps):

            # Select action randomly or according to policy
            if (t < self.args.start_timesteps):
                action = np.clip(2 * np.random.random_sample(size=act_dim) - 1, -env.power, env.power)
            else:
                action = self.policy.sample_action(state)

            next_state, reward, done, _ = env.step(a=action)

            if self.args.test_policy:
                image_l = env.render()
                # image_r = images[episode_timesteps]
                # cv2.imshow('image',np.concatenate((image_l,image_r), axis=-2))
                cv2.imshow('image', image_l)
                cv2.waitKey(2)

            episode_timesteps += 1

            self.buf.add(state, action, next_state, reward, env.terminal_signal)
            state = next_state
            episode_reward += reward

            # Train agent after collecting sufficient data
            if (t >= self.args.start_timesteps):
                self.policy.train(self.buf, self.args.batch_size)

            if done:
                self.logger.log_train_episode(t, episode_num, episode_timesteps, episode_reward, self.policy.loss_dict, env, self.args)
                self.policy.reset_record()
                state, done = env.reset(store_buf=True), False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

            if t % self.args.test_interval == 0:
                test_reward, min_test_reward, video = self.run_tests(env.power_base)
                for i, v in enumerate(video):
                    if len(v) != 0:
                        imageio.mimsave(os.path.join(self.video_dir, 't_{}_{}.mp4'.format(t, i)), v, fps=30)
                criteria = test_reward if self.args.avg_reward else min_test_reward
                self.curriculum = self.update_power(env, criteria, t)
                if (test_reward > best_reward):
                    self.policy.save(os.path.join(self.video_dir, 'best_model'))
                    best_reward = test_reward
                    self.logger.info("Best model saved")
                self.logger.log_test(test_reward, min_test_reward, self.curriculum, env.power_base)
            t += 1


    def update_power(self, env, criteria, t, threshold=60):
        if not self.curriculum:
            return False
        if criteria > threshold:
            env.power_base = max(env.power_end, 0.95 * env.power_base)
            if env.power_base == env.power_end:
                return False
            self.last_duration = t - self.last_power_update
            self.last_power_update = t

        else:
            current_stage_length = t - self.last_power_update
            if current_stage_length > min(1000000, max(300000, 1.5 * self.last_duration)) and env.power_base < 1.0:
                env.power_base = env.power_base / 0.95
                env.power_end = env.power_base
                return False

        return True


    def run_tests(self, power_base, teacher_policy=None):
        test_env_generator = env_function(self.args)
        test_env = test_env_generator(self.args, self.args.seed + 10)
        test_env.power = power_base
        if self.args.teacher_student:
            test_env.teacher_policy = teacher_policy
        test_reward = []
        video_index = [np.random.random_integers(0, self.args.test_iterations - 1)]
        video_array = []
        for i in range(self.args.test_iterations):
            video = []
            state, done = test_env.reset(test_time=True), False
            episode_timesteps = 0
            episode_reward = 0
            while not done:
                action = self.policy.select_action(state)
                state, reward, done, _ = test_env.step(action, test_time=True)
                episode_reward += reward
                episode_timesteps += 1
                if i in video_index:
                    video.append(test_env.render())
            video_array.append(video)
            test_reward.append(episode_reward)
        test_reward = np.array(test_reward)
        return test_reward.mean(), test_reward.min(), video_array


if __name__ == "__main__":
    main()
