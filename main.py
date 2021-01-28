import time
import cv2
import numpy as np
import gym
import os
import utils
import json
import sys
from gym.spaces import Box, Discrete
import torch
from tensorboardX import SummaryWriter
from PPO import PPO

from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from env import CustomMountainCarEnv, HumanoidStandupEnv
from utils import PGBuffer
from env import CustomMountainCarEnv
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='MountainCarContinuous-v0')
    parser.add_argument('--original', default=False, action='store_true', help='if set true, use the default power/strength parameters')
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument('--max_timesteps', type=int, default=5000000, help='Number of simulation steps to run')
    parser.add_argument('--test_interval', type=int, default=10000, help='Number of simulation steps between tests')
    parser.add_argument('--gamma', type=float, default=0.999, help='discount factor')
    parser.add_argument('--lam', type=float, default=0.98, help='GAE-lambda factor')
    parser.add_argument('--steps_per_epoch', type=int, default=1000, help='Number of env steps to run during optimizations')
    parser.add_argument('--max_ep_len', type=int, default=1000)
    parser.add_argument("--work_dir", default='./experiment/')

    parser.add_argument('--train_pi_iters', type=int, default=4)
    parser.add_argument('--train_v_iters', type=int, default=40)
    parser.add_argument('--pi_lr', type=float, default=3e-4, help='Policy learning rate')
    parser.add_argument('--v_lr', type=float, default=1e-3, help='Value learning rate')

    parser.add_argument('--psi_mode', type=str, default='gae', help='value to modulate logp gradient with [future_return, gae]')
    parser.add_argument('--loss_mode', type=str, default='vpg', help='Loss mode [vpg, ppo]')
    parser.add_argument('--clip_ratio', type=float, default=0.2, help='PPO clipping ratio')

    parser.add_argument('--render_interval', type=int, default=100, help='render every N')
    parser.add_argument('--log_interval', type=int, default=100, help='log every N')
    parser.add_argument("--tag", default="")

    args = parser.parse_args()

    ts = time.gmtime()
    ts = time.strftime("%m-%d-%H-%M", ts)
    exp_name = args.env + '_' + ts + '_' + 'seed_' + str(args.seed)
    exp_name = exp_name + '_' + args.tag if args.tag != '' else exp_name
    experiment_dir = os.path.join(args.work_dir, exp_name)

    utils.make_dir(experiment_dir)
    video_dir = utils.make_dir(os.path.join(experiment_dir, 'video'))
    model_dir = utils.make_dir(os.path.join(experiment_dir, 'model'))
    buffer_dir = utils.make_dir(os.path.join(experiment_dir, 'buffer'))
    logger = utils.create_logger(experiment_dir)
    

    with open(os.path.join(experiment_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    logger.info(str(sys.argv))
    logger.info("---------------------------------------")
    logger.info(f"Policy: PPO, Env: {args.env}, Seed: {args.seed}")
    logger.info("---------------------------------------")

    tb = SummaryWriter(log_dir=os.path.join(experiment_dir, 'tb_logger'))

    env = CustomMountainCarEnv()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = HumanoidStandupEnv(args.original)
    env.render()
    env.seed(args.seed)
    obs_dim = env.observation_space.shape[0]
    if isinstance(env.action_space, Discrete):
        discrete = True
        act_dim = env.action_space.n
    else:
        discrete = False
        act_dim = env.action_space.shape[0]

    # actor critic 
    policy = PPO(obs_dim, act_dim, discrete).to(device)
    # print('Number of parameters', count_vars(ac))

    # Set up experience buffer
    steps_per_epoch = int(args.steps_per_epoch)
    buf = PGBuffer(obs_dim, act_dim, discrete, steps_per_epoch, args, device)

    # Prepare for interaction with environment
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    ep_count = 0  # just for logging purpose, number of episodes run
    t = 0
    video_tag = False
    # Main loop: collect experience in env and update/log each epoch
    while t < int(args.max_timesteps):
        a, v, logp = policy.step(torch.as_tensor(o, dtype=torch.float32).to(device))

        next_obs, reward, done, _ = env.step(a)
        # env.render()
        ep_ret += reward
        ep_len += 1
        t += 1

        # save and log
        buf.store(o, a, reward, v, logp)
        o = next_obs

        # timeout = ep_len == args.max_ep_len
        # terminal = done or timeout
        epoch_ended = t % args.steps_per_epoch == 0 and t != 0

        if done or epoch_ended:
            ep_count += 1
            # if trajectory didn't reach terminal state, bootstrap value target
            if not env.terminal_signal or epoch_ended:
                _, v, _ = policy.step(torch.as_tensor(o, dtype=torch.float32).to(device))
            else:
                v = 0
            buf.finish_path(v)

            logger.info("Total T: {t} Episode Num: {episode_num} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}"
                  .format(t=t, episode_num=ep_count, episode_timesteps=ep_len, episode_reward=ep_ret))
            tb.add_scalar("Train/Reward", ep_ret, t)
            o, ep_ret, ep_len = env.reset(), 0, 0

            if epoch_ended:
                policy.update(buf)

        if t % args.test_interval == 0:
            test_reward, video = run_tests(policy, env, args, video_tag)
            tb.add_scalar("Test/Reward", test_reward, t)
            if len(video) != 0:
                out = cv2.VideoWriter(os.path.join('video', 'power_{:.5f}_speed_{:.5f}.mp4'.format(power,max_speed)), cv2.VideoWriter_fourcc(*'MP4V'), 30, (600, 400))
                for frame in video:
                    out.write(frame)
                out.release()
            video_tag = False
            env.adjust_power(test_reward)


def run_tests(policy, train_env, args, video_tag):
    test_env = HumanoidStandupEnv(args.original)
    test_env.seed(args.seed + 77)
    test_env.set_power(train_env.export_power())
    avg_reward = 0
    video = []
    for i in range(10):
        state, done = test_env.reset(), False
        while not done:
            action = policy.act(torch.as_tensor(state, dtype=torch.float32).to(device))
            if i==0 and False:
                video.append(test_env.render(mode="rgb_array"))
            state, reward, done, _ = test_env.step(action)
            avg_reward += reward

    logger.info("-------------------------------------------------")
    logger.info("Evaluation over 10 episodes: {:.3f}".format(avg_reward / 10.0))
    logger.info("-------------------------------------------------")
    # if video_tag: test_env.viewer.close()
    return avg_reward / 10.0, video


if __name__ == "__main__":
    main()
