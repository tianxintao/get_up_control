import time
import imageio
import numpy as np
import os
import utils
import sys
import torch
from tensorboardX import SummaryWriter
from PPO import PPO
from SAC import SAC
from env import CustomMountainCarEnv, HumanoidStandupEnv, HumanoidStandupRandomEnv, HumanoidBenchEnv
from utils import ReplayBuffer
from main import ArgParserTrain
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ArgParserCollect(ArgParserTrain):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_defaults(test_iterations=20, power=0.4, work_dir='./temp/')

def main():

    args = ArgParserCollect().parse_args()

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
    
    logger.info("---------------------------------------")
    logger.info(f"Env: {args.env}, Seed: {args.seed}, Policy Directory: {args.load_dir}")
    logger.info("---------------------------------------")

    env = HumanoidStandupEnv(args, args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # env.seed(args.seed)
    obs_dim = env.obs_shape[0]
    act_dim = env.action_space.shape[0]
    
    policy = SAC(obs_dim,act_dim,
        init_temperature=0.1,
        alpha_lr=1e-4,
        actor_lr=1e-4,
        critic_lr=1e-4,
        tau=5e-3,
        discount=0.99,
        critic_target_update_freq=2,
        args=args)
    if args.load_dir:
        env.power_base = args.power
        policy.load(os.path.join(args.load_dir+'/model','best_model'),load_optimizer=True)

    avg_speed = []
    for i in range(args.test_iterations):
        state, done = env.reset(test_time=True), False
        buf = ReplayBuffer(obs_dim, act_dim, args, max_size=int(args.replay_buffer_size))
        video = []
        episode_timesteps = 0
        episode_reward = 0  
        while not done:
            action = policy.select_action(np.array(state["scalar"]), terrain=state["terrain"])
            video.append(env.render(mode="rgb_array"))
            next_state, reward, done, reaction_force = env.step(action)
            buf.add(state, action, next_state, reward, env.terminal_signal, reaction_force)
            state = next_state
            episode_reward += reward
            episode_timesteps += 1
        logger.info("Episode Num: {episode_num} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}"\
                .format(episode_num=i + 1, episode_timesteps=episode_timesteps, episode_reward=episode_reward))
        buf.save(os.path.join(buffer_dir,'trajectory_{}'.format(i)))
        imageio.mimsave(os.path.join(video_dir, 'trajectory_{}.mp4'.format(i)), video, fps=30)
    logger.info("Overall average velocity: {velocity:.5f}".format(velocity=sum(avg_speed)/len(avg_speed)))

if __name__ == "__main__":
    main()
