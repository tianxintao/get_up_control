import time
import imageio
import numpy as np
import cv2
import os
import utils
import json
import sys
from gym.spaces import Box, Discrete
import torch
from tensorboardX import SummaryWriter
from PPO import PPO
from SAC import SAC
from torch.utils.tensorboard import SummaryWriter
from env import CustomMountainCarEnv, HumanoidStandupEnv, HumanoidStandupRandomEnv, HumanoidBenchEnv
from utils import PGBuffer, ReplayBuffer
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def env_function(args):
    if args.env == 'HumanoidStandup':
        return HumanoidStandupEnv
    elif args.env == "HumanoidRandom":
        return HumanoidStandupRandomEnv
    elif args.env == "HumanoidBench":
        return HumanoidBenchEnv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HumanoidStandup', choices=['HumanoidStandup','HumanoidRandom','HumanoidBench','MountainCarContinuous-v0'])
    parser.add_argument("--policy", default="SAC",choices=['SAC', 'PPO'])
    parser.add_argument('--original', default=False, action='store_true', help='if set true, use the default power/strength parameters')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--scheduler', default=False, action='store_true')
    parser.add_argument('--test_policy', default=False, action='store_true')
    parser.add_argument('--velocity_penalty', default=False, action='store_true')
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--power", default=1.0, type=float)
    parser.add_argument("--power_end", default=0.4, type=float)
    parser.add_argument('--max_timesteps', type=int, default=10000000, help='Number of simulation steps to run')
    parser.add_argument('--test_interval', type=int, default=20000, help='Number of simulation steps between tests')
    parser.add_argument('--test_iterations', type=int, default=10, help='Number of simulation steps between tests')
    parser.add_argument('--replay_buffer_size', type=int, default=1e6, help='Number of simulation steps to run')
    parser.add_argument('--gamma', type=float, default=0.999, help='discount factor')
    parser.add_argument('--max_ep_len', type=int, default=1000)
    parser.add_argument('--custom_reset', default=False, action='store_true')
    parser.add_argument("--work_dir", default='./experiment/')
    parser.add_argument("--load_dir", default=None, type=str)

    # Terrain hyperparameters
    parser.add_argument('--add_terrain', default=False, action='store_true')
    parser.add_argument('--terrain_dim', default=16, type=int)
    parser.add_argument('--heightfield_dim', default=9, type=int)

    # SAC hyperparameters
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--discount", default=0.99)
    parser.add_argument("--init_temperature", default=0.1)
    parser.add_argument("--critic_target_update_freq", default=2, type=int)
    parser.add_argument("--alpha_lr", default=1e-4, type=float)
    parser.add_argument("--actor_lr", default=1e-4, type=float)
    parser.add_argument("--critic_lr", default=1e-4, type=float)
    parser.add_argument("--tau", default=0.005)
    parser.add_argument("--start_timesteps", default=10000, type=int)

    # PPO hyperparameters
    # Deprecated: used for mountain car problems
    parser.add_argument('--steps_per_epoch', type=int, default=1000, help='Number of env steps to run during optimizations')
    parser.add_argument('--lam', type=float, default=0.98, help='GAE-lambda factor')
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

    if args.add_terrain:
        assert args.env=="HumanoidRandom", "{} environment does not output terrain".format(args.env)

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
    logger.info(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    logger.info("---------------------------------------")

    tb = SummaryWriter(log_dir=os.path.join(experiment_dir, 'tb_logger'))

    env_generator = env_function(args)
    env = env_generator(args, args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # env.seed(args.seed)
    obs_dim = env.obs_shape[0]
    if isinstance(env.action_space, Discrete):
        discrete = True
        act_dim = env.action_space.n
    else:
        discrete = False
        act_dim = env.action_space.shape[0]
    
        start_time = time.time()

    if args.policy == "PPO":
        policy = PPO(obs_dim, act_dim, discrete).to(device)
        steps_per_epoch = int(args.steps_per_epoch)
        buf = PGBuffer(obs_dim, act_dim, discrete, steps_per_epoch, args, device)
        train_ppo(policy,env,tb,logger,buf,args,video_dir)
    elif args.policy == "SAC":
        buf = ReplayBuffer(obs_dim, act_dim, args, max_size=int(args.replay_buffer_size))
        policy = SAC(obs_dim,act_dim,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            tau=args.tau,
            discount=args.discount,
            critic_target_update_freq=args.critic_target_update_freq,
            args=args)
        if args.load_dir:
            env.power_base = 0.4
            # buf.load(os.path.join(args.load_dir+'/buffer','checkpoint.npz'))
            policy.load(os.path.join(args.load_dir+'/model','checkpoint'),load_optimizer=True)
        train_sac(policy, env, tb, logger, buf, args, video_dir, buffer_dir, model_dir, act_dim)



def train_sac(policy, env, tb, logger, replay_buffer, args, video_dir, buffer_dir, model_dir, action_dim):
    state, done = env.reset(), False
    t = 0
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    best_reward = -np.inf
    deterministic = True if args.load_dir != None else False

    while t < int(args.max_timesteps):

        # Select action randomly or according to policy
        if t < args.start_timesteps and args.load_dir == None:
            action = np.clip(2 * np.random.random_sample(size=action_dim) - 1, -env.power, env.power)
        else:
            action = policy.sample_action(np.array(state["scalar"]), terrain=state["terrain"])


        next_state, reward, done, _ = env.step(a=action)

        if args.test_policy:
            cv2.imshow('image',env.render())
            cv2.waitKey(1)

        episode_timesteps += 1

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, env.terminal_signal)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps and not args.test_policy:
            policy.train(replay_buffer, curriculum, args.batch_size)


        if done:
            console_output = "Total T: {t} Episode Num: {episode_num} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}"\
                .format(t=t + 1, episode_num=episode_num + 1, episode_timesteps=episode_timesteps, episode_reward=episode_reward)
            tb.add_scalar("Train/Reward", episode_reward, t + 1)
            if args.debug and t >= args.start_timesteps:
                tb.add_scalar("Train/Critic_loss", np.array(policy.critic_loss).mean(), t+1)
                tb.add_scalar("Train/Actor_loss", np.array(policy.actor_loss).mean(), t+1)
                tb.add_scalar("Train/Temperature_loss", np.array(policy.temperature_loss).mean(), t+1)
                tb.add_scalar("Train/Temperature", np.array(policy.temperature).mean(), t+1)
                console_output += "|C_Loss: {:.3f}".format(np.array(policy.critic_loss).mean())
                console_output += "|A_Loss: {:.3f}".format(np.array(policy.actor_loss).mean())
                console_output += "|T_Loss: {:.3f}".format(np.array(policy.temperature_loss).mean())
                console_output += "|T: {:.3f}".format(np.array(policy.temperature).mean())
                # console_output += "|G_mean: {:.3f}".format(np.array(policy.grad).mean())
                # console_output += "|G_max: {:.3f}".format(np.array(policy.grad).max())
                policy.reset_record()
            # Reset environment
            logger.info(console_output)
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        if t % args.test_interval == 0:
            test_reward, min_test_reward, video = run_tests(policy, env, args, True)
            tb.add_scalar("Test/Alpha", env.power_base, t)
            tb.add_scalar("Test/Reward", test_reward, t)
            if len(video) != 0:
                imageio.mimsave(os.path.join(video_dir, 't_{}.mp4'.format(t)), video, fps=30)
            save_checkpoint = env.adjust_power(min_test_reward)
            if(test_reward > best_reward):
                policy.save(os.path.join(model_dir,'best_model'))
                best_reward = test_reward
                logger.info("Best model saved")
            if args.velocity_penalty: replay_buffer.penalty_coeff = max(best_reward - 200, 0)/600
            curriculum = env.power_base > args.power_end and env.power_base < args.power
            logger.info("-------------------------------------------------")
            logger.info("Evaluation over 10 episodes: {:.3f}, minimum reward: {:.3f}, Curriculum: {}, Penalty: {:.3f}".\
                format(test_reward, min_test_reward, curriculum, replay_buffer.penalty_coeff))
            logger.info("-------------------------------------------------")
            logger.info("Current power: {:.3f}".format(env.power_base))
            if (save_checkpoint):
                replay_buffer.save(os.path.join(buffer_dir,'checkpoint'))
                policy.save(os.path.join(model_dir,'checkpoint'))
                logger.info("Checkpoint saved, deterministic env starts")
                deterministic = True
        t += 1


    
# Deprecated: used for mountain car problems
def train_ppo(policy,env,tb,logger,buf,args,video_dir):
    o, ep_ret, ep_len = env.reset(), 0, 0

    ep_count = 0  # just for logging purpose, number of episodes run
    t = 0
    video_tag = True
    # Main loop: collect experience in env and update/log each epoch
    while t < int(args.max_timesteps):
        a, v, logp = policy.step(torch.as_tensor(o, dtype=torch.float32).to(device))

        next_obs, reward, done, _ = env.step(a)
        ep_ret += reward
        ep_len += 1
        t += 1

        # save and log
        buf.store(o, a, reward, v, logp)
        o = next_obs
        epoch_ended = t % args.steps_per_epoch == 0 and t != 0

        if done or epoch_ended:
            ep_count += 1
            # if trajectory didn't reach terminal state, bootstrap value target
            if not env.terminal_signal or epoch_ended:
                _, v, _ = policy.step(torch.as_tensor(o, dtype=torch.float32).to(device))
            else:
                v = 0
            buf.finish_path(v)

            console_output = "Total T: {t} Episode Num: {episode_num} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}"\
                  .format(t=t, episode_num=ep_count, episode_timesteps=ep_len, episode_reward=ep_ret)
            tb.add_scalar("Train/Reward", ep_ret, t)
            o, ep_ret, ep_len = env.reset(), 0, 0

            if epoch_ended:
                policy.update(buf)

        if t % args.test_interval == 0:
            test_reward, video = run_tests(policy, env, args, video_tag, logger)
            tb.add_scalar("Test/Reward", test_reward, t)
            if len(video) != 0 and t % (5 * args.test_interval) == 0:
                out = cv2.VideoWriter(os.path.join(video_dir, 't_{}.mp4'.format(t)), cv2.VideoWriter_fourcc(*'MP4V'), 60, (128, 128))
                for frame in video:
                    out.write(frame)
                out.release()
            # video_tag = False
            env.adjust_power(test_reward)



def run_tests(policy, train_env, args, video_tag):
    test_env_generator = env_function(args)
    test_env = test_env_generator(args, args.seed)
    # test_env.seed(args.seed + 10)
    test_env.set_power(train_env.export_power())
    test_reward = []
    video = []
    for i in range(args.test_iterations):
        state, done = test_env.reset(test_time=True), False
        episode_reward = 0
        while not done:
            action = policy.select_action(np.array(state["scalar"]), terrain=state["terrain"])
            if i==0 and video_tag:
                video.append(test_env.render(mode="rgb_array"))
            state, reward, done, _ = test_env.step(action)
            episode_reward += reward
        test_reward.append(episode_reward)
    test_reward = np.array(test_reward)
    return test_reward.mean(), test_reward.min(), video


if __name__ == "__main__":
    main()
