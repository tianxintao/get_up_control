import time
import imageio
import numpy as np
import cv2
import os
import utils
import json
import sys
import copy
from gym.spaces import Box, Discrete
import torch
import copy
from tensorboardX import SummaryWriter
from PPO import PPO
from SAC import SAC
from TQC import TQC
from torch.utils.tensorboard import SummaryWriter
from env import CustomMountainCarEnv, CheetahEnv, HumanoidStandupEnv, HumanoidStandupCollectEnv, HumanoidStandupRandomEnv, HumanoidBenchEnv, HumanoidChairEnv, HumanoidStandingEnv, HumanoidStandupVelocityEnv
from utils import PGBuffer, ReplayBuffer
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ArgParserTrain(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument('--env', type=str, default='HumanoidStandup', choices=['Cheetah', 'HumanoidStandup', 'HumanoidRandom','HumanoidBench','HumanoidChair','HumanoidStanding','HumanoidStandupVelocityEnv','MountainCarContinuous-v0'])
        self.add_argument("--policy", default="SAC",choices=['SAC', 'PPO', 'TQC'])
        self.add_argument('--original', default=False, action='store_true', help='if set true, use the default power/strength parameters')
        self.add_argument('--debug', default=False, action='store_true')
        self.add_argument('--scheduler', default=False, action='store_true')
        self.add_argument('--imitation_reward', default=False, action='store_true')
        self.add_argument('--collect_traj', default=False, action='store_true')
        self.add_argument('--test_policy', default=False, action='store_true')
        self.add_argument('--velocity_penalty', default=False, action='store_true')
        self.add_argument('--relabel_data', default=False, action='store_true')
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

        # TQC hyperparameters
        self.add_argument("--n_quantiles", default=25, type=int)
        self.add_argument("--n_nets", default=5, type=int)

        # PPO hyperparameters
        # Deprecated: used for mountain car problems
        self.add_argument('--steps_per_epoch', type=int, default=4000, help='Number of env steps to run during optimizations')
        self.add_argument('--lam', type=float, default=0.97, help='GAE-lambda factor')
        self.add_argument('--train_pi_iters', type=int, default=4)
        self.add_argument('--train_v_iters', type=int, default=40)
        self.add_argument('--pi_lr', type=float, default=3e-4, help='Policy learning rate')
        self.add_argument('--v_lr', type=float, default=1e-3, help='Value learning rate')
        self.add_argument('--psi_mode', type=str, default='gae', help='value to modulate logp gradient with [future_return, gae]')
        self.add_argument('--loss_mode', type=str, default='vpg', help='Loss mode [vpg, ppo]')
        self.add_argument('--clip_ratio', type=float, default=0.2, help='PPO clipping ratio')

        self.add_argument('--render_interval', type=int, default=100, help='render every N')
        self.add_argument('--log_interval', type=int, default=100, help='log every N')
        self.add_argument("--tag", default="")

def env_function(args):
    if args.env == 'HumanoidStandup':
        if args.collect_traj:
            return HumanoidStandupCollectEnv
        elif args.changing_velocity:
            return HumanoidStandupVelocityEnv
        return HumanoidStandupEnv
    elif args.env == "HumanoidRandom":
        return HumanoidStandupRandomEnv
    elif args.env == "HumanoidBench":
        return HumanoidBenchEnv
    elif args.env == "HumanoidChair":
        return HumanoidChairEnv
    elif args.env == "HumanoidStanding":
        return HumanoidStandingEnv
    elif args.env == "Cheetah":
        return CheetahEnv

def main():
    args = ArgParserTrain().parse_args()
    
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

    teacher_policy = None
    teacher_env = None

    # env.seed(args.seed)
    obs_dim = env.obs_shape[0]
    if isinstance(env.action_space, Discrete):
        discrete = True
        act_dim = env.action_space.n
    else:
        discrete = False
        act_dim = env.action_space.shape[0]
    
        start_time = time.time()

    # if args.policy == "PPO":
    #     policy = PPO(obs_dim, act_dim, discrete).to(device)
    #     steps_per_epoch = int(args.steps_per_epoch)
    #     buf = PGBuffer(obs_dim, act_dim, discrete, steps_per_epoch, args, device)
    #     train_ppo(policy,env,tb,logger,buf,args,video_dir,buffer_dir)
    if args.policy == "SAC" or "TQC":
        buf = ReplayBuffer(obs_dim, act_dim, args, max_size=int(args.replay_buffer_size), extra_dim_dict=env.get_extra_dim())
        if args.policy == "SAC":
            policy = SAC(obs_dim,act_dim,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            tau=args.tau,
            discount=args.discount,
            critic_target_update_freq=args.critic_target_update_freq,
            args=args)
        elif args.policy == "TQC":
            top_quantiles_to_drop = 10
            policy = TQC(
                state_dim=obs_dim,
                action_dim=act_dim,
                discount=args.discount,
                tau=args.tau,
                top_quantiles_to_drop=top_quantiles_to_drop,
                args=args
            )
    elif args.policy == "PPO":
        steps_per_epoch = int(args.steps_per_epoch)
        buf = PGBuffer(obs_dim, act_dim, discrete, steps_per_epoch, args, device)
        policy = PPO(obs_dim, act_dim, discrete).to(device)

    if args.teacher_student:
        teacher_policy = SAC(68,
        act_dim,
        init_temperature=args.init_temperature,
        alpha_lr=args.alpha_lr,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        tau=args.tau,
        discount=args.discount,
        critic_target_update_freq=args.critic_target_update_freq,
        args=args)
        teacher_policy.load(os.path.join(args.load_dir+'/model','best_model'),load_optimizer=False)
        for param in teacher_policy.parameters():
            param.requires_grad = False
        env.teacher_policy = teacher_policy
    
    # if args.load_dir:
    #     env.power_base = args.power
    #     # buf.load(os.path.join(args.load_dir+'/buffer','checkpoint.npz'))
        # policy.load(os.path.join(args.load_dir+'/model','best_model'),load_optimizer=True)
    policy.load(os.path.join('experiment/HumanoidStandup_10-18-17-34_seed_0_varying_speed_0.2_0.8'+'/model','best_model'),load_optimizer=True)
    #     # env.set_chair_parameters(0, 20, 0.1)
    
    train_sac(policy, env, tb, logger, buf, args, video_dir, buffer_dir, model_dir, act_dim, teacher_policy, teacher_env)




def train_sac(policy, env, tb, logger, replay_buffer, args, video_dir, buffer_dir, model_dir, action_dim, teacher_policy, teacher_env):
    initial_state = None
    state, done = env.reset(), False
    t = 0
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    best_reward = -np.inf
    deterministic = True if args.load_dir != None else False

    while t < int(args.max_timesteps):
        
        behavior_action = 1
        # Select action randomly or according to policy
        if args.policy == "SAC" or "TQC":
            if (t < args.start_timesteps):
                action = np.clip(2 * np.random.random_sample(size=action_dim) - 1, -env.power, env.power)
            else:
                action = policy.sample_action(np.array(state["scalar"]), terrain=state["terrain"])
        elif args.policy == "PPO":
            action, v, logp = policy.sample_action(np.array(state["scalar"]), terrain=state["terrain"])


        next_state, reward, done, reaction_force = env.step(a=action)

        if args.test_policy:
            image_l = env.render()
            # image_r = images[episode_timesteps]
            # cv2.imshow('image',np.concatenate((image_l,image_r), axis=-2))
            cv2.imshow('image',image_l)
            cv2.waitKey(2)

        episode_timesteps += 1

        if args.policy == "SAC" or "TQC":
            # Store data in replay buffer
            replay_buffer.add(state, action, next_state, reward, env.terminal_signal, reaction_force, behavior_action)
        else:
            replay_buffer.store(state, action, reward, v, logp)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if (args.policy == "SAC" or args.policy == "TQC") and t >= args.start_timesteps and not args.test_policy and replay_buffer.size >= args.start_timesteps:
            policy.train(replay_buffer, env.curriculum_finished, args.batch_size)
        
        epoch_ended = (t + 1) % args.steps_per_epoch == 0 and t != 0 if args.policy == "PPO" else False

        if done or epoch_ended:
            console_output = "Total T: {t} Episode Num: {episode_num} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}"\
                .format(t=t + 1, episode_num=episode_num + 1, episode_timesteps=episode_timesteps, episode_reward=episode_reward)
            tb.add_scalar("Train/Reward", episode_reward, t + 1)

            if args.policy == "PPO":
                if not env.terminal_signal or epoch_ended:
                    _, v, _ = policy.sample_action(np.array(state["scalar"]), terrain=state["terrain"])
                else:
                    v = 0
                replay_buffer.finish_path(v)

            if epoch_ended:
                policy.update(replay_buffer)

            if (args.debug and t >= args.start_timesteps):
                if args.policy == "SAC" or "TQC":
                    tb.add_scalar("Train/Actor_loss", np.array(policy.actor_loss).mean(), t+1)
                    console_output += "|A_Loss: {:.3f}".format(np.array(policy.actor_loss).mean())
                    tb.add_scalar("Train/Critic_loss", np.array(policy.critic_loss).mean(), t+1)
                    tb.add_scalar("Train/Temperature_loss", np.array(policy.temperature_loss).mean(), t+1)
                    tb.add_scalar("Train/Temperature", np.array(policy.temperature).mean(), t+1)
                    console_output += "|C_Loss: {:.3f}".format(np.array(policy.critic_loss).mean())
                    console_output += "|T_Loss: {:.3f}".format(np.array(policy.temperature_loss).mean())
                    console_output += "|T: {:.3f}".format(np.array(policy.temperature).mean())
                    if args.predict_force:
                        console_output += "|F_Loss: {:.3f}".format(np.array(policy.reaction_force_loss).mean())
                    if args.changing_velocity:
                        console_output += "|Velocity: {:.3f}".format(env.speed)
                elif args.policy == "PPO" and epoch_ended:
                    console_output += "|pi_Loss: {:.3f}".format(np.array(policy.pi_loss).mean())
                    console_output += "|V_Loss: {:.3f}".format(np.array(policy.v_loss).mean())

            policy.reset_record()
            # Reset environment
            logger.info(console_output)
            # if args.teacher_student:
            #     while True:
            #         state, done = env.reset(initial_state=initial_state), False
            #         if np.abs(env.physics.center_of_mass_position()[2] - traj["com"][0][2]) < 0.3: break
            # else:
            state, done = env.reset(), False

            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        if t % args.test_interval == 0:
            test_reward, min_test_reward, video, test_env = run_tests(policy, env, args, True, teacher_policy)
            tb.add_scalar("Test/Alpha", env.power_base, t)
            tb.add_scalar("Test/Reward", test_reward, t)
            for i,v in enumerate(video):
                if len(v) != 0:
                    imageio.mimsave(os.path.join(video_dir, 't_{}_{}.mp4'.format(t, i)), v, fps=30)
            criteria = test_reward if args.avg_reward else min_test_reward
            save_checkpoint = env.adjust_power(criteria, replay_buffer)

            if(test_reward > best_reward):
                policy.save(os.path.join(model_dir,'best_model'))
                best_reward = test_reward
                logger.info("Best model saved")
            if args.velocity_penalty: replay_buffer.penalty_coeff = max(best_reward - 200, 0)/600
            curriculum = env.power_base < args.power and (not env.curriculum_finished)
            logger.info("-------------------------------------------------")
            logger.info("Evaluation over 10 episodes: {:.3f}, minimum reward: {:.3f}, Curriculum: {}".\
                format(test_reward, min_test_reward, curriculum))
            logger.info("-------------------------------------------------")
            logger.info("Current power: {:.3f}".format(env.power_base))
            if args.env == "HumanoidChair": logger.info("Current angle: {:.3f}".format(env.chair_angle_range))
            if (save_checkpoint):
                replay_buffer.save(os.path.join(buffer_dir,'checkpoint'))
                policy.save(os.path.join(model_dir,'checkpoint'))
                logger.info("Checkpoint saved, deterministic env starts")
                deterministic = True
        t += 1


def run_tests(policy, train_env, args, video_tag, teacher_policy=None):
    test_env_generator = env_function(args)
    test_env = test_env_generator(args, args.seed+10)
    test_env.teacher_policy = teacher_policy
    # test_env.seed(args.seed + 10)
    test_env.set_power(train_env.export_power())
    # test_env.set_chair_parameters(train_env.chair_angle_mean, train_env.chair_angle_range, train_env.height_diff)
    test_reward = []
    # video_index = [np.random.random_integers(0, args.test_iterations-1)]
    video_index = np.arange(0, args.test_iterations)
    video_array = []
    for i in range(args.test_iterations):
        video = []
        # test_traj, test_images, _ = run_one_episode(test_teacher_env, teacher_policy, args)
        # test_env.trajectoty_data = test_traj
        state, done = test_env.reset(test_time=True), False
        episode_reward = 0
        while not done:
            action = policy.select_action(np.array(state["scalar"]), terrain=state["terrain"])
            state, reward, done, _ = test_env.step(action, test_time=True)
            episode_reward += reward
        if i in video_index and video_tag:
        #     length = len(test_env.images)
        #     test_images = test_images[:length]
            video_array.append(np.array(test_env.images))
        test_reward.append(episode_reward)
    test_reward = np.array(test_reward)
    return test_reward.mean(), test_reward.min(), video_array, test_env


if __name__ == "__main__":
    main()


# # Deprecated: used for mountain car problems
# def train_ppo(policy,env,tb,logger,buf,args,video_dir,buffer_dir):
#     o, ep_ret, ep_len = env.reset(), 0, 0
#     o = o["scalar"]
#     speed = 80

#     ep_count = 0  # just for logging purpose, number of episodes run
#     t = 0
#     video_tag = True
#     # Main loop: collect experience in env and update/log each epoch
#     while t < int(args.max_timesteps):
#         a, v, logp = policy.step(torch.as_tensor(o, dtype=torch.float32).to(device))

#         next_obs, reward, done, _ = env.step(a)
#         next_obs = next_obs["scalar"]
#         ep_ret += reward
#         ep_len += 1
#         t += 1

#         # save and log
#         buf.store(o, a, reward, v, logp)
#         o = next_obs
#         epoch_ended = t % args.steps_per_epoch == 0 and t != 0

#         if done or epoch_ended:
#             ep_count += 1
#             # if trajectory didn't reach terminal state, bootstrap value target
#             if not env.terminal_signal or epoch_ended:
#                 _, v, _ = policy.step(torch.as_tensor(o, dtype=torch.float32).to(device))
#             else:
#                 v = 0
#             buf.finish_path(v)

#             if epoch_ended:
#                 policy.update(buf)

#             console_output = "Total T: {t} Episode Num: {episode_num} Episode T: {episode_timesteps} Reward: {episode_reward:.3f} | pi_loss {pi_loss:.3f} | v_loss {v_loss:.3f} "\
#                   .format(t=t, episode_num=ep_count, episode_timesteps=ep_len, episode_reward=ep_ret, pi_loss=np.array(policy.pi_loss).mean(), v_loss=np.array(policy.v_loss).mean())
#             logger.info(console_output)

#             tb.add_scalar("Train/Reward", ep_ret, t)
#             o, ep_ret, ep_len = env.reset(), 0, 0
#             o = o["scalar"]
#             policy.reset_record()

#         if t % args.test_interval == 0:
#             test_reward, min_test_reward, video, test_env = run_tests(policy, env, args, True)
#             tb.add_scalar("Test/Reward", test_reward, t)
#             if len(video) != 0:
#                 imageio.mimsave(os.path.join(video_dir, 't_{}.mp4'.format(t)), video, fps=30)
#             if (test_reward > 125):
#                 data = {
#                     "xquat": np.array(test_env.xquat),
#                     "extremities": np.array(test_env.extremities),
#                     "com": np.array(test_env.com),
#                     "com_vel": np.array(test_env.com_vel)
#                 }
#                 env.trajectoty_data = utils.interpolate_motion(data)
#                 speed = speed * 0.8
#                 filename = 'trajectory3d_interpolated_' + str(int(speed)) + '.npz'
#                 np.savez(os.path.join(buffer_dir, filename), **data)
#             logger.info("-------------------------------------------------")
#             logger.info("Evaluation over 10 episodes: {:.3f}, minimum reward: {:.3f}".\
#                 format(test_reward, min_test_reward))
#             logger.info("-------------------------------------------------")
#             if args.imitation_data and args.relabel_data:
#                 logger.info("Current speed: {:.3f}".format(speed))
#             env.adjust_power(test_reward)