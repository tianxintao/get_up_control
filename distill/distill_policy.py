import os

import imageio
import numpy as np
import torch

from main import Trainer, ArgParserTrain
from student_policy import StudentPolicy
from utils import ReplayBuffer


def main():
    args = ArgParserDistill().parse_args()
    trainer = DistillTrainer(args)
    if args.test_distillation:
        trainer.test_distill()
    else:
        trainer.distill_policy()


class ArgParserDistill(ArgParserTrain):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument('--test_distillation', default=False, action='store_true')
        self.add_argument('--distill_dir', default=None, type=str)
        self.add_argument('--load_buffer', default=None, type=str)
        self.add_argument('--save_buffer', default=False, action='store_true')


class DistillTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.buf = ReplayBuffer(self.env._state.shape[0], self.act_dim, args, max_size=int(args.replay_buffer_size))
        if self.args.load_buffer:
            self.buf.load(os.path.join(self.args.load_buffer, 'buffer/distill_data.npz'))

        self.student_policy = StudentPolicy(
            self.env._state.shape[0],
            self.env.teacher_env.action_space.shape[0],
            self.args
        )

    def env_function(self):
        return HumanoidStandupVelocityDistillEnv

    def reduce_state(self, conditioned_state):
        state = []
        obs_dim = self.env.teacher_env.env.observation_spec()["observations"].shape[0]
        state.append(conditioned_state[:obs_dim])
        state.append([conditioned_state[-1]])
        return np.concatenate(state)

    def test_distill(self):
        self.student_policy.trunk.load_state_dict(
            torch.load(os.path.join(self.args.distill_dir + '/model', 'best_model_policy.pt')))
        test_reward, min_test_reward, video = self.run_tests(self.env.power_base, self.student_policy)
        print("Average Test Reward: {:.4f}".format(test_reward))
        for i, v in enumerate(video):
            if len(v) != 0:
                imageio.mimsave(os.path.join(self.video_dir, 'step_{}.mp4'.format(i)), v, fps=30)

    def distill_policy(self):

        collect_policy = self.policy
        best_reward_overall = -np.inf
        for iteration in range(80):
            episode_reward = 0
            episode_timesteps = 0
            episode_num = 0
            best_reward = -np.inf
            state = self.env.reset(store_buf=False, test_time=True)
            done = False

            self.logger.log_data_collection(iteration)

            tuples = 40000 if iteration == 0 else 10000

            if not (self.args.load_buffer and iteration == 0):
                for t in range(tuples):
                    if iteration == 0:
                        current_state = self.env.teacher_state
                    else:
                        current_state = state
                    with torch.no_grad():
                        action = collect_policy.select_action(current_state)
                        expert_action = self.policy.select_action(self.env.teacher_state)
                        next_state, reward, done, extra = self.env.step(a=action, supply_target=(iteration != 0))
                        pd_residual = expert_action + extra["pd_base"] - extra["current_pose"]
                        episode_timesteps += 1
                        self.buf.add(state, pd_residual, next_state, reward, self.env.terminal_signal)
                        state = next_state
                        episode_reward += reward
                        # image_l = self.env.render()
                        # cv2.imshow('image', image_l)
                        # cv2.waitKey(3)

                    if done:
                        self.logger.log_episode_collection(t, episode_num, episode_timesteps, episode_reward, self.env)
                        state, done = self.env.reset(store_buf=False, test_time=True), False
                        episode_reward = 0
                        episode_timesteps = 0
                        episode_num += 1

                if self.args.save_buffer:
                    self.buf.save(os.path.join(self.buffer_dir, "distill_data"))

                self.logger.log_policy_training(iteration)

            gradient_updates = 40000 if iteration == 0 else 10000

            for j in range(gradient_updates):
                self.student_policy.train(self.buf, self.args.batch_size)
                if (j + 1) % (gradient_updates / 2) == 0:
                    test_reward, _, video = self.run_tests(self.env.power_base, self.student_policy)
                    self.logger.info("Train Iteration: {}, MSELoss: {:.4f}, Test Reward: {:.4f}".format(
                        iteration, np.array(self.student_policy.loss_dict["action"]).mean(), test_reward)
                    )
                    self.student_policy.reset_record()

                    if (test_reward > best_reward):
                        best_state_dict = self.student_policy.state_dict()
                        if (test_reward > best_reward_overall):
                            file_name = os.path.join(self.model_dir, 'best_model')
                            self.student_policy.save(file_name)
                            best_reward_overall = test_reward
                            self.logger.info("Best model saved")
                        best_reward = test_reward

                    for i, v in enumerate(video):
                        if len(v) != 0:
                            imageio.mimsave(
                                os.path.join(self.video_dir, 'iteration_{}_step_{}_{}.mp4'.format(iteration, j + 1, i)),
                                v, fps=30)

            collect_policy = self.student_policy
            # self.student_policy.load_state_dict(best_state_dict)


if __name__ == "__main__":
    main()
