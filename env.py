from gym.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv
from gym.utils import seeding
from dm_control import suite, mujoco
from dm_control.suite.base import Task
from dm_control.utils import rewards
from dm_control.suite.humanoid import Humanoid, Physics
from dm_control.rl import control
from dm_control.suite import common
from utils import quaternion_multiply
import dm_env
from SAC import SAC
import os
import numpy as np
import math
import imageio
import utils
import cv2
import copy
import torch

STANDING_POSE = np.array([
                [ 1.     ,  0.     ,  0.     , -0.     ],
                [ 1.     ,  0.     ,  0.     , -0.     ],
                [ 0.99726,  0.00106, -0.04865,  0.00822],
                [ 0.99666, -0.02682, -0.05084,  0.00691],
                [ 0.99172, -0.05925, -0.04908, -0.02655],
                [ 0.97489, -0.05076,  0.20083, -0.0405 ],
                [ 0.99674, -0.00639,  0.02945, -0.00895],
                [ 0.99172, -0.05925, -0.04908, -0.02655],
                [ 0.97489, -0.05076,  0.20083, -0.0405 ],
                [ 0.99674, -0.00639,  0.02945, -0.00895],
                [ 0.95135,  0.14246,  0.26623, -0.04127],
                [ 0.51532, -0.01268,  0.73672, -0.43528],
                [ 0.51532, -0.01268,  0.73672, -0.43528],
                [ 0.95135,  0.14246,  0.26623, -0.04127],
                [ 0.51532, -0.01268,  0.73672, -0.43528],
                [ 0.51532, -0.01268,  0.73672, -0.43528],
                ])
STANDING_ORIENTATION = 0.986

class HumanoidStandupEnv():
    _STAND_HEIGHT = 1.55

    def __init__(self, args, seed):
        self.env = suite.load(domain_name="humanoid", task_name="stand", task_kwargs={'random': seed})
        self.args = args
        self.env._flat_observation = True
        self.physics = self.env.physics
        self.power_base = args.power
        self.power_end = args.power_end
        self.trajectoty_data = None
        self.action_space = self.env.action_spec()
        self.reset()
        self.obs_shape = self._state.shape
        self.physics.reload_from_xml_path('data/humanoid_static.xml')
        self.buf = None

    def step(self, a, test_time=False):
        self._step_num += 1
        self.env.step(a)
        self.obs = self.env._task.get_observation(self.physics)
        return self._state, self._reward, self._done, None

    def reset(self, test_time=False, store_buf=False):
        self._step_num = 0
        self._initial_steps = True
        if not test_time: self.sample_power()
        self.env.reset()
        self.obs = self.env._task.get_observation(self.physics)
        current_state = self._state
        for _ in range(50):
            action = np.random.normal(loc=0.0, scale=0.1, size=self.action_space.shape[0])
            self.env.step(action)
            self.obs = self.env._task.get_observation(self.physics)
            if store_buf:
                self.buf.add(current_state, action, self._state, self._reward, self.terminal_signal)
                current_state = self._state
        self._initial_steps = False
        self.terminal_signal = False
        return self._state

    def render(self, mode=None):
        return self.env.physics.render(height=480, width=480, camera_id=1)

    def sample_power(self, std=0.04):
        self.power = np.clip(self.power_base + np.random.randn() * std, self.power_end, 1)

    def set_power(self, power_dict):
        self.power = power_dict["power"]
        self.trajectoty_data = power_dict["trajectory"]

    @property
    def _state(self):
        _state = []
        for part in self.obs.values():
            _state.append(part if not np.isscalar(part) else [part])
        _state.append([self.power])
        return np.concatenate(_state)

    @property
    def _done(self):
        if self._step_num >= 250:
            return True
        return False

    @property
    def _standing(self):
        standing = rewards.tolerance(self.physics.head_height(),
                                     bounds=(self._STAND_HEIGHT, float('inf')),
                                     margin=self._STAND_HEIGHT / 4, value_at_margin=0.2, sigmoid='long_tail')
        return standing

    @property
    def _small_control(self):
        control_val = rewards.tolerance(self.physics.control(), margin=1,
                                        value_at_margin=0,
                                        sigmoid='quadratic').mean()
        return (4 + 1 * control_val) / 5

    @property
    def _dont_move(self):
        horizontal_velocity = self.physics.center_of_mass_velocity()[[0, 1]]
        if not self._standing_reward:
            return rewards.tolerance(horizontal_velocity, bounds=[-0.3, 0.3], margin=1.2).mean()
        else:
            return rewards.tolerance(horizontal_velocity, margin=1.2).mean()

    @property
    def _closer_feet(self):
        left_feet_pos = self.physics.data.xpos[10, :2]
        right_feet_pos = self.physics.data.xpos[7, :2]
        feet_distance = np.sqrt(((right_feet_pos - left_feet_pos) ** 2).sum())
        feet_reward = rewards.tolerance(feet_distance,
                                        bounds=(0, 0.9), sigmoid='quadratic',
                                        margin=0.38, value_at_margin=0)
        return feet_reward

    @property
    def _upright(self):
        if self.physics.center_of_mass_position()[2] < 0.5:
            return 1.0
        if self._standing_reward:
            upright = rewards.tolerance(self.physics.torso_upright(),
                                        bounds=(0.96, float('inf')), sigmoid='linear',
                                        margin=0.16, value_at_margin=0)
        else:
            upright = rewards.tolerance(self.physics.torso_upright(),
                                        bounds=(0.9, float('inf')), sigmoid='linear',
                                        margin=1.9, value_at_margin=0)
        
        return upright
    
    @property
    def _standing_pose(self):
        body_rotation = self.physics.data.xquat[1:]
        root_rotation = body_rotation[0]
        root_conjugate = np.array(root_rotation)
        root_conjugate[-3:] = -root_conjugate[-3:]
        body_rotation_offset = [quaternion_multiply(root_conjugate, r) for r in body_rotation]

        epsilon = 1e-7
        product = body_rotation_offset * STANDING_POSE
        angle_diff = 2 * (np.arccos(np.clip(np.abs(product.sum(axis=-1)), -1+epsilon, 1-epsilon)) ** 2)

        return np.exp(-angle_diff.mean()/2)

    @property
    def _reward(self):
        # self._standing_reward = self.physics.center_of_mass_position()[2] > 0.75 and (not self._initial_steps)
        self._standing_reward = False
        if not self._standing_reward:
            return self._upright * self._standing * self._small_control * self._dont_move * self._closer_feet
        else:
            body_height = rewards.tolerance(self.physics.center_of_mass_position()[2],
                                     bounds=(0.80, float('inf')),
                                     margin=0.1, value_at_margin=0.3)
            # print("body height reward: {:4f}".format(body_height))
            # print("upright reward: {:4f}".format(self._upright))
            # print("pose tracking reward: {:.4f}".format(self._standing_pose))
            # print("move reward: {:.4f}".format(self._dont_move))
            # print("final reward: {:.4f}".format(body_height * self._upright * self._standing_pose * self._dont_move))
            return body_height * self._upright * self._standing_pose * self._dont_move
            


class HumanoidStandupVelocityEnv(HumanoidStandupEnv):
    qpos_to_ctrl_index = np.array([1, 0, 2, 3, 4, 5, 6, 8, 7, 9, 10, 11, 12, 14, 13, 15, 16, 17, 18, 19, 20])

    def __init__(self, args, seed):
        if args.teacher_student:
            self.teacher_env = HumanoidStandupHybridEnv(args, args.seed)
        super().__init__(args, seed)

    def run_one_episode(self):
        video = []
        self.teacher_env.power = self.args.teacher_power
        state, done = self.teacher_env.reset(test_time=True), False
        self.teacher_env.xquat = []
        self.teacher_env.extremities = []
        self.teacher_env.com = []
        self.teacher_env.com_vel = []
        self.teacher_env.qpos = []
        self.teacher_env.qvel = []
        episode_reward = 0
        while not done:
            action = self.teacher_policy.select_action(state["scalar"], terrain=state["terrain"])
            state, reward, done, _ = self.teacher_env.step(action, test_time=True)
            episode_reward += reward
        trajectory = {
            "xquat": np.array(self.teacher_env.xquat),
            "extremities": np.array(self.teacher_env.extremities),
            "com": np.array(self.teacher_env.com),
            "com_vel": np.array(self.teacher_env.com_vel),
            "qpos": np.array(self.teacher_env.qpos),
            "qvel": np.array(self.teacher_env.qvel)
        }
        return trajectory, self.teacher_env.images, self.teacher_env.initial_state

    def reset(self, test_time=False, initial_reset=False):
        self._step_num = 0
        self.reset_count += 1
        if not test_time: self.sample_power()
        if test_time:
            self.images = []
            self.xquat = []
            self.extremities = []
            self.com = []
            self.com_vel = []
            self.qpos = []
            self.qvel = []
        self.speed = np.random.randint(0, 7) * 0.1 + 0.2
        # self.speed = 0.8
        # if self.trajectoty_data != None:
        #     self.interpolated_trajectory = utils.interpolate_motion(self.trajectoty_data, self.speed)

        repeat = True
        if self.custom_reset:
            repeat = True
            while repeat:
                self.env.reset()

                self.physics.data.qpos[:] = self.default_qpos
                self.physics.data.qpos[:3] = [0, 0, 0.5]
                self.physics.data.qpos[3:7] = [0.7071, 0, 0.7071, 0]
                self.physics.after_reset()
                repeat = False
                # if self.physics.data.ncon == 0: repeat = False
        else:
            if not self.args.teacher_student or self.teacher_policy == None or initial_reset:
                self.timestep = self.env.reset()
                self.interpolated_trajectory = self.trajectoty_data

            else:
                while True:
                    trajectory, self.teacher_images, initial_state = self.run_one_episode()
                    self.physics.data.qpos[:] = initial_state["qpos"]
                    self.physics.data.qvel[:] = initial_state["qvel"]
                    self.physics.after_reset()
                    self.interpolated_trajectory = utils.interpolate_motion(trajectory, self.speed)

                    action = np.zeros(self.action_space.shape)
                    self.env.step(action)
                    # while self.env.physics.center_of_mass_position()[2] > 0.25:
                    while self.env.physics.center_of_mass_velocity()[2] < 0:
                        self.env.step(action)
                        if test_time == True and self.teacher_policy != None: self.images.append(self.render())

                    self._step_num = 0
                    if np.abs(self.env.physics.center_of_mass_position()[2] - self.interpolated_trajectory["com"][0][
                        2]) < 0.1: break

        self.obs = self.env._task.get_observation(self.env._physics)
        self.terminal_signal = False

        if not test_time: self.sample_power()
        return self._state

    def step(self, a, test_time=False):
        self._step_num += 1
        if self.original: a = a * self.power
        self.timestep = self.substep(a)
        self.obs = self.env._task.get_observation(self.physics)
        # mujoco.engine.mjlib.mj_rnePostConstraint(self.physics.model.ptr, self.physics.data.ptr)
        # print(self._reaction_force, self.physics.data.ncon)
        reaction_force = self._reaction_force if self.args.predict_force else None
        if test_time: self.images.append(self.render())
        if test_time and (self.args.imitation_reward or self.args.teacher_student):
            self.xquat.append(self.physics.data.xquat[1:].flatten())
            self.extremities.append(self.physics.extremities())
            self.com.append(self.physics.center_of_mass_position())
            self.com_vel.append(self.physics.center_of_mass_velocity())
            self.qpos.append(self.physics.data.qpos.copy())
            self.qvel.append(self.physics.data.qvel.copy())
        return self._state, self._reward, self._done, reaction_force

    def substep(self, action):
        for _ in range(self.env._n_sub_steps):
            pose_diff = \
            ((self.interpolated_trajectory["qpos"][self._step_num - 1][7:] + action) - self.physics.data.qpos[7:])[
                self.qpos_to_ctrl_index]
            kp = np.ones_like(self.physics.model.actuator_gear[:, 0]) * 100
            kp[7] = kp[8] = kp[13] = kp[14] = 30
            kp = kp / self.physics.model.actuator_gear[:, 0]
            kd = kp / 10
            final_action = kp * pose_diff - ((kd * self.physics.data.qvel[6:])[self.qpos_to_ctrl_index])
            self.env._task.before_step(final_action, self.env._physics)
            self.env._physics.step()
        self.env._task.after_step(self.env._physics)

    @property
    def _done(self):
        if np.abs(
                self.env.physics.center_of_mass_position()[2] - self.interpolated_trajectory["com"][self._step_num - 1][
                    2]) > 0.2:
            self.terminal_signal = True
            return True
        if self._step_num >= 375:
            return True
        return False

    @property
    def _state(self):
        _state = []
        for part in self.obs.values():
            _state.append(part if not np.isscalar(part) else [part])
        if self.args.imitation_reward:
            _state.append(self.interpolated_trajectory["qpos"][self._step_num + 10])
            _state.append(self.interpolated_trajectory["qpos"][self._step_num + 20])
            # _state.append([self.speed])
        _state.append([self.power])
        return {
            "scalar": np.concatenate(_state),
            "terrain": None,
        }

    @property
    def _reward(self):
        if self.args.imitation_reward:
            return self.compute_imitaion_reward(
                self.physics.data.xquat[1:].flatten(),
                self.physics.extremities(),
                self.physics.center_of_mass_position(),
                self._step_num - 1)

    def render(self, mode=None):
        image_l = super().render()
        image_r = self.teacher_images.pop(0)
        return np.concatenate((image_l, image_r), axis=1)

    # def compute_imitaion_reward(self, xquat, extremities, com, index):

    #     epsilon = 1e-7
    #     product = xquat.reshape(-1, 4) * self.interpolated_trajectory["xquat"][index].reshape(-1, 4)
    #     angle_diff = 2 * (np.arccos(np.clip(np.abs(product.sum(axis=-1)), -1+epsilon, 1-epsilon)) ** 2)
    #     root_orientation = angle_diff[0]
    #     xquat_dist = angle_diff[1:].sum()
    #     # xquat_dist = np.sum((xquat - self.trajectoty_data["xquat"][index]) ** 2)
    #     extremities_dist = np.sum((extremities - self.interpolated_trajectory["extremities"][index]) ** 2) * 40
    #     com_dist = np.sum((com[2] - self.interpolated_trajectory["com"][index][2]) ** 2) * 10
    #     distance = 0.1 * xquat_dist + 1.0 * extremities_dist + 1.0 * com_dist
    #     # return 0.2 * np.exp(-xquat_dist) + 0.3 * np.exp(-extremities_dist) + 0.5 * np.exp(-com_dist)
    #     return 0.2 * np.exp(-com_dist) + 0.1 * np.exp(-root_orientation) + 0.15 * np.exp(-extremities_dist) + 0.55 * np.exp(-xquat_dist)

    def compute_imitaion_reward(self, xquat, extremities, com, index):

        epsilon = 1e-7
        product = xquat.reshape(-1, 4) * self.interpolated_trajectory["xquat"][index].reshape(-1, 4)
        xquat_dist = 2 * np.arccos(np.clip(np.abs(product.sum(axis=-1)), -1 + epsilon, 1 - epsilon))
        # xquat_dist = np.sum((xquat - self.trajectoty_data["xquat"][index]) ** 2)
        extremities_dist = np.sum((extremities - self.interpolated_trajectory["extremities"][index]) ** 2)
        com_dist = np.sum((com - self.interpolated_trajectory["com"][index]) ** 2)
        distance = 0.1 * xquat_dist + 1.0 * extremities_dist + 1.0 * com_dist
        return np.exp(-np.sum(distance / 4))


class HumanoidStandupHybridEnv(HumanoidStandupEnv):
    qpos_to_ctrl_index = np.array([1, 0, 2, 3, 4, 5, 6, 8, 7, 9, 10, 11, 12, 14, 13, 15, 16, 17, 18, 19, 20])

    def __init__(self, args, seed):
        super().__init__(args, seed)

        self.default_qpos = self.physics.data.qpos.copy()
        # adjust elbow
        self.default_qpos[24] = -2.2
        self.default_qpos[27] = -2.2
        # adjust shoulder
        self.default_qpos[23] = -0.3
        self.default_qpos[26] = 0.3

        self.standing_policy = SAC(
            self.obs_shape[0],
            self.action_space.shape[0],
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            tau=args.tau,
            discount=args.discount,
            critic_target_update_freq=args.critic_target_update_freq,
            args=args
        )
        self.standup_policy = copy.deepcopy(self.standing_policy)
        self.standing_policy.load(os.path.join(args.standing_policy + '/model', 'best_model'), load_optimizer=False)
        self.standup_policy.load(os.path.join(args.standup_policy + '/model', 'best_model'), load_optimizer=False)
        for param in self.standing_policy.parameters():
            param.requires_grad = False
        for param in self.standup_policy.parameters():
            param.requires_grad = False

    def step(self, a, test_time=False):
        self.action = a
        self._step_num += 1
        with torch.no_grad():
            self.obs = self.env._task.get_observation(self.physics)
            standup_state = np.array(self._state["scalar"])
            standup_state[-1] = 0.4
            standup_action = self.standup_policy.select_action(standup_state)
            standing_action = self.standing_policy.select_action(np.array(self._state["scalar"]))

        self.timestep = self.substep(a, standup_action, standing_action)
        self.obs = self.env._task.get_observation(self.physics)

        if test_time: self.images.append(self.render())
        if test_time and (self.args.imitation_reward or self.args.teacher_student):
            self.xquat.append(self.physics.data.xquat[1:].flatten())
            self.extremities.append(self.physics.extremities())
            self.com.append(self.physics.center_of_mass_position())
            self.com_vel.append(self.physics.center_of_mass_velocity())
            self.qpos.append(self.physics.data.qpos.copy())
            self.qvel.append(self.physics.data.qvel.copy())
        return self._state, self._reward, self._done, None

    def substep(self, coeff, standup_action, standing_action):
        coeff_scaled = coeff
        for _ in range(self.env._n_sub_steps):
            pose_diff = ((self.default_qpos[7:] + standing_action) - self.physics.data.qpos[7:])[
                self.qpos_to_ctrl_index]
            kp = self.physics.model.actuator_gear[:, 0] / 100 * 2
            kd = kp / 5
            final_standing_action = kp * pose_diff - ((kd * self.physics.data.qvel[6:])[self.qpos_to_ctrl_index])
            final_action = coeff_scaled * final_standing_action + (1 - coeff_scaled) * standup_action
            self.env._task.before_step(final_action, self.env._physics)
            self.env._physics.step()
        self.env._task.after_step(self.env._physics)

    @property
    def _reward(self):
        return super()._reward * self._small_control

    @property
    def _standing(self):
        return rewards.tolerance(self.physics.head_height(),
                                 bounds=(1.4, float('inf')),
                                 margin=1.4 / 3)


class HumanoidStandingEnv(HumanoidStandupEnv):
    qpos_to_ctrl_index = np.array([1, 0, 2, 3, 4, 5, 6, 8, 7, 9, 10, 11, 12, 14, 13, 15, 16, 17, 18, 19, 20])

    def __init__(self, args, seed):
        self.default_qpos = None
        super().__init__(args, seed)
        self.default_qpos = self.physics.data.qpos.copy()
        # adjust elbow
        self.default_qpos[24] = -2.2
        self.default_qpos[27] = -2.2
        # adjust shoulder
        self.default_qpos[23] = -0.3
        self.default_qpos[26] = 0.3

    def reset(self, test_time=False, store_buf=False):
        self.images = []
        repeat = True
        while repeat:
            with self.physics.reset_context():
                self.env.reset()
                if not self.default_qpos is None:
                    self.physics.data.qpos[:] = self.default_qpos
                utils.randomize_limited_and_rotational_joints(self.physics, k=0.1)
                self.physics.named.data.qpos[:3] = [0, 0, 1.4]
                self.physics.data.qpos[24] = -2.2
                self.physics.data.qpos[27] = -2.2
                self.physics.data.qpos[23] = -0.3
                self.physics.data.qpos[26] = 0.3

            self.physics.after_reset()
            if self.physics.data.ncon == 0: repeat = False

        self.obs = self.env._task.get_observation(self.physics)
        self._step_num = 0
        self.terminal_signal = False
        if not test_time: self.sample_power()
        return self._state

    def step(self, a, test_time=False):
        self.action = a
        self._step_num += 1
        self.timestep = self.substep(a)
        self.obs = self.env._task.get_observation(self.physics)

        if test_time: self.images.append(self.render())
        return self._state, self._reward, self._done, None

    def substep(self, action):
        for _ in range(self.env._n_sub_steps):
            pose_diff = ((self.default_qpos[7:] + action) - self.physics.data.qpos[7:])[self.qpos_to_ctrl_index]
            kp = self.physics.model.actuator_gear[:, 0] / 100 * 2
            kd = kp / 5
            final_action = kp * pose_diff - ((kd * self.physics.data.qvel[6:])[self.qpos_to_ctrl_index])
            self.env._task.before_step(final_action, self.env._physics)
            self.env._physics.step()
        self.env._task.after_step(self.env._physics)

    @property
    def _done(self):
        if self.physics.center_of_mass_position()[2] < 0.6:
            self.terminal_signal = True
            return True
        if self._step_num >= 1000:
            return True
        return False

    @property
    def _reward(self):
        return self._upright * self._standing * self._small_control * self._dont_move * self._closer_feet

    @property
    def _standing(self):
        return rewards.tolerance(self.physics.head_height(),
                                 bounds=(1.4, float('inf')),
                                 margin=1.4 / 3)

    @property
    def _small_control(self):
        control_val = rewards.tolerance(self.action, margin=1,
                                        value_at_margin=0,
                                        sigmoid='quadratic').mean()
        return (2 + 3 * control_val) / 5

    @property
    def _closer_feet(self):
        left_feet_pos = self.physics.data.xpos[10, :2]
        right_feet_pos = self.physics.data.xpos[7, :2]
        feet_distance = np.sqrt(((right_feet_pos - left_feet_pos) ** 2).sum())
        feet_reward = rewards.tolerance(feet_distance,
                                        bounds=(0, 0.5), sigmoid='quadratic',
                                        margin=0.1, value_at_margin=0.1)
        return (2 + feet_reward) / 3


class Humanoid2DStandupEnv(HumanoidStandupEnv):
    def __init__(self, args, seed):
        physics = Physics.from_xml_path('./data/humanoid_2D.xml')
        task = Humanoid(move_speed=0, pure_state=False)
        environment_kwargs = {}
        self.env = control.Environment(
            physics, task, time_limit=1000, control_timestep=.025,
            **environment_kwargs)
        self.args = args
        self.count = 0
        self.env._flat_observation = True
        self.physics = self.env.physics
        self.default_qpos = self.physics.data.qpos.copy()
        self.custom_reset = args.custom_reset
        self.power_end = args.power_end
        self.original = args.original
        self.power_base = args.power
        self.reset()
        self.action_space = self.env.action_spec()
        self.obs_shape = self._state["scalar"].shape
        self.trajectoty_data = np.load('./data/trajectory_64.npz')

    def reset(self, test_time=False):
        repeat = True
        self.xquat = []
        self.extremities = []
        self.com = []
        while repeat:
            with self.physics.reset_context():
                self.env.reset()
                if not self.default_qpos is None:
                    self.physics.data.qpos[:] = self.default_qpos
                    self.physics.data.qpos[:3] = [0, 0, 1.5]
                    self.physics.data.qpos[3:7] = [0.7071, 0.7071, 0, 0]
                    self.physics.after_reset()
                utils.randomize_limited_and_rotational_joints(self.physics, k=0.1)

            if self.physics.data.ncon == 0: repeat = False

        self.obs = self.env._task.get_observation(self.env._physics)
        self._step_num = 0
        self.terminal_signal = False
        if not test_time: self.sample_power()
        return self._state

    # def step(self, a):
    #     ret = super().step(a)
    #     self.xquat.append(self.physics.data.xquat[1:].flatten())
    #     self.extremities.append(self.physics.extremities())
    #     self.com.append(self.physics.center_of_mass_position())
    #     return ret

    @property
    def _done(self):
        if self._step_num >= 250:
            return True
        return self.timestep.last()
