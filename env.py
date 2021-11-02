from gym.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv
from gym.utils import seeding
from dm_control import suite, mujoco
from dm_control.suite.base import Task
from dm_control.utils import rewards
from dm_control.suite.humanoid import Humanoid, Physics
from dm_control.rl import control
from dm_control.suite import common
from SAC import SAC
import os
import numpy as np
import math
import imageio
import utils
import cv2
import copy
import torch


class CustomMountainCarEnv(Continuous_MountainCarEnv):
    # By default,
    max_speed_default = 0.07
    power_default = 0.0015

    def __init__(self, original):
        super().__init__()
        self.time_limit = 400
        self.original = original
        if self.original:
            self.max_speed = self.max_speed_default
            self.power = self.power_default
        else:
            self.max_speed = 0.175
            self.power = 0.00375

    def reset(self):
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        # if max_speed_current: self.max_speed = max_speed_current
        # if power_current: self.power = power_current
        self.step_count = 0
        self.terminal_signal = False
        return np.array(self.state)

    def adjust_power(self, test_reward):
        if self.original:
            return
        if test_reward > 90:
            self.max_speed = max(0.98 * self.max_speed, self.max_speed_default)
            self.power = max(0.98 * self.power, self.power_default)

    def export_power(self):
        return {
            "max_speed": self.max_speed,
            "power": self.power,
        }

    def set_power(self, power_dict):
        self.max_speed = power_dict["max_speed"]
        self.power = power_dict["power"]

    def step(self, action, exclude_energy=False):

        self.step_count += 1
        position = self.state[0]
        velocity = self.state[1]
        force = min(max(action[0], self.min_action), self.max_action)

        velocity += force * self.power - 0.0025 * math.cos(3 * position)
        if (velocity > self.max_speed): velocity = self.max_speed
        if (velocity < -self.max_speed): velocity = -self.max_speed
        position += velocity
        if (position > self.max_position): position = self.max_position
        if (position < self.min_position): position = self.min_position
        if (position == self.min_position and velocity < 0): velocity = 0

        # Convert a possible numpy bool to a Python bool.
        self.terminal_signal = bool(
            position >= self.goal_position and velocity >= self.goal_velocity
        )

        done = self.terminal_signal

        reward = 0
        if done:
            reward = 100.0

        if not exclude_energy:
            reward -= math.pow(action[0] * (self.power / 0.0015), 2) * 0.1 * ((0.0015 / self.power) ** 2)
        else:
            reward -= 0.05

        self.state = np.array([position, velocity])
        done = done or (self.step_count == 1000)
        return self.state, reward, done, {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed)  # Action space is not seeded by default
        return [seed]


class HumanoidStandupEnv():
    _STAND_HEIGHT = 1.55

    def __init__(self, args, seed):
        self.env = suite.load(domain_name="humanoid", task_name="stand", task_kwargs={'random': seed})
        self.reset_count = 0
        self.count = 0
        self.args = args
        self.env._flat_observation = True
        self.physics = self.env.physics
        self.default_qpos = self.physics.data.qpos.copy()
        self.custom_reset = args.custom_reset
        self.power_end = args.power_end
        self.original = args.original
        self.power_base = args.power
        self.trajectoty_data = None
        self.initial_state = None
        self.teacher_policy = None
        self.action_space = self.env.action_spec()
        if self.args.imitation_reward:
            self.trajectoty_data = np.load(self.args.imitation_data)
        self.reset()
        self.obs_shape = self._state["scalar"].shape
        self.physics.reload_from_xml_path('data/humanoid_static.xml')
        

    def get_extra_dim(self):
        if self.args.imitation_reward:
            return {
                "xquat": self.trajectoty_data["xquat"].shape,
                "extremities": self.trajectoty_data["extremities"].shape,
                "com": self.trajectoty_data["com"].shape
            }


    def step(self, a, test_time=False):
        self._step_num += 1
        if self.original: a = a * self.power
        # self.timestep = self.env.step(a)
        self.timestep = self.env.step(a)
        self.obs = self.timestep.observation
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

    def reset(self, test_time=False):
        self._step_num = 0
        self.reset_count += 1
        if test_time: self.images = []
        if not test_time: self.sample_power()
        if (self.args.imitation_reward or self.args.teacher_student) and test_time:
            self.xquat = []
            self.extremities = []
            self.com = []
            self.com_vel = []
            self.qpos = []
            self.qvel = []
        if self.custom_reset:
            # while True:
            #     self.physics.after_reset()
            #     if self.physics.data.ncon <= 0: break
            # self.physics(self, self.physics)
            repeat = True
            while repeat:
                self.env.reset()

                self.physics.data.qpos[:] = self.default_qpos
                self.physics.data.qpos[:3] = [0, 0, 0.5]
                self.physics.data.qpos[3:7] = [0.7071, 0, 0.7071, 0]
                # if not test_time and np.random.rand() > 0.2:
                #     # self._step_num = np.random.randint(1, 80)
                #     self._step_num = 60
                #     self.physics.data.qpos[:] = self.trajectoty_data["qpos"][self._step_num-1]
                #     self.physics.data.qvel[:] = self.trajectoty_data["qvel"][self._step_num-1]
                self.physics.after_reset()
                repeat = False
                # if self.physics.data.ncon == 0: repeat = False
        else:
            reset_step = 0
            self.timestep = self.env.reset()
            if test_time:
                self.initial_state = {
                    "qpos": self.physics.data.qpos.copy(),
                    "qvel": self.physics.data.qvel.copy()
                }
            action = np.zeros(self.action_space.shape)
            self.env.step(action)
            # while self.env.physics.center_of_mass_position()[2] > 0.25:
            while self.env.physics.center_of_mass_velocity()[2] < 0 or reset_step < 50:
                reset_step += 1
                self.env.step(action)
                if test_time: self.images.append(self.render())
            self._step_num = 0

        # with self.env.physics.reset_context():
        #     # Add custom reset
        self.obs = self.env._task.get_observation(self.env._physics)
        
        self.terminal_signal = False

        # self.velocity_record = []
        
        return self._state

    def render(self, mode=None):
        return self.env.physics.render(height=480, width=480, camera_id=1)

    @property
    def curriculum_finished(self):
        return (np.abs(self.power_base - self.power_end) <= 1e-3 and self.count > 1)

    def sample_power(self, std=0.02):
        if self.curriculum_finished or self.original:
            self.power = self.power_base
            return
        self.power = np.clip(self.power_base + np.random.randn() * std, self.power_end, 1)

    def adjust_power(self, test_reward, replay_buffer, threshold=40):
        if self.original or self.curriculum_finished:
            return False
        if test_reward > threshold:
            self.power_base = max(0.95 * self.power_base, self.power_end)
            # replay_buffer.reset()
            if np.abs(self.power_base - self.power_end) <= 1e-3:
                self.count += 1
        return self.curriculum_finished

    def export_power(self):
        return {
            "power": self.power_base,
            "trajectory": self.trajectoty_data
        }

    def set_power(self, power_dict):
        self.power = power_dict["power"]
        self.trajectoty_data = power_dict["trajectory"]
        
    @property
    def _state(self):
        _state = []
        for part in self.obs.values():
            _state.append(part if not np.isscalar(part) else [part])
        # if self.args.imitation_reward:
        #     _state.append([np.sin(self._step_num/200*np.pi),np.cos(self._step_num/200*np.pi)])
        _state.append([self.power])
        # print("velocity_z:{}, velocity:{}".format(np.concatenate(_state)[42],np.abs(np.concatenate(_state)[40:67]).mean()))
        return {
            "scalar": np.concatenate(_state),
            "terrain": None,
        }

    @property
    def _done(self):
        if self._step_num >= 400:
            return True
        return False

    @property
    def _standing(self):
        standing =  rewards.tolerance(self.physics.head_height(),
                                 bounds=(self._STAND_HEIGHT, float('inf')),
                                 margin=self._STAND_HEIGHT / 4)
        return (9 * standing + 1) / 10

    @property
    def _small_control(self):
        control_val = rewards.tolerance(self.physics.control(), margin=1,
                                        value_at_margin=0,
                                        sigmoid='quadratic').mean()
        return (2 + 3 * control_val) / 5

    @property
    def _dont_move(self):
        horizontal_velocity = self.physics.center_of_mass_velocity()[[0, 1]]
        return rewards.tolerance(horizontal_velocity, bounds=[-0.3, 0.3], margin=1.2).mean()

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
    def upright(self):
        upright = rewards.tolerance(self.physics.torso_upright(),
                                    bounds=(0.9, float('inf')), sigmoid='linear',
                                    margin=0.9, value_at_margin=0)
        return upright

    # @property
    # def _slow_motion(self):
    #     # if not self.args.velocity_penalty:
    #     #     return 1.0
    #     # else:
    #     #     if self.physics.center_of_mass_velocity()[2] >= 0.5:
    #     #         self.velocity_record.append(np.abs(self.physics.velocity()[6:]).mean())
    #     #         control_val = rewards.tolerance(self.physics.velocity()[6:], margin=10,
    #     #                                     value_at_margin=0.1,
    #     #                                     sigmoid='quadratic').mean()
    #     #         return (3 + control_val) / 4
    #     #     return 1.0
    #     if not self.args.velocity_penalty or self.physics.center_of_mass_velocity()[2] < 0.5:
    #         return 0.0
    #     return 0.1 * np.abs(self.physics.velocity()[6:]).mean()

    @property
    def _reward(self):
        
        if self.args.imitation_reward and not self.args.teacher_student:
            return self.compute_imitaion_reward(
                self.physics.data.xquat[1:].flatten(), 
                self.physics.extremities(), 
                self.physics.center_of_mass_position(), 
                self._step_num-1)

        # return self._standing * upright * self._dont_move
        return  self.upright * self._standing * self._small_control * self._dont_move * self._closer_feet
            
        
        # if self.physics.data.qpos[2] > 1.15:
        #     ankle_x = np.array([self.physics.data.qpos[14],self.physics.data.qpos[20]])
        #     ankle = rewards.tolerance(ankle_x, bounds=(0.0,0.0), margin = 0.8).mean()
        # else:
        #     ankle = 1.0
        

    def compute_imitaion_reward(self, xquat, extremities, com, index):

        epsilon = 1e-7
        product = xquat.reshape(-1, 4) * self.trajectoty_data["xquat"][index].reshape(-1, 4)
        xquat_dist = 2 * np.arccos(np.clip(np.abs(product.sum(axis=-1)), -1+epsilon, 1-epsilon))
        # xquat_dist = np.sum((xquat - self.trajectoty_data["xquat"][index]) ** 2)
        extremities_dist = np.sum((extremities - self.trajectoty_data["extremities"][index]) ** 2)
        com_dist = np.sum((com - self.trajectoty_data["com"][index]) ** 2)
        distance = 0.1 * xquat_dist + 1.0 * extremities_dist + 1.0 * com_dist
        return np.exp(-np.sum(distance/4))

    @property
    def _reaction_force(self):
        reaction_force = 0
        check_list = []
        # print("----------------------------------------------------------------------------------")
        if self.physics.data.ncon > 0:
            for record in self.physics.data.contact:
                # print("contact between {} and {}, Normal Direction: {}".format(
                #     self.physics.model.id2name(self.physics.model.geom_bodyid[record[-3]], 'body'),
                #     self.physics.model.id2name(self.physics.model.geom_bodyid[record[-4]], 'body'),
                #     record[2][:3]
                # ))
                if record[-3] == 0 or record[-4] == 0:
                    body_id = int(self.physics.model.geom_bodyid[record[-3] + record[-4]])
                    if not body_id in check_list:
                        reaction_force += self.physics.named.data.cfrc_ext[body_id][-1]
                        check_list.append(body_id)
        return min(reaction_force, 1600)/1600


class HumanoidStandupVelocityEnv(HumanoidStandupEnv):

    qpos_to_ctrl_index = np.array([1, 0, 2, 3, 4, 5, 6, 8, 7, 9, 10, 11, 12, 14, 13, 15, 16, 17, 18, 19, 20])

    def __init__(self, args, seed):
        if args.teacher_student:
            self.teacher_env = HumanoidStandupHybridEnv(args, args.seed)
        super().__init__(args,seed)
        
    def run_one_episode(self):
        video = []
        self.teacher_env.power = self.args.teacher_power
        state, done = self.teacher_env.reset(test_time=True), False
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
        return  trajectory, self.teacher_env.images, self.teacher_env.initial_state
    
    def reset(self, test_time=False):
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
            if not self.args.teacher_student or self.teacher_policy == None:
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
                        if test_time==True and self.teacher_policy != None: self.images.append(self.render())

                    self._step_num = 0
                    if np.abs(self.env.physics.center_of_mass_position()[2] - self.interpolated_trajectory["com"][0][2]) < 0.1: break
            

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
            pose_diff = ((self.interpolated_trajectory["qpos"][self._step_num-1][7:] + action) - self.physics.data.qpos[7:])[self.qpos_to_ctrl_index]
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
        if np.abs(self.env.physics.center_of_mass_position()[2] - self.interpolated_trajectory["com"][self._step_num-1][2]) > 0.2:
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
            _state.append(self.interpolated_trajectory["qpos"][self._step_num+10])
            _state.append(self.interpolated_trajectory["qpos"][self._step_num+20])
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
                self._step_num-1)

    def render(self, mode=None):
        image_l = super().render()
        image_r = self.teacher_images.pop(0)
        return np.concatenate((image_l, image_r),axis = 1)

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
        xquat_dist = 2 * np.arccos(np.clip(np.abs(product.sum(axis=-1)), -1+epsilon, 1-epsilon))
        # xquat_dist = np.sum((xquat - self.trajectoty_data["xquat"][index]) ** 2)
        extremities_dist = np.sum((extremities - self.interpolated_trajectory["extremities"][index]) ** 2)
        com_dist = np.sum((com - self.interpolated_trajectory["com"][index]) ** 2)
        distance = 0.1 * xquat_dist + 1.0 * extremities_dist + 1.0 * com_dist
        return np.exp(-np.sum(distance/4))


# class HumanoidStandupCollectEnv(HumanoidStandupEnv):

#     def reset(self, test_time=False):
#         state = super().reset(test_time=test_time)
#         self.xquat = []
#         self.extremities = []
#         self.com = []
#         self.com_vel = []
#         self.qpos = []
#         self.qvel = []
#         return state

#     def step(self, a, test_time=False):
#         ret = super().step(a)
#         self.xquat.append(self.physics.data.xquat[1:].flatten())
#         self.extremities.append(self.physics.extremities())
#         self.com.append(self.physics.center_of_mass_position())
#         self.com_vel.append(self.physics.center_of_mass_velocity())
#         self.qpos.append(self.physics.data.qpos.copy())
#         self.qvel.append(self.physics.data.qvel.copy())
#         return ret

class HumanoidStandupHybridEnv(HumanoidStandupEnv):

    qpos_to_ctrl_index = np.array([1, 0, 2, 3, 4, 5, 6, 8, 7, 9, 10, 11, 12, 14, 13, 15, 16, 17, 18, 19, 20])
    def __init__(self, args, seed):
        super().__init__(args,seed)

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
        self.standing_policy.load(os.path.join(args.standing_policy+'/model','best_model'),load_optimizer=False)
        self.standup_policy.load(os.path.join(args.standup_policy+'/model','best_model'),load_optimizer=False)
        for param in self.standing_policy.parameters():
            param.requires_grad = False
        for param in self.standup_policy.parameters():
            param.requires_grad = False

    def step(self, a, test_time=False):
        self.action = a
        self._step_num += 1
        with torch.no_grad():
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
            pose_diff = ((self.default_qpos[7:] + standing_action) - self.physics.data.qpos[7:])[self.qpos_to_ctrl_index]
            kp = self.physics.model.actuator_gear[:, 0] / 100 * 2
            kd = kp / 5
            final_standing_action = kp * pose_diff - ((kd * self.physics.data.qvel[6:])[self.qpos_to_ctrl_index])
            final_action = coeff_scaled * final_standing_action + (1-coeff_scaled) * standup_action
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
        super().__init__(args,seed)
        self.default_qpos = self.physics.data.qpos.copy()
        # adjust elbow
        self.default_qpos[24] = -2.2
        self.default_qpos[27] = -2.2
        # adjust shoulder
        self.default_qpos[23] = -0.3
        self.default_qpos[26] = 0.3

    def reset(self, test_time=False):
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

        self.obs = self.env._task.get_observation(self.env._physics)
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
        return super()._reward * self._small_control

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

class HumanoidStandupRandomEnv(HumanoidStandupEnv):
    random_terrain_path = './data/terrain.png'
    terrain_shape = 180
    # slope_terrain_path = './data/slope.png'
    xml_path = './data/humanoid_terrain.xml'

    def __init__(self, args, seed):
        with open('./data/humanoid_terrain.txt', 'r') as reader:
            self.xml_string = reader.read()
        HumanoidStandupEnv.__init__(self, args, seed)
        # self.create_random_terrain()
        # self.create_slope_terrain()
        self.create_random_terrain()

    def create_random_terrain(self):
        self.terrain = np.random.random_sample((self.terrain_shape, self.terrain_shape))
        imageio.imwrite(self.random_terrain_path, (self.terrain * 255).astype(np.uint8))

    def reset(self, test_time=False):
        self.create_random_terrain()
        with open(self.xml_path, 'w') as f:
            f.write(self.xml_string.format(terrain_height=self.args.max_height))
        self.physics.reload_from_xml_path(self.xml_path)
        return HumanoidStandupEnv.reset(self,test_time=test_time)

    # def create_slope_terrain(self):
    #     self.terrain_shape = 40
    #     slope_length = int(self.terrain_shape / 4)
    #     image_mid = int(self.terrain_shape / 2)
    #     if not os.path.exists(self.slope_terrain_path):
    #         image = np.zeros((self.terrain_shape, self.terrain_shape))
    #         # image = (image + (np.arange(20) + 1).reshape((1,20))) / 20 * 255
    #         image[:, image_mid - slope_length:image_mid + slope_length] = (np.arange(2 * slope_length) + 1).reshape(
    #             (1, 2 * slope_length)) / 20 * 255
    #         imageio.imwrite(self.slope_terrain_path, image)

    @property
    def _state(self):
        state = super()._state
        if self.args.add_terrain:
            state["terrain"] = self.get_terrain_height()
        return state
 
    def get_terrain_height(self):
        half_size = self.args.heightfield_dim // 2
        x, y, z = self.physics.center_of_mass_position()
        x_ind = int((x + 30) / 60 * self.terrain_shape)
        y_ind = int((y + 30) / 60 * self.terrain_shape)
        # collect the hightfield data from the nearby 5x5 region
        # self.terrain_profile = self.terrain[y_ind-1:y_ind+2,x_ind-1:x_ind+2]
        # height = self.terrain_profile.mean() * self.max_height
        return self.terrain[y_ind-half_size:y_ind+half_size+1,x_ind-half_size:x_ind+half_size+1] * self.args.max_height

    # @property
    # def _standing(self):
    #     mean_height = self.terrain_profile.mean() * self.max_height
    #     # print(self.physics.head_height()-self.get_terrain_height())
    #     return rewards.tolerance(self.physics.head_height()-mean_height,
    #                              bounds=(self._STAND_HEIGHT, float('inf')),
    #                              margin=self._STAND_HEIGHT / 4)

class HumanoidBenchEnv(HumanoidStandupEnv):
    _STAND_HEIGHT = 1.55
    _bench_height = 0.3
    bench_center = np.array([0, 0, _bench_height])
    xml_path = './data/humanoid_bench.xml'

    # feet_pos = np.array([0.57, 0.05, 0.05])

    # starting_pose = np.array(
    #     [ 0.35    ,  0.        ,  0.68      ,  0.866     ,  0.        ,
    #     0.5       ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        , -1.8       , -1.5       ,  0.        ,
    #     0.        ,  0.        ,  0.        , -1.8       , -1.5       ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ]
    # )

    # intermediate_pose = np.array(
    #     [ 0.43    ,  0.        ,  0.91      ,  0.024     ,  0.        ,
    #     0.383     ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        , -1.6       , -1.4       ,  0.        ,
    #     0.        ,  0.        ,  0.        , -1.6       , -1.4       ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ]
    # )
    
    # final_pose = np.array(
    #     [ 0.5     ,  0.        ,  1.3       ,  1.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.35      ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.35      ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ]
    # )

    def __init__(self, args, seed):
        self.default_qpos = None
        HumanoidStandupEnv.__init__(self, args, seed)
        self.physics.reload_from_xml_path(self.xml_path)
        self.default_qpos = self.physics.data.qpos.copy()

    def reset(self, test_time=False):
        repeat = True
        # self.physics.reload_from_xml_path(self.xml_path)
        while repeat:
            with self.physics.reset_context():
                self.env.reset()
                if not self.default_qpos is None:
                    self.physics.data.qpos[:] = self.default_qpos
                utils.randomize_limited_and_rotational_joints(self.physics, k=0.1)
                self.physics.named.data.qpos[:3] = [0, 0, 1.0]
                self.physics.named.data.qpos['left_hip_y'] = -90 / 180 * 3.14
                self.physics.named.data.qpos['right_hip_y'] = -90 / 180 * 3.14
                self.physics.named.data.qpos['left_knee'] = -75 / 180 * 3.14
                self.physics.named.data.qpos['right_knee'] = -75 / 180 * 3.14
            self.physics.after_reset()
            if self.physics.data.ncon == 0: repeat = False
        self.obs = self.env._task.get_observation(self.physics)
        self._step_num = 0
        self.terminal_signal = False
        if not test_time: self.sample_power()
        return self._state

    def adjust_power(self, test_reward, threshold=40):
        return super().adjust_power(test_reward, threshold=40)

    @property
    def _state(self):
        _state = []
        for part in self.obs.values():
            _state.append(part if not np.isscalar(part) else [part])

        torso_pos = self.physics.named.data.xpos['torso']
        torso_frame = self.physics.named.data.xmat['torso'].reshape(3, 3)
        dist_to_chair = (self.bench_center - torso_pos).dot(torso_frame)
        _state.append(dist_to_chair)
        
        _state.append(self.physics.named.data.sensordata['butt_touch'])

        _state.append([self.power])



        return {
            "scalar": np.concatenate(_state),
            "terrain": None,
        }

    @property
    def _done(self):
        if self.physics.center_of_mass_position()[2] < (self._bench_height + 0.1):
            self.terminal_signal = True
            return True
        if self._step_num >= 1000:
            return True
        return self.timestep.last()

    @property
    def _standing(self):
        return rewards.tolerance(self.physics.head_height(),
                                 bounds=(self._STAND_HEIGHT, float('inf')),
                                 margin=self._STAND_HEIGHT / 4)

    @property
    def _reaction_force(self):
        reaction_force = 0
        contact_array = np.zeros(16)
        check_list = []
        body_names = ["right_hand","left_hand","right_foot","left_foot"]
        body_array = [False] * 4
        # print("----------------------------------------------------------------------------------")
        if self.physics.data.ncon > 0:
            for record in self.physics.data.contact:
                # print("contact between {} and {}, Normal Direction: {}".format(
                #     self.physics.model.id2name(self.physics.model.geom_bodyid[record[-3]], 'body'),
                #     self.physics.model.id2name(self.physics.model.geom_bodyid[record[-4]], 'body'),
                #     record[2][:3]
                # ))
                # if record[-3] == 0 or record[-4] == 0:
                #     body_id = int(self.physics.model.geom_bodyid[record[-3] + record[-4]])
                #     check_list.append(body_id)
                
                # if record[-3] == 1 or record[-4] == 1:
                #     body_id = int(self.physics.model.geom_bodyid[record[-3] + record[-4] - 1])
                #     check_list.append(body_id)
                if record[-3] == 0 or record[-4] == 0:
                    body_id = int(self.physics.model.geom_bodyid[record[-3] + record[-4]])
                    # if not body_id in check_list:
                    #     reaction_force += self.physics.named.data.cfrc_ext[body_id][-1]
                    #     check_list.append(body_id)
                    if body_id == 8: body_array[2] = True
                    if body_id == 11: body_array[3] = True

                if record[-3] == 1 or record[-4] == 1:
                    body_id = int(self.physics.model.geom_bodyid[record[-3] + record[-4] - 1]) 
                    # if not body_id in check_list:
                    #     reaction_force += self.physics.named.data.cfrc_ext[body_id][-1]
                    #     check_list.append(body_id)
                    if body_id == 14: body_array[0] = True
                    if body_id == 17: body_array[1] = True

        force_array = self.physics.named.data.cfrc_ext[body_names][:,-1] * np.array(body_array).astype(float)
        # force_from_bench = self.physics.named.data.cfrc_ext[1][-1] # Body_id 1 responds to bench
        # force_array = np.abs(np.array([reaction_force, force_from_bench])).clip(min=0,max=1600)/1600
        # index = np.array(list(set(check_list)))
        # if len(index) > 0:
        #     contact_array[index-2] = 1
        # print(force_array)
        # print(force_array)
        return np.abs(force_array).clip(min=0,max=[800,800,1600,1600])/np.array([800,800,1600,1600])

    # @property
    # def _reward(self):
    #     weights = np.zeros_like(self.starting_pose)
    #     weights[:7] = 2
    #     weights[[12,13,18,19]] = 1
    #     if self._step_num < 30:
    #         dist = np.square((self.physics.data.qpos - self.starting_pose)*weights).sum() / 7
    #     elif self._step_num < 60:
    #         dist = np.square((self.physics.data.qpos - self.intermediate_pose)*weights).sum() / 7
    #     else:
    #         dist = np.square((self.physics.data.qpos - self.final_pose)*weights).sum() / 7

    #     left_feet_loss = np.square((self.physics.extremities()[3:6] - self.feet_pos)).mean() * 0.5
    #     right_feet_loss = np.square((self.physics.extremities()[9:12] - self.feet_pos)).mean() * 0.5
    #     dist += left_feet_loss + right_feet_loss
    #     reward = np.exp(-dist * 2)
    #     return reward


class HumanoidChairEnv(HumanoidBenchEnv):
    xml_path = './data/humanoid_chair.xml'
    bench_width = 0.4
    base_height = 0.25

    def __init__(self, args, seed):
        self.chair_angle_mean = 0
        self.chair_angle_range = 3
        self.height_diff = 0
        with open('./data/humanoid_chair.txt', 'r') as reader:
            self.model_str = reader.read()
        super().__init__(args, seed)
        
    def render(self, mode=None):
        return self.env.physics.render(height=256, width=256, camera_id=1)

    def reset(self, test_time=False):
        # self.physics.reload_from_xml_string(
        #     self.model_str.format(handle_height=0.1, handle_position=0.15),
        #     common.ASSETS
        # )
        angle, height = self.sample_angle()
        
        seat_height = self.base_height + height
        root_height = max(seat_height + 0.8, 1.0)
        self.physics.reload_from_xml_string(
            self.model_str.format(chair_angle=angle, seat_height=seat_height),
            common.ASSETS
        )
        self.bench_center = np.array([-0.15, 0, seat_height+0.05])

        repeat = True
        while repeat:
            with self.physics.reset_context():
                self.env.reset()
                if not self.default_qpos is None:
                    self.physics.data.qpos[:] = self.default_qpos
                utils.randomize_limited_and_rotational_joints(self.physics, k=0.1)
                self.physics.named.data.qpos[:3] = [0.0, 0, root_height]
                self.physics.named.data.qpos['left_hip_y'] = -(90-angle) / 180 * 3.14
                self.physics.named.data.qpos['right_hip_y'] = -(90-angle) / 180 * 3.14
                self.physics.named.data.qpos['left_knee'] = -(90-angle) / 180 * 3.14
                self.physics.named.data.qpos['right_knee'] = -(90-angle) / 180 * 3.14
            self.physics.after_reset()
            if self.physics.data.ncon == 0: repeat = False
        self.obs = self.env._task.get_observation(self.physics)
        self._step_num = 0
        self.terminal_signal = False
        if not test_time: self.sample_power()
        return self._state
        # return super().reset(test_time=test_time)

    @property
    def _state(self):
        _state = super()._state["scalar"]
        power = _state[-1]
        _state = _state[:-1]


        return {
            "scalar": np.concatenate((_state, [self.chair_angle/90, self.sampled_height_diff], [power])),
            "terrain": None,
        }
    
    def sample_power(self, std=0.02):
        if np.abs(self.power_base - self.power_end) <= 1e-3 or self.original:
            self.power = self.power_base
            return
        self.power = np.clip(self.power_base + np.random.randn() * std, self.power_end, 1)

    def adjust_power(self, test_reward, threshold=40):
        if self.original or (np.abs(self.power_base - self.power_end) <= 1e-3 and self.chair_angle_mean==0 and self.chair_angle_range==20 and self.height_diff == 0.1):
            return False
        if test_reward > threshold:
            self.power_base = max(0.95 * self.power_base, self.power_end)
            # self.chair_angle_mean = np.clip(self.chair_angle_mean-3, 0, 0)
            self.chair_angle_range = np.clip(self.chair_angle_range*1.1, 0, 20)
            self.height_diff = np.clip(self.height_diff + 0.02, 0, 0.1)
        return np.abs(self.power_base - self.power_end) <= 1e-3 and self.chair_angle_mean==0 and self.chair_angle_range==20 and self.height_diff == 0.1

    def sample_angle(self):
        self.chair_angle = np.clip((np.random.rand() * 2 - 1) * self.chair_angle_range + self.chair_angle_mean,-20,20)
        self.sampled_height_diff = np.clip((np.random.rand() * 2 - 1) * self.height_diff,-0.1,0.1)
        return self.chair_angle, self.sampled_height_diff

    def set_chair_parameters(self, mean, chair_range, height_diff):
        self.chair_angle_mean = mean
        self.chair_angle_range = chair_range
        self.height_diff = height_diff

    # @property
    # def _reward(self):
    #     # upright = rewards.tolerance(self.physics.torso_upright(),
    #     #                             bounds=(0.9, float('inf')), sigmoid='linear',
    #     #                             margin=1.9, value_at_margin=0)

    #     # return self._standing * upright * self._dont_move
    #     min_x_pos = min(self.physics.named.data.xpos[:,0])
    #     bench_x_proj = self.bench_width / 2 * np.cos(self.chair_angle/180*np.pi)
    #     face_forward = rewards.tolerance(self.physics.named.data.xmat['torso', 'xx'],
    #                         bounds=(0.8, float('inf')), sigmoid='linear',
    #                         margin=1.8, value_at_margin=0)
    #     leave_bench = rewards.tolerance(min_x_pos,
    #                         bounds=(bench_x_proj-0.12, float('inf')), sigmoid='gaussian',
    #                         margin=bench_x_proj*2+0.05, value_at_margin=0.1)
    #     return super()._reward * (1.0 + face_forward) / 2

class CheetahEnv():

    def __init__(self, args, seed):
        self.env = suite.load(domain_name="cheetah", task_name="run", task_kwargs={'random': seed})
        self.reset_count = 0
        self.count = 0
        self.power_base = self.power = 0
        self.curriculum_finished = True
        self.args = args
        self.env._flat_observation = True
        self.physics = self.env.physics
        self.action_space = self.env.action_spec()
        self.reset()
        self.obs_shape = self._state["scalar"].shape

    def step(self, a, test_time=False):
        self._step_num += 1
        self.timestep = self.env.step(a)
        self.obs = self.timestep.observation
        if test_time: self.images.append(self.render())
        return self._state, self._reward, self._done, None

    def reset(self, test_time=False):
        self._step_num = 0
        self.reset_count += 1
        if test_time: self.images = []
        self.env.reset()

        self.obs = self.env._task.get_observation(self.env._physics)
        self.terminal_signal = False        
        return self._state

    def render(self, mode=None):
        return self.env.physics.render(height=128, width=128, camera_id=1)

    def adjust_power(self, test_reward, replay_buffer, threshold=40):
        return

    def export_power(self):
        return

    def set_power(self, power_dict):
        return
        
    @property
    def _state(self):
        _state = []
        for part in self.obs.values():
            _state.append(part if not np.isscalar(part) else [part])
        return {
            "scalar": np.concatenate(_state),
            "terrain": None,
        }

    @property
    def _done(self):
        if self._step_num >= 1000:
            return True
        return False

    @property
    def _reward(self):
        return rewards.tolerance(self.physics.speed(),
                             bounds=(10, float('inf')),
                             margin=10,
                             value_at_margin=0,
                             sigmoid='linear')