from gym.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv
from gym.utils import seeding
from dm_control import suite
from dm_control.suite.base import Task
from dm_control.utils import rewards
import os
import numpy as np
import math

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
            "power": self.power
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
            reward -= math.pow(action[0] * (self.power/0.0015), 2) * 0.1 * ((0.0015/self.power) ** 2)
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
    _STAND_HEIGHT = 1.59

    def __init__(self, original, power=0.9, seed=0, custom_reset=False, power_end=0.35):
        self.env = suite.load(domain_name="humanoid", task_name="stand", task_kwargs={'random':seed})
        self.env._flat_observation = True
        self.physics = self.env.physics
        self.custom_reset = custom_reset
        self.power_end=power_end
        self.power_base = power
        self.reset()
        self.action_space = self.env.action_spec()
        self.obs_shape = self._state.shape
        self.original = original
        
    def step(self, a):
        self._step_num += 1
        self.timestep = self.env.step(a)
        self.obs = self.timestep.observation
        return self._state, self._reward, self._done, self.timestep

    def reset(self, test_time=False):
        if self.custom_reset:
            # while True:
            #     self.physics.after_reset()
            #     if self.physics.data.ncon <= 0: break
            # self.physics(self, self.physics)
            repeat = True
            while repeat:
                self.env.reset()
                with self.physics.reset_context():
                    self.physics.named.data.qpos[:3] = [0,0,0.5]
                    self.physics.named.data.qpos[3:7] = [0.707,0,-1,0]
                    self.physics.after_reset()
                if self.physics.data.ncon == 0: repeat=False
        else:
            self.timestep = self.env.reset()
        # with self.env.physics.reset_context():
        #     # Add custom reset
        self.obs = self.env._task.get_observation(self.env._physics)
        self._step_num = 0
        self.terminal_signal = False
        if not test_time: self.sample_power()
        return self._state
    
    def render(self, mode=None):
        return self.env.physics.render(height=128, width=128, camera_id = 0)

    def sample_power(self, std=0.02):
        if np.abs(self.power_base - self.power_end) <= 1e-3:
            self.power = self.power_base
            return
        self.power = np.clip(self.power_base + np.random.randn() * std, self.power_end, 1)

    def adjust_power(self, test_reward):
        if self.original or np.abs(self.power_base - self.power_end) <= 1e-3:
            return False
        if test_reward > 180:
            self.power_base = max(0.95 * self.power_base, self.power_end)
        return np.abs(self.power_base - self.power_end) <= 1e-3

    def export_power(self):
        return {
            "power": self.power_base
        }

    def set_power(self, power_dict):
        self.power = power_dict["power"]
    
    @property
    def _state(self):
        _state = []
        for part in self.obs.values():
            _state.append(part if not np.isscalar(part) else [part])
        _state.append([self.power])
        return np.concatenate(_state)

    @property
    def _done(self):
        if self._step_num >= 1000:
            return True
        return self.timestep.last()

    @property
    def _reward(self):
        standing = rewards.tolerance(self.physics.head_height(),
                                 bounds=(self._STAND_HEIGHT, float('inf')),
                                 margin=self._STAND_HEIGHT/4)
        upright = rewards.tolerance(self.physics.torso_upright(),
                                bounds=(0.9, float('inf')), sigmoid='linear',
                                margin=1.9, value_at_margin=0)
        
        small_control = rewards.tolerance(self.physics.control(), margin=1,
                                      value_at_margin=0,
                                      sigmoid='quadratic').mean()
        small_control = (4 + small_control) / 5

        horizontal_velocity = self.physics.center_of_mass_velocity()[[0, 1]]
        dont_move = rewards.tolerance(horizontal_velocity, bounds=[-0.3,0.3], margin=1.2).mean()

        # joint_limit = self.physics.model.jnt_range[1:]
        # joint_angle = self.physics.data.qpos[7:].copy()
        # dif = 0.01 * (joint_limit[:,1] - joint_limit[:,0])
        # between_limit = np.logical_and(joint_angle>joint_limit[:,0] + dif, joint_angle<joint_limit[:,1] - dif)
        # joint_limit_cost = np.where(between_limit, 1, 0).mean()

        return standing*upright*small_control*dont_move
