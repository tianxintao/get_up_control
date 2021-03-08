from gym.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv
from gym.utils import seeding
from dm_control import suite
from dm_control.suite.base import Task
from dm_control.utils import rewards
import os
import numpy as np
import math
import imageio


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

    def __init__(self, original, power=1.0, seed=0, custom_reset=False, power_end=0.35):
        self.env = suite.load(domain_name="humanoid", task_name="stand", task_kwargs={'random': seed})
        self.env._flat_observation = True
        self.physics = self.env.physics
        self.custom_reset = custom_reset
        self.power_end = power_end
        self.original = original
        self.power_base = power
        self.reset()
        self.action_space = self.env.action_spec()
        self.obs_shape = self._state.shape
        self.physics.reload_from_xml_path('data/humanoid_static.xml')

    def step(self, a):
        self._step_num += 1
        if self.original: a = a * self.power
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
                    self.physics.named.data.qpos[:3] = [0, 0, 0.5]
                    self.physics.named.data.qpos[3:7] = [0.707, 0, -1, 0]
                    self.physics.after_reset()
                if self.physics.data.ncon == 0: repeat = False
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
        return self.env.physics.render(height=128, width=128, camera_id=0)

    def sample_power(self, std=0.02):
        if np.abs(self.power_base - self.power_end) <= 1e-3 or self.original:
            self.power = self.power_base
            return
        self.power = np.clip(self.power_base + np.random.randn() * std, self.power_end, 1)

    def adjust_power(self, test_reward):
        if self.original or np.abs(self.power_base - self.power_end) <= 1e-3:
            return False
        if test_reward > 90:
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
    def _standing(self):
        return rewards.tolerance(self.physics.head_height(),
                                 bounds=(self._STAND_HEIGHT, float('inf')),
                                 margin=self._STAND_HEIGHT / 4)

    @property
    def _small_control(self):
        control_val = rewards.tolerance(self.physics.control(), margin=1,
                                        value_at_margin=0,
                                        sigmoid='quadratic').mean()
        return (4 + control_val) / 5

    @property
    def _dont_move(self):
        horizontal_velocity = self.physics.center_of_mass_velocity()[[0, 1]]
        return rewards.tolerance(horizontal_velocity, bounds=[-0.3, 0.3], margin=1.2).mean()

    @property
    def _reward(self):
        upright = rewards.tolerance(self.physics.torso_upright(),
                                    bounds=(0.9, float('inf')), sigmoid='linear',
                                    margin=1.9, value_at_margin=0)

        joint_limit = self.physics.model.jnt_range[1:]
        joint_angle = self.physics.data.qpos[7:].copy()
        between_limit = np.logical_and(joint_angle>joint_limit[:,0], joint_angle<joint_limit[:,1])
        joint_limit_cost = np.where(between_limit, 1, 0).mean()

        return self._standing * upright * self._standing * self._dont_move


class HumanoidStandupRandomEnv(HumanoidStandupEnv):
    random_terrain_path = './data/terrain.png'
    max_height = 0.3
    # slope_terrain_path = './data/slope.png'
    xml_path = './data/humanoid.xml'

    def __init__(self, original, power=1.0, seed=0, custom_reset=False, power_end=0.35):
        HumanoidStandupEnv.__init__(self, original, power, seed, custom_reset, power_end)
        # self.create_random_terrain()
        # self.create_slope_terrain()
        self.create_random_terrain()
        self.physics.reload_from_xml_path(self.xml_path)

    def create_random_terrain(self):
        image = np.random.random_sample((60, 60))
        imageio.imwrite(self.random_terrain_path, image)

    def reset(self, test_time=False):
        self.create_random_terrain()
        self.physics.reload_from_xml_path(self.xml_path)
        return super.reset(test_time=test_time)

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

    def get_terrain_height(self):
        terrain = imageio.imread(self.slope_terrain_path)
        x, y, z = self.physics.center_of_mass_position()
        x_ind = int((x + 10) / 20 * self.terrain_shape)
        y_ind = int((y + 10) / 20 * self.terrain_shape)
        # collect the hightfield data from the nearby 5x5 region
        height = terrain[y_ind-2:y_ind+3,x_ind-2:x_ind+3].mean() * self.max_height
        return height

    @property
    def _standing(self):
        # print(self.physics.head_height()-self.get_terrain_height())
        return rewards.tolerance(self.physics.head_height()-self.get_terrain_height(),
                                 bounds=(self._STAND_HEIGHT, float('inf')),
                                 margin=self._STAND_HEIGHT / 4)

class HumanoidBenchEnv(HumanoidStandupEnv):
    _STAND_HEIGHT = 1.55
    _bench_height = 0.3
    bench_center = np.array([0, 0, _bench_height])
    xml_path = './data/humanoid_bench.xml'

    def __init__(self, original, power=1.0, seed=0, custom_reset=False, power_end=0.4):
        self.default_qpos = None
        HumanoidStandupEnv.__init__(self, original, power, seed, custom_reset, power_end)
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

        return np.concatenate(_state)

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
                                 margin=self._STAND_HEIGHT / 2)







































# class HumanoidBalanceEnv(HumanoidStandupEnv):
#     _STAND_HEIGHT = 1.4
#
#     def __init__(self, original, power=1.0, seed=0, custom_reset=False, power_end=0.4):
#         self.power = power
#         HumanoidStandupEnv.__init__(self, original, power, seed, custom_reset, power_end)
#
#     def reset(self, test_time=None):
#
#         repeat = True
#         while repeat:
#             with self.physics.reset_context():
#                 self.env.reset()
#                 self.physics.named.data.qpos[:3] = [0, 0, 1.5]
#                 self.physics.named.data.qpos[3:7] = [1, 0, 0, 0]
#                 self.physics.after_reset()
#             if self.physics.data.ncon == 0: repeat = False
#
#         self.obs = self.env._task.get_observation(self.physics)
#         self._step_num = 0
#         self.terminal_signal = False
#
#         return self._state
#
#     @property
#     def _standing(self):
#         return rewards.tolerance(self.physics.head_height(),
#                                  bounds=(self._STAND_HEIGHT, float('inf')),
#                                  margin=self._STAND_HEIGHT / 4)
#
#     @property
#     def _dont_move(self):
#         horizontal_velocity = self.physics.center_of_mass_velocity()[[0, 1]]
#         return rewards.tolerance(horizontal_velocity, margin=2).mean()
#
#     @property
#     def _done(self):
#         if self._step_num >= 1000:
#             return True
#         if self.physics.center_of_mass_position()[2] < 0.65:
#             self.terminal_signal = True
#             return True
#         return self.timestep.last()
