from gym.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv
from gym.utils import seeding
from dm_control import suite, mujoco
from dm_control.suite.base import Task
from dm_control.utils import rewards
from dm_control.suite.humanoid import Humanoid, Physics
from dm_control.rl import control
from dm_control.suite import common
import os
import numpy as np
import math
import imageio
import utils


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

    def __init__(self, args, seed):
        self.env = suite.load(domain_name="humanoid", task_name="stand", task_kwargs={'random': seed})
        self.args = args
        self.env._flat_observation = True
        self.physics = self.env.physics
        self.custom_reset = args.custom_reset
        self.power_end = args.power_end
        self.original = args.original
        self.power_base = args.power
        self.reset()
        self.action_space = self.env.action_spec()
        self.obs_shape = self._state["scalar"].shape
        self.physics.reload_from_xml_path('data/humanoid_static.xml')

    def step(self, a):
        self._step_num += 1
        if self.original: a = a * self.power
        self.timestep = self.env.step(a)
        self.obs = self.timestep.observation
        mujoco.engine.mjlib.mj_rnePostConstraint(self.physics.model.ptr, self.physics.data.ptr)
        # print(self._reaction_force, self.physics.data.ncon)
        reaction_force = self._reaction_force if self.args.predict_force else None
        return self._state, self._reward, self._done, reaction_force

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
                    self.physics.named.data.qpos[:3] = [0, 0, 0.8]
                    self.physics.named.data.qpos[3:7] = [0.7071, 0.7071, 0, 0]
                    self.physics.after_reset()
                if self.physics.data.ncon == 0: repeat = False
        else:
            self.timestep = self.env.reset()
        # with self.env.physics.reset_context():
        #     # Add custom reset
        self.obs = self.env._task.get_observation(self.env._physics)
        self._step_num = 0
        self.terminal_signal = False
        # self.velocity_record = []
        if not test_time: self.sample_power()
        return self._state

    def render(self, mode=None):
        return self.env.physics.render(height=128, width=128, camera_id=1)

    def sample_power(self, std=0.02):
        if np.abs(self.power_base - self.power_end) <= 1e-3 or self.original:
            self.power = self.power_base
            return
        self.power = np.clip(self.power_base + np.random.randn() * std, self.power_end, 1)

    def adjust_power(self, test_reward, threshold=90):
        if self.original or np.abs(self.power_base - self.power_end) <= 1e-3:
            return False
        if test_reward > threshold:
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
        # print("velocity_z:{}, velocity:{}".format(np.concatenate(_state)[42],np.abs(np.concatenate(_state)[40:67]).mean()))
        return {
            "scalar": np.concatenate(_state),
            "terrain": None,
        }

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
        upright = rewards.tolerance(self.physics.torso_upright(),
                                    bounds=(0.9, float('inf')), sigmoid='linear',
                                    margin=1.9, value_at_margin=0)

        return self._standing * upright * self._dont_move


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


class Humanoid2DStandupEnv(HumanoidStandupEnv):
    def __init__(self, args, seed):
        physics = Physics.from_xml_path('./data/humanoid_2D.xml')
        task = Humanoid(move_speed=0, pure_state=False)
        environment_kwargs = {}
        self.env = control.Environment(
                        physics, task, time_limit=1000, control_timestep=.025,
                        **environment_kwargs)
        self.args = args
        self.env._flat_observation = True
        self.physics = self.env.physics
        self.custom_reset = args.custom_reset
        self.power_end = args.power_end
        self.original = args.original
        self.power_base = args.power
        self.reset()
        self.action_space = self.env.action_spec()
        self.obs_shape = self._state["scalar"].shape


class HumanoidStandupRandomEnv(HumanoidStandupEnv):
    random_terrain_path = './data/terrain.png'
    max_height = 0.3
    terrain_shape = 60
    # slope_terrain_path = './data/slope.png'
    xml_path = './data/humanoid.xml'

    def __init__(self, args, seed):
        HumanoidStandupEnv.__init__(self, args, seed)
        # self.create_random_terrain()
        # self.create_slope_terrain()
        self.create_random_terrain()
        self.physics.reload_from_xml_path(self.xml_path)

    def create_random_terrain(self):
        self.terrain = np.random.random_sample((self.terrain_shape, self.terrain_shape))
        imageio.imwrite(self.random_terrain_path, (self.terrain * 255).astype(np.uint8))

    def reset(self, test_time=False):
        self.create_random_terrain()
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
        x_ind = int((x + 10) / 20 * self.terrain_shape)
        y_ind = int((y + 10) / 20 * self.terrain_shape)
        # collect the hightfield data from the nearby 5x5 region
        self.terrain_profile = self.terrain[y_ind-1:y_ind+2,x_ind-1:x_ind+2]
        # height = self.terrain_profile.mean() * self.max_height
        return self.terrain[y_ind-half_size:y_ind+half_size+1,x_ind-half_size:x_ind+half_size+1]

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


class HumanoidChairEnv(HumanoidBenchEnv):
    xml_path = './data/humanoid_chair.xml'

    def __init__(self, args, seed):
        self.chair_angle_mean = 30
        self.chair_angle_range = 3
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
        self.physics.reload_from_xml_string(
            self.model_str.format(chair_angle=self.sample_angle()),
            common.ASSETS
        )
        return super().reset(test_time=test_time)

    @property
    def _state(self):
        _state = super()._state["scalar"]
        power = _state[-1]
        _state = _state[:-1]


        return {
            "scalar": np.concatenate((_state, [self.chair_angle/20], [power])),
            "terrain": None,
        }
    
    def sample_power(self, std=0.02):
        if np.abs(self.power_base - self.power_end) <= 1e-3 or self.original:
            self.power = self.power_base
            return
        self.power = np.clip(self.power_base + np.random.randn() * std, self.power_end, 1)

    def adjust_power(self, test_reward, threshold=40):
        if self.original or np.abs(self.power_base - self.power_end) <= 1e-3:
            return False
        if test_reward > threshold:
            self.power_base = max(0.95 * self.power_base, self.power_end)
            self.chair_angle_mean = np.clip(self.chair_angle_mean-3, 0, 30)
            self.chair_angle_range = np.clip(self.chair_angle_range*1.1, 0, 20)
        return np.abs(self.power_base - self.power_end) <= 1e-3

    def sample_angle(self):
        self.chair_angle = np.random.rand() * self.chair_angle_range + self.chair_angle_mean
        return self.chair_angle

    def set_chair_parameters(self, mean, chair_range):
        self.chair_angle_mean = mean
        self.chair_angle_range = chair_range