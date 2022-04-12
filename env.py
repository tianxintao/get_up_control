import copy

import numpy as np
from dm_control import suite
from dm_control.utils import rewards

import utils

STANDING_POSE_NOARM = [0, -0.11665, 0, -0.03601, -0.03219, 0.05025,
                       -0.39803, -0.30198, 0.13615, -0.03601, -0.03219, 0.05025,
                       -0.39803, -0.30198, -0.13615,
                       -0.40763, 0.41967, -1.5795]
STANDING_POSE_DISABLED = [0, -0.11665, 0, -0.03601, -0.03219, 0.05025,
                          -0.39803, -0.30198, 0.13615, -0.03601, -0.03219, 0.05025,
                          -0.30198, -0.13615, 0.40763, -0.41967,
                          -0.40763, 0.41967, -1.5795]
STANDING_POSE = [0, -0.11665, 0, -0.03601, -0.03219, 0.05025,
                 -0.39803, -0.30198, 0.13615, -0.03601, -0.03219, 0.05025,
                 -0.39803, -0.30198, -0.13615, 0.40763, -0.41967, -1.5795,
                 -0.40763, 0.41967, -1.5795]
# std: 0.00431
COM_HEIGHT = 0.80326


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
        self.action_space = self.env.action_spec().shape[0]
        self.reset()
        self.obs_shape = self._state.shape[0]
        self.physics.reload_from_xml_path('data/humanoid_static.xml')
        self.buf = None

    def step(self, a, test_time=False):
        self._step_num += 1
        self.env.step(a)
        self.obs = self.env._task.get_observation(self.physics)
        return self._state, self._reward, self._done, None

    def reset(self, test_time=False, store_buf=False, speed=None):
        self._step_num = 0
        self._initial_steps = True
        if not test_time: self.sample_power()
        self.env.reset()
        self.obs = self.env._task.get_observation(self.physics)
        current_state = self._state
        self.starting_images = []
        if self.args.to_file:
            self.geom_names = self.env.physics.named.data.geom_xpos.axes.row.names
            self.starting_geoms = {"state": []}
            for name in self.geom_names:
                self.starting_geoms[name + "_pos"] = []
                self.starting_geoms[name + "_angleaxis"] = []
        for _ in range(80):
            # action = np.random.normal(loc=0.0, scale=0.1, size=self.action_space)
            action = np.zeros(self.action_space)
            self.env.step(action)
            self.obs = self.env._task.get_observation(self.physics)
            if self.args.to_file:
                self.starting_geoms["state"].append(self._state[:-1])
                for name in self.geom_names:
                    self.starting_geoms[name + "_pos"].append(self.env.physics.named.data.geom_xpos[name].copy())
                    self.starting_geoms[name + "_angleaxis"].append(
                        utils.rotmatrix_to_angleaxis(self.env.physics.named.data.geom_xmat[name].copy().reshape(3, 3)))
            if True: self.starting_images.append(self.render())
            if store_buf:
                self.buf.add(current_state, action, self._state, self._reward, self.terminal_signal)
                current_state = self._state
        self._initial_steps = False
        self.terminal_signal = False
        return self._state

    def render(self, mode=None):
        return self.env.physics.render(height=384, width=384, camera_id=1)

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
    def _upright(self):
        if self.physics.center_of_mass_position()[2] < 0.5:
            return 1.0
        upright = rewards.tolerance(self.physics.torso_upright(),
                                    bounds=(0.9, float('inf')), sigmoid='linear',
                                    margin=1.9, value_at_margin=0)

        return upright

    @property
    def _reward(self):
        return self._upright * self._standing * self._dont_move * self._closer_feet


class HumanoidStandupVelocityEnv(HumanoidStandupEnv):
    qpos_to_ctrl_index = np.array([1, 0, 2, 3, 4, 5, 6, 8, 7, 9, 10, 11, 12, 14, 13, 15, 16, 17, 18, 19, 20])
    standing_pose = STANDING_POSE
    state_length = 67

    def __init__(self, args, seed):
        self.args = args
        self.teacher_policy = None
        if args.teacher_student:
            self.teacher_env = HumanoidStandupEnv(args, seed)
            self.render_env = suite.load(domain_name="humanoid", task_name="stand", task_kwargs={'random': seed})
            self.teacher_policy = None
        HumanoidStandupEnv.__init__(self, args, seed)
        self.physics.reload_from_xml_path('data/humanoid_high_freq.xml')

    def set_teacher_policy(self, policy):
        self.teacher_policy = policy

    def run_one_episode(self, test_time=False):
        self.teacher_env.power = self.args.teacher_power
        while True:
            self.reset_teacher_trajectory()
            end_next_step = False
            state, done = self.teacher_env.reset(test_time=True), False
            if len(self.teacher_env.starting_images) > 0:
                self.starting_images = np.concatenate((self.teacher_env.starting_images,
                                                       self.teacher_env.starting_images,
                                                       self.teacher_env.starting_images), axis=2)
            if self.args.to_file:
                self.geom_names = self.teacher_env.geom_names
                self.starting_geoms = self.teacher_env.starting_geoms
                self.teacher_geoms = copy.deepcopy(self.starting_geoms)
            self.teacher_initial_state["qpos"] = self.teacher_env.physics.data.qpos.copy()
            self.teacher_initial_state["qvel"] = self.teacher_env.physics.data.qvel.copy()
            self.teacher_episode_length = 0

            while not done:
                action = self.teacher_policy.select_action(state)
                state, _, done, _ = self.teacher_env.step(action, test_time=True)
                self.teacher_episode_length += 1
                self.teacher_traj["com"].append(self.teacher_env.physics.center_of_mass_position().copy())
                self.teacher_traj["com_ori"].append(self.teacher_env.physics.torso_vertical_orientation().copy())
                self.teacher_traj["qpos"].append(self.teacher_env.physics.data.qpos.copy())
                self.teacher_traj["qvel"].append(self.teacher_env.physics.data.qvel.copy())
                if self.args.to_file:
                    self.teacher_geoms["state"].append(self.teacher_env._state[:-1])
                    for name in self.geom_names:
                        self.teacher_geoms[name + "_pos"].append(
                            self.teacher_env.physics.named.data.geom_xpos[name].copy())
                        self.teacher_geoms[name + "_angleaxis"].append(utils.rotmatrix_to_angleaxis(
                            self.teacher_env.physics.named.data.geom_xmat[name].copy().reshape(3, 3)))

                if test_time: self.teacher_images.append(self.teacher_env.render())

                if end_next_step: break

                if self.teacher_env.physics.head_height() > 1.4:
                    end_next_step = True

            if end_next_step: break

        for (key, val) in self.teacher_traj.items():
            self.teacher_traj[key] = np.array(val)

        self.modify_teacher_traj()

    def modify_teacher_traj(self):
        # k = 30
        # i = min(self.teacher_episode_length - 11, self.teacher_episode_length - 1)
        # for (key, val) in self.teacher_traj.items():
        #     repeated_attributes = self.teacher_traj[key][i].copy()
        #     for _ in range(k):
        #         self.teacher_traj[key] = np.insert(self.teacher_traj[key], i, repeated_attributes, axis=0)
        # self.teacher_episode_length += k
        pass

    def reset_teacher_trajectory(self):
        self.teacher_traj = {
            "com": [],
            "qpos": [],
            "qvel": [],
            "com_ori": [],
        }
        self.teacher_images = []
        self.teacher_initial_state = {}

    def reset(self, test_time=False, store_buf=False, speed=None):
        self._step_num = 0
        self.power = 1.0

        if not test_time or (speed == None):
            self.speed = np.random.uniform(low=self.args.slow_speed, high=self.args.fast_speed)
        else:
            self.speed = speed

        self.env.reset()

        if self.teacher_policy == None:
            self.teacher_episode_length = 0
            self.interpolated_trajectory = {}
            self.interpolated_trajectory["qpos"] = self.teacher_env.physics.data.qpos.copy()[None, :]
            self.interpolated_trajectory["com"] = self.teacher_env.physics.center_of_mass_position()[None, :]
            self.interpolated_trajectory["com_ori"] = self.teacher_env.physics.torso_vertical_orientation()[None, :]
        else:
            while True:
                self.run_one_episode(test_time=test_time)
                if self.args.to_file: self.geom_traj = copy.deepcopy(self.starting_geoms)
                self.interpolated_trajectory = utils.interpolate_motion(self.teacher_traj, self.speed, uneven=False)
                self.teacher_episode_length = self.interpolated_trajectory["qpos"].shape[0]
                if not test_time and np.random.uniform(0, 1) > 0.2:
                    self._step_num = np.random.randint(0, self.teacher_episode_length)
                self.interpolated_trajectory["qpos"] = np.append(
                    self.interpolated_trajectory["qpos"],
                    [np.append([0, 0, 1.5, 0, 0, 0, 0], self.standing_pose)],
                    axis=0)
                self.interpolated_trajectory["com"] = np.append(self.interpolated_trajectory["com"],
                                                                [[0, 0, 1.4]], axis=0)
                self.interpolated_trajectory["com_ori"] = np.append(self.interpolated_trajectory["com_ori"],
                                                                    [[0, 0, 1.0]], axis=0)
                with self.physics.reset_context():
                    self.physics.data.qpos[:] = self.interpolated_trajectory["qpos"][self._step_num]
                    self.physics.data.qvel[:] = self.interpolated_trajectory["qvel"][self._step_num]
                    self.physics.after_reset()

                if np.abs(self.env.physics.center_of_mass_position()[2] -
                          self.interpolated_trajectory["com"][self._step_num][2]) < 0.1:
                    break

        self.obs = self.env._task.get_observation(self.env._physics)
        self.terminal_signal = False

        return self._state

    def step(self, a, test_time=False):
        self.timestep = self.substep(a)
        self.obs = self.env._task.get_observation(self.physics)
        self._step_num += 1

        if self.args.to_file:
            self.geom_traj["state"].append(self._state[:self.state_length])
            for name in self.geom_names:
                self.geom_traj[name + "_pos"].append(self.env.physics.named.data.geom_xpos[name].copy())
                self.geom_traj[name + "_angleaxis"].append(
                    utils.rotmatrix_to_angleaxis(self.env.physics.named.data.geom_xmat[name].copy().reshape(3, 3)))

        return self._state, self._reward, self._done, None

    def substep(self, action):
        for _ in range(4 * self.env._n_sub_steps):
            if self._step_num < self.teacher_episode_length:
                pose_diff = \
                    ((self.interpolated_trajectory["qpos"][self._step_num][7:] + action) - self.physics.data.qpos[7:])[
                        self.qpos_to_ctrl_index]
                kp = np.ones_like(self.physics.model.actuator_gear[:, 0])
                kd = kp / 10
                final_action = kp * pose_diff - ((kd * self.physics.data.qvel[6:])[self.qpos_to_ctrl_index])
                self.env._task.before_step(final_action, self.env._physics)
                self.env._physics.step()
            else:
                pose_diff = ((self.standing_pose + action) - self.physics.data.qpos[7:])[self.qpos_to_ctrl_index]
                kp = np.ones_like(self.physics.model.actuator_gear[:, 0])
                kd = kp / 5
                final_action = kp * pose_diff - ((kd * self.physics.data.qvel[6:])[self.qpos_to_ctrl_index])
                self.env._task.before_step(final_action, self.env._physics)
                self.env._physics.step()
        self.env._task.after_step(self.env._physics)

    def get_target_pose(self, index):
        joint_angles = self.interpolated_trajectory["qpos"][index][7:]
        com_height = self.interpolated_trajectory["com"][index][2]
        com_orientation = self.interpolated_trajectory["com_ori"][index]
        return np.concatenate((joint_angles, [com_height], com_orientation))

    @property
    def _done(self):
        if self._step_num >= self.teacher_episode_length and self.physics.center_of_mass_position()[2] < 0.5:
            self.terminal_signal = True
            return True
        if self._step_num < self.teacher_episode_length and np.abs(
                self.env.physics.center_of_mass_position()[2] - self.interpolated_trajectory["com"][self._step_num - 1][
                    2]) > 0.5:
            self.terminal_signal = True
            return True
        if self._step_num < (self.teacher_episode_length + 150):
            return False
        return True

    @property
    def _state(self):
        _state = []
        for part in self.obs.values():
            _state.append(part if not np.isscalar(part) else [part])
        _state.append(self.get_target_pose(min(self._step_num, self.teacher_episode_length)))
        _state.append(self.get_target_pose(min(self._step_num + 5, self.teacher_episode_length)))
        _state.append([self.power])
        return np.concatenate(_state)

    @property
    def _dont_move(self):
        horizontal_velocity = self.physics.center_of_mass_velocity()[[0, 1]]
        return rewards.tolerance(horizontal_velocity, bounds=[-0.0, 0.0], margin=1.2).mean()

    @property
    def _reward(self):
        index = self._step_num - 1

        current_leg_speed = np.array([self.physics.data.qvel[11], self.physics.data.qvel[17]])
        if self._step_num < self.teacher_episode_length:
            target_leg_speed = np.array(
                [self.interpolated_trajectory["qvel"][index][11], self.interpolated_trajectory["qvel"][index][17]])
            velocity_dist = rewards.tolerance((current_leg_speed - target_leg_speed),
                                              bounds=(-0.5, 0.5),
                                              margin=1.3).mean()
            velocity_dist = (2 + velocity_dist) / 3
            com_dist = rewards.tolerance(
                (self.interpolated_trajectory["com"][index][2] - self.physics.center_of_mass_position()[2]),
                bounds=(0.0, 0.0),
                margin=0.5)
            ori_dist = rewards.tolerance(
                (self.interpolated_trajectory["com_ori"][index] - self.physics.torso_vertical_orientation()),
                bounds=(-0.02, 0.02),
                margin=0.6,
                value_at_margin=0.3).prod()

            return velocity_dist * com_dist * ori_dist
        else:
            rotation_dist = ((self.standing_pose - self.physics.data.qpos[7:]) ** 2).sum() / 4
            com_dist = ((COM_HEIGHT - self.physics.center_of_mass_position()[2]) ** 2) * 10
            return self._upright * self._dont_move * np.exp(-com_dist) * np.exp(-rotation_dist)

    def render(self, mode=None):
        image_l = self.physics.render(height=384, width=384, camera_id=1)
        with self.render_env.physics.reset_context():
            self.render_env.physics.data.qpos[:] = self.interpolated_trajectory["qpos"][
                min(self._step_num - 1, len(self.interpolated_trajectory["qpos"]) - 1)]
            self.render_env.physics.data.qvel[:] = self.interpolated_trajectory["qvel"][
                min(self._step_num - 1, len(self.interpolated_trajectory["qvel"]) - 1)]
            self.render_env.physics.after_reset()
        image_m = self.render_env.physics.render(height=384, width=384, camera_id=1)
        image_r = self.teacher_images[min(self._step_num - 1, len(self.teacher_images) - 1)]
        return np.concatenate((image_l, image_m, image_r), axis=1)


class HumanoidVariantStandupEnv(HumanoidStandupEnv):

    def __init__(self, args, seed):
        super().__init__(args, seed)
        if args.variant == "Disabled":
            self.physics.reload_from_xml_path('data/humanoid_disabled.xml')
            self.obs_shape -= 4
            self.action_space -= 2
        elif args.variant == "Noarm":
            self.physics.reload_from_xml_path('data/humanoid_noarm.xml')
            self.obs_shape -= 6
            self.action_space -= 3


class HumanoidVariantStandupVelocityEnv(HumanoidStandupVelocityEnv, HumanoidVariantStandupEnv):

    def __init__(self, args, seed):
        self.args = args
        self.teacher_policy = None
        if args.teacher_student:
            self.teacher_env = HumanoidVariantStandupEnv(args, seed)
            self.render_env = HumanoidVariantStandupEnv(args, seed + 10)
            self.teacher_policy = None
        HumanoidVariantStandupEnv.__init__(self, args, seed)
        if args.variant == "Disabled":
            self.physics.reload_from_xml_path('data/humanoid_disabled_high_freq.xml')
            self.qpos_to_ctrl_index = np.array([1, 0, 2, 3, 4, 5, 6, 8, 7, 9, 10, 11, 13, 12, 14, 15, 16, 17, 18])
            self.standing_pose = STANDING_POSE_DISABLED
            self.state_length = 63
        elif args.variant == "Noarm":
            self.physics.reload_from_xml_path('data/humanoid_noarm_high_freq.xml')
            self.qpos_to_ctrl_index = np.array([1, 0, 2, 3, 4, 5, 6, 8, 7, 9, 10, 11, 12, 14, 13, 15, 16, 17])
            self.standing_pose = STANDING_POSE_NOARM
            self.state_length = 61

    def set_teacher_policy(self, policy):
        HumanoidStandupVelocityEnv.set_teacher_policy(self, policy)

    def reset(self, test_time=False, store_buf=False, speed=None):
        return HumanoidStandupVelocityEnv.reset(self, test_time=test_time, store_buf=store_buf, speed=speed)

    def step(self, a, test_time=False):
        return HumanoidStandupVelocityEnv.step(self, a, test_time=test_time)

    def render(self, mode=None):
        return HumanoidStandupVelocityEnv.render(self, mode=mode)
