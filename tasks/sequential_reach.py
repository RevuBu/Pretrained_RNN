import numpy as np
import motornet as mn
import gymnasium as gym
import torch as th
from typing import Any, Union
from .kinematics import minjerk, comMiniJerk, joints_to_hand, hand_to_joints


class RandomTargetReach:
    def __init__(self, **kwargs):
        self.joint_range = [[0.0, 140.0], [0.0, 160.0]]
        self.aparams = {"l1": 0.309, "l2": 0.333}
        self.dt = 0.01
        self.workspace = [-0.3, 0.3, 0.2, 0.5]

    def rand_joint_state(self, batch_size):
        sho_joint = np.deg2rad(
            np.random.uniform(self.joint_range[0][0], self.joint_range[0][1], size=5 * batch_size)
        )
        elb_joint = np.deg2rad(
            np.random.uniform(self.joint_range[1][0], self.joint_range[1][1], size=5 * batch_size)
        )
        joint_state = np.column_stack((sho_joint, elb_joint))
        H, E = joints_to_hand(joint_state, self.aparams)
        x_min, x_max, y_min, y_max = self.workspace
        mask = (
            (H[:, 0] >= x_min) & (H[:, 0] <= x_max) & (H[:, 1] >= y_min) & (H[:, 1] <= y_max)
        )
        joint_state = joint_state[mask]
        H = H[mask]
        if H.shape[0] >= batch_size:
            return joint_state[:batch_size], H[:batch_size]
        else:
            return self.rand_joint_state(batch_size)

    def distance(self, start_postion, goal):
        dis = goal - start_postion
        dis = np.sqrt(dis[:, 0] ** 2 + dis[:, 1] ** 2)
        return dis

    def genTargetwithinRange(self, start_postion, dis_range):
        _, target = self.rand_joint_state(1)
        dis = self.distance(start_postion, target)[0]
        if (dis > dis_range[0]) & (dis < dis_range[1]):
            return target
        else:
            return self.genTargetwithinRange(start_postion, dis_range)

    def genTargets(self, trial_num, seq_len, dis_range):
        targets = np.zeros((trial_num, seq_len, 2))
        joint_state, start_postion = self.rand_joint_state(trial_num)
        targets[:, 0] = start_postion
        for i in range(1, seq_len):
            target_i = np.zeros((trial_num, 2))
            for j in range(trial_num):
                target_i[j] = self.genTargetwithinRange(targets[j, i - 1], dis_range)
            targets[:, i] = target_i
        return targets, joint_state

    def genReach(self, options=None, **kwargs):
        options = {} if options is None else options
        trial_num = options.get("trial_num", 1)
        seq_len = options.get("seq_len", 3)
        dis_range = options.get("dis_range", [0.10, 0.15])
        targets = options.get("targets", self.genTargets(trial_num, seq_len, dis_range)[0])
        joint_state = options.get("joint_state", None)
        hold_durtion = options.get("hold_durtion", 0.08)
        ave_vel = options.get("average_velocity", 0.5)

        trial_num, seq_len, _ = targets.shape
        marker = np.zeros((trial_num, 2 * (seq_len - 1)), dtype=int)
        dis = self.distance(targets[:, 0], targets[:, 1])
        move_dur = np.round(dis / ave_vel, 2)
        move_ntps = np.array([int(n) for n in (move_dur / self.dt)])
        hold_ntps = np.array(int(hold_durtion / self.dt))
        marker[:, 0] = hold_ntps
        traj_list, vel_list = comMiniJerk(targets[:, 0], targets[:, 1], move_dur, move_ntps)
        for i in range(trial_num):
            traj_list[i] = np.vstack(
                (traj_list[i][0] * np.ones((hold_ntps, 2)), traj_list[i])
            )
            vel_list[i] = np.vstack((np.zeros((hold_ntps, 2)), vel_list[i]))
            marker[i, 1] = marker[i, 0] + move_ntps[i]

        for i in range(1, seq_len - 1):
            dis = self.distance(targets[:, i], targets[:, i + 1])
            move_dur = np.round(dis / ave_vel, 2)
            move_ntps = np.array([int(n) for n in (move_dur / self.dt)])
            traj_i, vel_i = comMiniJerk(targets[:, i], targets[:, i + 1], move_dur, move_ntps)
            for j in range(trial_num):
                traj_i[j] = np.vstack(
                    (traj_i[j][0] * np.ones((hold_ntps, 2)), traj_i[j])
                )
                traj_list[j] = np.vstack((traj_list[j], traj_i[j]))
                vel_i[j] = np.vstack((np.zeros((hold_ntps, 2)), vel_i[j]))
                vel_list[j] = np.vstack((vel_list[j], vel_i[j]))
                marker[j, 2 * i] = marker[j, 2 * i - 1] + hold_ntps
                marker[j, 2 * i + 1] = marker[j, 2 * i] + move_ntps[j]

        max_ntps = int(np.max(marker[:, -1]))
        trajactory = np.zeros((trial_num, max_ntps, 2))
        velocity = np.zeros_like(trajactory)
        gocue = np.zeros((trial_num, max_ntps, 2))

        for i in range(trial_num):
            trajactory[i, : marker[i, -1]] = traj_list[i]
            velocity[i, : marker[i, -1]] = vel_list[i]
            gocue[i, : marker[i, 0], 1] = 1
            gocue[i, marker[i, 0] : marker[i, -1], 1] = 0
            gocue[i, marker[i, 0] : marker[i, -1], 0] = ave_vel

        if joint_state is None:
            joint_state = hand_to_joints(targets[:, 0], self.aparams)
        goal = targets[:, 1:]

        return trajactory, velocity, marker, gocue, joint_state, goal


class DoubleReachTask:
    def __init__(self, **kwargs):
        self.joint_range = [[0.0, 140.0], [0.0, 160.0]]
        self.aparams = {"l1": 0.309, "l2": 0.333}
        self.dt = 0.01
        self.workspace = [-0.3, 0.3, 0.2, 0.5]

    def distance(self, start_postion, goal):
        dis = goal - start_postion
        dis = np.sqrt(dis[0] ** 2 + dis[1] ** 2)
        return dis

    def genTargets(self, center_joint, angle_interval, target_radius):
        center_joint = np.deg2rad(np.array(center_joint)[None, :]).astype(np.float32)
        center_cartesian = joints_to_hand(center_joint, self.aparams)[0][0]
        angle_set = np.deg2rad(np.arange(0, 360, angle_interval))
        end_cp = target_radius * np.stack([np.cos(angle_set), np.sin(angle_set)], axis=-1)
        target_set = center_cartesian + end_cp

        x_min, x_max, y_min, y_max = self.workspace
        if not (
            (target_set[:, 0] >= x_min).all()
            & (target_set[:, 0] <= x_max).all()
            & (target_set[:, 1] >= y_min).all()
            & (target_set[:, 1] <= y_max).all()
        ):
            raise ValueError("Target exceeds workspace range!")

        g_joint = hand_to_joints(target_set, self.aparams)
        H, _ = joints_to_hand(g_joint, self.aparams)
        if not (np.round(H, 3) == np.round(target_set, 3)).all():
            raise ValueError("Target exceeds arm range!")

        target_num = target_set.shape[0]
        targets = []
        for i in range(target_num):
            for j in range(target_num):
                if i != j:
                    targets.append(np.stack([target_set[i], target_set[j]], axis=0))
        targets = np.stack(targets, axis=0)
        return targets, center_joint, center_cartesian

    def genReach(self, options=None, **kwargs):
        options = {} if options is None else options
        batch_size = options.get("batch_size", 1)
        center_joint = options.get("center_joint", [38.0, 113.3])
        angle_interval = options.get("angle_interval", 60)
        target_radius = options.get("target_radius", 0.12)
        target_on = options.get("target_on", 0.1)
        delay_durtion = options.get("delay_durtion", 0.6)
        reach_durtion = options.get("reach_durtion", 0.3)
        hold_durtion = options.get("hold_durtion", 0.05)
        catch_trial_proportion = options.get("catch_trial_proportion", 0.2)

        targets, center_joint, center_cartesian = self.genTargets(
            center_joint, angle_interval, target_radius
        )
        block_size = targets.shape[0]
        reps = int(np.ceil(batch_size / block_size))
        targets = np.tile(targets, reps=(reps, 1, 1))
        trial_num = targets.shape[0]
        start_postion = np.tile(center_cartesian, reps=(trial_num, 1))

        move_ntps = int(np.round(reach_durtion / self.dt, 2))
        to = int(round(target_on / self.dt, 2))
        go1 = int(round(delay_durtion / self.dt, 2))
        tt1 = go1 + move_ntps
        go2 = tt1 + 5
        tt2 = go2 + move_ntps
        end = tt2 + int(round(hold_durtion / self.dt, 2))
        marker = np.zeros(6, dtype=int)
        marker[0] = to
        marker[1] = go1
        marker[2] = tt1
        marker[3] = go2
        marker[4] = tt2
        marker[5] = end
        ave_vel_1 = target_radius / reach_durtion
        trajactory = np.zeros((trial_num, end, 2))
        velocity = np.zeros_like(trajactory)
        gocue = np.concatenate(
            (np.zeros((trial_num, end, 1)), np.ones((trial_num, end, 1))), axis=2
        )
        gocue[:, go1:tt1, 0] = ave_vel_1
        gocue[:, go1:tt2, 1] = 0

        for i in range(trial_num):
            dis_2 = self.distance(targets[i, 0], targets[i, 1])
            ave_vel_2 = dis_2 / reach_durtion
            _, traj_1, vel_1, _ = minjerk(
                center_cartesian, targets[i, 0], reach_durtion, move_ntps
            )
            _, traj_2, vel_2, _ = minjerk(
                targets[i, 0], targets[i, 1], reach_durtion, move_ntps
            )
            trajactory[i, :go1] = center_cartesian
            trajactory[i, go1:tt1, :] = traj_1
            trajactory[i, tt1:go2, :] = targets[i, 0]
            trajactory[i, go2:tt2, :] = traj_2
            trajactory[i, tt2:end, :] = targets[i, 1]
            velocity[i, go1:tt1, :] = vel_1
            velocity[i, go2:tt2, :] = vel_2
            gocue[i, go2:tt2, 0] = ave_vel_2

        catch_trial_num = int(round(catch_trial_proportion * trial_num, 2))
        catch_trial = np.random.choice(range(trial_num), catch_trial_num, replace=False)
        trajactory[catch_trial] = start_postion[catch_trial, None, :]
        velocity[catch_trial] = 0
        gocue[catch_trial, :, 0] = 0
        gocue[catch_trial, :, 1] = 1

        joint_state = np.tile(center_joint, (trial_num, 1))
        goal = np.hstack((targets[:, 0], targets[:, 1]))
        goal_for_train = np.hstack((targets[:, 0], targets[:, 1]))
        goal_for_train[catch_trial, :2] = start_postion[catch_trial]
        goal_for_train[catch_trial, 2:] = start_postion[catch_trial]

        return trajactory, velocity, marker, gocue, joint_state, goal, goal_for_train

    def genSingleReach(self, options=None, **kwargs):
        options = {} if options is None else options
        batch_size = options.get("batch_size", 1)
        center_joint = options.get("center_joint", [38.0, 113.3])
        angle_interval = options.get("angle_interval", 60)
        target_radius = options.get("target_radius", 0.12)
        target_on = options.get("target_on", 0.2)
        delay_durtion = options.get("delay_durtion", 0.8)
        reach_durtion = options.get("reach_durtion", 0.4)
        hold_durtion = options.get("hold_durtion", 0.1)
        goal_class = options.get("goal_class", 0)
        catch_trial_proportion = options.get("catch_trial_proportion", 0)

        center_joint = np.deg2rad(np.array(center_joint)[None, :]).astype(np.float32)
        center_cartesian = joints_to_hand(center_joint, self.aparams)[0][0]
        angle_set = np.deg2rad(np.arange(0, 360, angle_interval))
        end_cp = target_radius * np.stack([np.cos(angle_set), np.sin(angle_set)], axis=-1)
        target_set = center_cartesian + end_cp
        block_size = target_set.shape[0]

        x_min, x_max, y_min, y_max = self.workspace
        if not (
            (target_set[:, 0] >= x_min).all()
            & (target_set[:, 0] <= x_max).all()
            & (target_set[:, 1] >= y_min).all()
            & (target_set[:, 1] <= y_max).all()
        ):
            raise ValueError("Target exceeds workspace range!")

        g_joint = hand_to_joints(target_set, self.aparams)
        H, _ = joints_to_hand(g_joint, self.aparams)
        if not (np.round(H, 3) == np.round(target_set, 3)).all():
            raise ValueError("Target exceeds arm range!")

        reps = int(np.ceil(batch_size / block_size))
        targets = np.tile(target_set, reps=(reps, 1))
        trial_num = targets.shape[0]
        start_postion = np.tile(center_cartesian, (trial_num, 1))
        move_dur = np.tile(reach_durtion, (trial_num, 1))
        move_ntps = np.array([int(n) for n in (move_dur / self.dt)])
        ave_vel = target_radius / reach_durtion
        traj_list, vel_list = comMiniJerk(start_postion, targets, move_dur, move_ntps)
        to = int(round(target_on / self.dt, 2))
        go = int(round(delay_durtion / self.dt, 2))
        tt = go + move_ntps[0]
        end = tt + int(round(hold_durtion / self.dt, 2))
        marker = np.zeros(4, dtype=int)
        marker[0] = to
        marker[1] = go
        marker[2] = tt
        marker[3] = end

        trajactory = np.zeros((trial_num, end, 2))
        trajactory[:, :go] = start_postion[:, None, :]
        trajactory[:, go:tt] = np.stack(traj_list)
        trajactory[:, tt:end] = targets[:, None, :]
        velocity = np.zeros_like(trajactory)
        velocity[:, go:tt] = np.stack(vel_list)
        gocue = np.concatenate(
            (np.zeros((trial_num, end, 1)), np.ones((trial_num, end, 1))), axis=2
        )
        gocue[:, go:tt, 0] = ave_vel
        gocue[:, go:tt, 1] = 0

        catch_trial_num = int(round(catch_trial_proportion * trial_num, 2))
        catch_trial = np.random.choice(range(trial_num), catch_trial_num, replace=False)
        trajactory[catch_trial] = start_postion[catch_trial, None, :]
        velocity[catch_trial] = 0
        gocue[catch_trial, :, 0] = 0
        gocue[catch_trial, :, 1] = 1

        joint_state = np.tile(center_joint, (trial_num, 1))
        if goal_class == 0:
            goal = np.hstack((targets, targets))
        else:
            goal = np.hstack((targets, np.zeros_like(targets)))
        goal_for_train = targets
        goal_for_train[catch_trial] = start_postion[catch_trial]
        return trajactory, velocity, marker, gocue, joint_state, goal, goal_for_train


class RTTEnv(mn.environment.Environment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__name__ = "RTTTask"

    def reset(self, *, seed=None, options=None):
        self._set_generator(seed)
        options = {} if options is None else options
        batch_size = options.get("batch_size", 1)
        joint_state = options.get("joint_state", th.ones((batch_size, 2)))
        move_ntps = options.get("movement_timepoints", th.ones(batch_size))
        marker = options.get("marker", th.ones(batch_size, 4))
        goal = options.get("goal", th.ones((batch_size, 2, 2)))
        gocue = options.get("gocue", th.zeros((batch_size, 1, 2)))
        deterministic = options.get("deterministic", False)
        obs_class = options.get("obs_class", "prop")

        self.effector.reset(
            options={"batch_size": batch_size, "joint_state": th.as_tensor(joint_state)}
        )
        self.elapsed = 0.0
        self.target_touched = np.zeros(batch_size, dtype=int)
        self.targets = th.as_tensor(goal).to(self.device)
        self.marker = th.as_tensor(marker)
        self.move_dur = move_ntps * self.dt
        self.goal = self.targets[:, 0]
        self.gocue = th.as_tensor(gocue).to(self.device)
        self.obs_class = obs_class
        action = th.zeros((batch_size, self.muscle.n_muscles)).to(self.device)

        self.obs_buffer["proprioception"] = [self.get_proprioception()] * len(
            self.obs_buffer["proprioception"]
        )
        self.obs_buffer["vision"] = [self.get_vision()] * len(self.obs_buffer["vision"])
        self.obs_buffer["action"] = [action] * self.action_frame_stacking

        obs = self.get_obs(deterministic=deterministic, obs_class=self.obs_class)
        info = {"states": self.states, "action": action, "noisy action": action}
        return obs, info

    def step(self, action, deterministic=False, **kwargs):
        self.elapsed += self.dt
        self.elapsed = round(self.elapsed, 3)

        if deterministic is False:
            noisy_action = self.apply_noise(action, noise=self.action_noise)
        else:
            noisy_action = action

        self.effector.step(noisy_action, **kwargs)
        obs = self.get_obs(action=noisy_action, obs_class=self.obs_class)
        reward = None
        truncated = self.elapsed > self.move_dur
        terminated = bool(self.elapsed >= self.max_ep_duration)

        for key in self.states:
            self.states[key] = self.states[key][truncated == False]
        self.move_dur = self.move_dur[truncated == False]
        self.target_touched = self.target_touched[truncated == False]
        self.targets = self.targets[truncated == False]
        self.marker = self.marker[truncated == False]
        self.goal = self.goal[truncated == False]
        self.gocue = self.gocue[truncated == False]
        obs = obs[truncated == False]

        t = int(self.elapsed / self.dt)
        for i in range(self.goal.shape[0]):
            idx = 2 * self.target_touched[i] + 1
            if self.marker[i, idx] < t:
                self.target_touched[i] += 1
            self.goal[i] = self.targets[i, self.target_touched[i]]

        info = {"states": self.states, "action": action, "noisy action": noisy_action}
        return obs, reward, terminated, truncated, info

    def get_proprioception(self):
        mlen = self.states["muscle"][:, 1:2, :] / self.muscle.l0_ce
        mvel = self.states["muscle"][:, 2:3, :] / self.muscle.vmax
        prop = th.concatenate([mlen, mvel], dim=-1).squeeze(dim=1)
        return self.apply_noise(prop, self.proprioception_noise)

    def get_vision(self):
        vis = self.states["fingertip"]
        return self.apply_noise(vis, self.vision_noise)

    def get_obs(self, action=None, deterministic=False, obs_class="prop"):
        self.update_obs_buffer(action=action)
        if obs_class == "prop":
            obs_as_list = [self.obs_buffer["proprioception"][0]]
        elif obs_class == "vis":
            obs_as_list = [self.obs_buffer["vision"][0]]
        else:
            obs_as_list = [
                self.obs_buffer["proprioception"][0],
                self.obs_buffer["vision"][0],
            ]

        obs_as_list.append(self.goal)
        t = int(round(self.elapsed / self.dt, 2)) - 1
        obs_as_list.append(self.gocue[:, t])
        obs = th.cat(obs_as_list, dim=-1)

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs.shape[-1],), dtype=np.float32
        )
        self.obs_noise = [self._obs_noise] * self.observation_space.shape[0]
        if deterministic is False:
            obs = self.apply_noise(obs, noise=self.obs_noise)
        obs = obs.to(th.float32)
        return obs


class DoubleReachEnv(mn.environment.Environment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__name__ = "DoubleReachTask"

    def reset(self, *, seed=None, options=None):
        self._set_generator(seed)
        options = {} if options is None else options
        batch_size = options.get("batch_size", 30)
        task = options.get("task", "DR")
        joint_state = options.get("joint_state", th.ones((batch_size, 2)))
        move_ntps = options.get("movement_timepoints", th.ones(batch_size))
        marker = options.get("marker", th.ones(6))
        goal = options.get("goal", th.ones((batch_size, 4)))
        gocue = options.get("gocue", th.zeros((batch_size, 10, 2)))
        deterministic = options.get("deterministic", False)
        obs_class = options.get("obs_class", "prop")

        self.effector.reset(
            options={"batch_size": batch_size, "joint_state": th.as_tensor(joint_state)}
        )
        self.elapsed = 0.0
        self.task = task
        self.marker = th.as_tensor(marker)
        self.move_dur = move_ntps * self.dt
        self.goal = th.as_tensor(goal).to(self.device)
        self.start_position = self.states["fingertip"]
        self.gocue = th.as_tensor(gocue).to(self.device)
        self.obs_class = obs_class
        action = th.zeros((batch_size, self.muscle.n_muscles)).to(self.device)

        self.obs_buffer["proprioception"] = [self.get_proprioception()] * len(
            self.obs_buffer["proprioception"]
        )
        self.obs_buffer["vision"] = [self.get_vision()] * len(self.obs_buffer["vision"])
        self.obs_buffer["action"] = [action] * self.action_frame_stacking

        obs = self.get_obs(deterministic=deterministic, obs_class=self.obs_class)
        info = {"states": self.states, "action": action, "noisy action": action}
        return obs, info

    def step(self, action, deterministic=False, **kwargs):
        self.elapsed += self.dt
        self.elapsed = round(self.elapsed, 3)
        if deterministic is False:
            noisy_action = self.apply_noise(action, noise=self.action_noise)
        else:
            noisy_action = action
        self.effector.step(noisy_action, **kwargs)
        truncated = False
        obs = self.get_obs(action=noisy_action, obs_class=self.obs_class)
        reward = None
        terminated = bool(self.elapsed >= self.max_ep_duration)
        info = {"states": self.states, "action": action, "noisy action": noisy_action}
        return obs, reward, terminated, truncated, info

    def get_proprioception(self):
        mlen = self.states["muscle"][:, 1:2, :] / self.muscle.l0_ce
        mvel = self.states["muscle"][:, 2:3, :] / self.muscle.vmax
        prop = th.concatenate([mlen, mvel], dim=-1).squeeze(dim=1)
        return self.apply_noise(prop, self.proprioception_noise)

    def get_vision(self):
        vis = self.states["fingertip"]
        return self.apply_noise(vis, self.vision_noise)

    def get_obs(self, action=None, deterministic=False, obs_class="prop"):
        self.update_obs_buffer(action=action)
        if obs_class == "prop":
            obs_as_list = [self.obs_buffer["proprioception"][0]]
        elif obs_class == "vis":
            obs_as_list = [self.obs_buffer["vision"][0]]
        else:
            obs_as_list = [
                self.obs_buffer["proprioception"][0],
                self.obs_buffer["vision"][0],
            ]

        t = int(round(self.elapsed / self.dt, 2)) - 1
        if self.task == "SR":
            if t < self.marker[0]:
                goal = th.zeros_like(self.goal)
                goal[:, :2] = self.start_position
                goal[:, 2:] = self.start_position
            else:
                goal = self.goal
        elif self.task == "DR":
            if t < self.marker[0]:
                goal = th.zeros_like(self.goal)
                goal[:, :2] = self.start_position
                goal[:, 2:] = self.start_position
            elif t < self.marker[2]:
                goal = self.goal
            else:
                goal = th.zeros_like(self.goal)
                goal[:, 2:] = self.goal[:, 2:]
        else:
            raise ValueError("Task type error!")
        obs_as_list.append(goal)
        obs_as_list.append(self.gocue[:, t])
        obs = th.cat(obs_as_list, dim=-1)

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs.shape[-1],), dtype=np.float32
        )
        self.obs_noise = [self._obs_noise] * self.observation_space.shape[0]
        if deterministic is False:
            obs = self.apply_noise(obs, noise=self.obs_noise)
        obs = obs.to(th.float32)
        return obs


class VREnv(mn.environment.Environment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__name__ = "VisuomotorRotationTask"

    def reset(self, *, seed=None, options=None):
        self._set_generator(seed)
        options = {} if options is None else options
        batch_size = options.get("batch_size", 8)
        rotation_angle = options.get("rotation_angle", 30.0)
        joint_state = options.get("joint_state", th.ones((batch_size, 2)))
        move_ntps = options.get("movement_timepoints", th.ones(batch_size))
        marker = options.get("marker", th.ones(6))
        goal = options.get("goal", th.ones((batch_size, 4)))
        gocue = options.get("gocue", th.zeros((batch_size, 10, 2)))
        deterministic = options.get("deterministic", False)
        obs_class = options.get("obs_class", "prop")

        self.effector.reset(
            options={"batch_size": batch_size, "joint_state": th.as_tensor(joint_state)}
        )
        self.elapsed = 0.0
        self.rotation_angle = rotation_angle
        self.marker = th.as_tensor(marker)
        self.move_dur = move_ntps * self.dt
        self.goal = th.as_tensor(goal).to(self.device)
        self.gocue = th.as_tensor(gocue).to(self.device)
        self.obs_class = obs_class
        action = th.zeros((batch_size, self.muscle.n_muscles)).to(self.device)

        self.obs_buffer["proprioception"] = [self.get_proprioception()] * len(
            self.obs_buffer["proprioception"]
        )
        self.obs_buffer["vision"] = [self.get_vision()] * len(self.obs_buffer["vision"])
        self.obs_buffer["action"] = [action] * self.action_frame_stacking
        obs = self.get_obs(deterministic=deterministic, obs_class=self.obs_class)
        info = {"states": self.states, "action": action, "noisy action": action}
        return obs, info

    def step(self, action, deterministic=False, **kwargs):
        self.elapsed += self.dt
        self.elapsed = round(self.elapsed, 3)
        if deterministic is False:
            noisy_action = self.apply_noise(action, noise=self.action_noise)
        else:
            noisy_action = action
        self.effector.step(noisy_action, **kwargs)
        obs = self.get_obs(action=noisy_action, obs_class=self.obs_class)
        reward = None
        truncated = self.elapsed > self.move_dur
        terminated = bool(self.elapsed >= self.max_ep_duration)

        for key in self.states:
            self.states[key] = self.states[key][truncated == False]
        self.move_dur = self.move_dur[truncated == False]
        self.goal = self.goal[truncated == False]
        self.gocue = self.gocue[truncated == False]
        obs = obs[truncated == False]

        info = {"states": self.states, "action": action, "noisy action": noisy_action}
        return obs, reward, terminated, truncated, info

    def get_proprioception(self):
        mlen = self.states["muscle"][:, 1:2, :] / self.muscle.l0_ce
        mvel = self.states["muscle"][:, 2:3, :] / self.muscle.vmax
        prop = th.concatenate([mlen, mvel], dim=-1).squeeze(dim=1)
        return self.apply_noise(prop, self.proprioception_noise)

    def get_vision(self):
        vis = self.states["fingertip"]
        return self.apply_noise(vis, self.vision_noise)

    def get_obs(self, action=None, deterministic=False, obs_class="prop"):
        self.update_obs_buffer(action=action)
        if obs_class == "prop":
            obs_as_list = [self.obs_buffer["proprioception"][0]]
        elif obs_class == "vis":
            obs_as_list = [self.obs_buffer["vision"][0]]
        else:
            obs_as_list = [
                self.obs_buffer["proprioception"][0],
                self.obs_buffer["vision"][0],
            ]

        t = int(round(self.elapsed / self.dt, 2)) - 1
        if t < self.marker[0]:
            goal = th.zeros_like(self.goal)
            goal[:, :2] = self.states["fingertip"]
            goal[:, 2:] = self.states["fingertip"]
        elif t < self.marker[1]:
            goal = self.goal
        else:
            goal = self.goal
            theta = np.deg2rad(self.rotation_angle)
            rot = th.tensor(
                [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
                dtype=th.float32,
            ).to(self.device)
            goal[:, :2] = goal[:, :2] @ rot.T
            goal[:, 2:] = goal[:, 2:] @ rot.T
        obs_as_list.append(goal)
        obs_as_list.append(self.gocue[:, t])
        obs = th.cat(obs_as_list, dim=-1)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs.shape[-1],), dtype=np.float32
        )
        self.obs_noise = [self._obs_noise] * self.observation_space.shape[0]
        if deterministic is False:
            obs = self.apply_noise(obs, noise=self.obs_noise)
        obs = obs.to(th.float32)
        return obs
