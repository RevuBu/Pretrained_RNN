import numpy as np
import motornet as mn
import gymnasium as gym
import torch as th
from typing import Any, Union
from .kinematics import minjerk, comMiniJerk, joints_to_hand, hand_to_joints


class BasicTask:
    def __init__(self, **kwargs):
        self.joint_range = [[0.0, 140.0], [0.0, 160.0]]
        self.aparams = {"l1": 0.309, "l2": 0.333}
        self.dt = 0.01
        self.workspace = [-0.3, 0.3, 0.2, 0.5]

    def genJointset(self, joint_interval=10):
        j1 = np.deg2rad(
            np.arange(self.joint_range[0][0] + 10, self.joint_range[0][1] - 19, joint_interval)
        )
        j2 = np.deg2rad(
            np.arange(self.joint_range[1][0] + 20, self.joint_range[1][1] - 19, joint_interval)
        )
        J1, J2 = np.meshgrid(j1, j2)
        joint_set = np.column_stack((J1.ravel(), J2.ravel()))
        H, E = joints_to_hand(joint_set, self.aparams)
        x_min, x_max, y_min, y_max = self.workspace
        mask = (
            (H[:, 0] >= x_min) & (H[:, 0] <= x_max) & (H[:, 1] >= y_min) & (H[:, 1] <= y_max)
        )
        self.H = H[mask]
        self.E = E[mask]
        self.joint_set = joint_set[mask]

    def genCartesianGridSet(self, grid_interval=0.05):
        x_min, x_max, y_min, y_max = self.workspace
        x = np.arange(x_min + 0.01, x_max - 0.01, grid_interval)
        y = np.arange(y_min + 0.05, y_max + 0.01, grid_interval)
        X, Y = np.meshgrid(x, y)
        grid_set = np.column_stack((X.ravel(), Y.ravel()))
        joint_set = hand_to_joints(grid_set, self.aparams)
        H, E = joints_to_hand(joint_set, self.aparams)
        mask = (np.round(H[:, 0], 3) == np.round(grid_set[:, 0], 3)) & (
            np.round(H[:, 1], 3) == np.round(grid_set[:, 1], 3)
        )
        self.joint_set = joint_set[mask]
        self.H = H[mask]
        self.E = E[mask]

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
            return self.rand_joint_state(2 * batch_size)

    def distance_mask(self, start_postion, goal, dis_range):
        dis = start_postion - goal
        dis = np.sqrt(dis[:, 0] ** 2 + dis[:, 1] ** 2)
        dis_mask = (dis > dis_range[0]) & (dis < dis_range[1])
        return dis[dis_mask], dis_mask

    def genReachTask(
        self,
        task="grid",
        trial_num=100,
        dis_range=[0.12, 0.48],
        vel_range=[0.3, 0.4, 0.5],
        delay_range=[0.2, 0.4],
        to_range=[0.1, 0.3],
        hold_dur=0.2,
        catch_trial_proportion=0.1,
    ):
        if task == "grid":
            size = self.joint_set.shape[0]
            joint_state = np.delete(self.joint_set, 0, axis=0)
            start_postion = np.delete(self.H, 0, axis=0)
            goal = np.tile(self.H[0], (size - 1, 1))
            for i in range(1, size):
                goal = np.vstack((goal, np.tile(self.H[i], (size - 1, 1))))
                start_postion = np.vstack(
                    (start_postion, np.delete(self.H, i, axis=0))
                )
                joint_state = np.vstack(
                    (joint_state, np.delete(self.joint_set, i, axis=0))
                )
        elif task == "random":
            joint_state, start_postion = self.rand_joint_state(trial_num)
            _, goal = self.rand_joint_state(trial_num)

        dis, dis_mask = self.distance_mask(start_postion, goal, dis_range)
        start_postion = start_postion[dis_mask]
        goal = goal[dis_mask]
        joint_state = joint_state[dis_mask]
        block_size = goal.shape[0]
        trial_num = block_size * len(vel_range)
        joint_state = np.tile(joint_state, reps=(len(vel_range), 1))
        goal = np.tile(goal, reps=(len(vel_range), 1))

        max_ntps = int(
            (dis_range[1] / vel_range[0] + delay_range[1] + hold_dur + 0.3) / self.dt
        )
        marker = np.zeros((trial_num, 4), dtype=int)
        hold_ntps = int(round(hold_dur / self.dt, 2))
        to_dur = np.random.uniform(to_range[0], to_range[1], size=trial_num)
        to_ntps = np.array([int(round(n)) for n in (to_dur / self.dt)])
        delay_dur = np.random.uniform(delay_range[0], delay_range[1], size=trial_num)
        delay_ntps = np.array([int(round(n)) for n in (delay_dur / self.dt)])

        trajactory = np.zeros((trial_num, max_ntps, 2))
        velocity = np.zeros((trial_num, max_ntps, 2))
        goal_for_train = np.zeros((trial_num, max_ntps, 2))
        gocue = np.concatenate(
            (np.zeros((trial_num, max_ntps, 1)), np.ones((trial_num, max_ntps, 1))), axis=2
        )
        catch_trial_num = int(round(catch_trial_proportion * trial_num, 2))
        catch_trial = np.random.choice(range(trial_num), catch_trial_num, replace=False)

        for idx, ave_vel in enumerate(vel_range):
            move_dur = np.round(dis / ave_vel, 2)
            move_ntps = np.array([int(round(n)) for n in (move_dur / self.dt)])
            traj_list, vel_list = comMiniJerk(start_postion, goal, move_dur, move_ntps)

            for trial in range(block_size * idx, block_size * (idx + 1)):
                t = trial % block_size
                to = to_ntps[trial]
                go = to + delay_ntps[trial]
                tt = go + move_ntps[t]
                end = tt + hold_ntps

                trajactory[trial, :go, :] = start_postion[t, None, :]
                trajactory[trial, go:tt, :] = traj_list[t]
                trajactory[trial, tt:end, :] = goal[t, None, :]
                velocity[trial, go:tt, :] = vel_list[t]
                goal_for_train[trial, :to, :] = start_postion[t, None, :]
                goal_for_train[trial, to:end, :] = goal[t, None, :]
                gocue[trial, go:end, 0] = ave_vel
                gocue[trial, go:end, 1] = 0
                marker[trial, 0] = to
                marker[trial, 1] = go
                marker[trial, 2] = tt
                marker[trial, 3] = end

        trajactory[catch_trial] = start_postion[catch_trial % block_size, None, :]
        velocity[catch_trial] = 0
        gocue[catch_trial, :, 0] = 0
        gocue[catch_trial, :, 1] = 1

        return trajactory, velocity, marker, gocue, joint_state, goal, goal_for_train

    def genHoldTask(self, workspace="joint_space", interval=10, hold_dur=[0.3, 1.0]):
        if workspace == "joint_space":
            self.genJointset(joint_interval=interval)
        elif workspace == "cartesian_space":
            self.genCartesianGridSet(joint_interval=interval)
        else:
            print("Workspace error")

        vel = 0.3
        dis_range = [hold_dur[0] * vel, hold_dur[1] * vel]
        traj, vel, _, gocue, j_, g = self.genReachTask(
            dis_range=dis_range,
            vel_range=[vel],
            delay_range=[0.0, 0.0],
            hold_dur=0.0,
            catch_trial_proportion=1,
        )

        n_min = int(round(hold_dur[0] / self.dt, 2))
        trial_num, n_max, _ = traj.shape
        size = self.H.shape[0]
        joint_state_1 = self.joint_set
        goal_1 = self.H
        joint_state_2, goal_2 = self.rand_joint_state(trial_num - size)
        joint_state = np.concatenate((joint_state_1, joint_state_2))
        goal = np.concatenate((goal_1, goal_2))
        trajactory = np.repeat(goal[:, None, :], n_max, axis=1)

        trajactory = np.concatenate((traj, trajactory), axis=0)
        velocity = np.zeros((2 * trial_num, n_max, 2))
        ntps = np.random.randint(n_min, n_max + 1, size=2 * trial_num)
        gocue = np.concatenate((gocue, gocue), axis=0)
        joint_state = np.concatenate((j_, joint_state), axis=0)
        goal = np.concatenate((g, goal), axis=0)

        return trajactory, velocity, ntps, gocue, joint_state, goal

    def centerOut(self, options=None, **kwargs):
        options = {} if options is None else options
        batch_size = options.get("batch_size", 1)
        center_joint = options.get("center_joint", [38.0, 113.3])
        angle_interval = options.get("angle_interval", 22.5)
        delay_durtion = options.get("delay_range", 0.6)
        hold_durtion = options.get("hold_durtion", 0.2)
        reach_distance = options.get("reach_distance", 0.12)
        ave_vel = options.get("average_velocity", 0.3)

        center_joint = np.deg2rad(np.array(center_joint)[None, :]).astype(np.float32)
        angle_set = np.deg2rad(np.arange(0, 360, angle_interval))
        reps = int(np.ceil(batch_size / len(angle_set)))
        angle = np.tile(angle_set, reps=reps)
        trial_num = reps * len(angle_set)
        center_cartesian = joints_to_hand(center_joint, self.aparams)[0][0]
        end_cp = reach_distance * np.stack([np.cos(angle), np.sin(angle)], axis=-1)
        start_postion = np.tile(center_cartesian.reshape(1, -1), (trial_num, 1))
        joint_state = np.tile(center_joint, (trial_num, 1))
        goal = start_postion[:, :2] + end_cp

        x_min, x_max, y_min, y_max = self.workspace
        if not (
            (goal[:, 0] >= x_min).all()
            & (goal[:, 0] <= x_max).all()
            & (goal[:, 1] >= y_min).all()
            & (goal[:, 1] <= y_max).all()
        ):
            raise ValueError("Target exceeds workspace range!")

        g_joint = hand_to_joints(goal, self.aparams)
        H, _ = joints_to_hand(g_joint, self.aparams)
        if not (np.round(H, 3) == np.round(goal, 3)).all():
            raise ValueError("Target exceeds arm range!")

        move_dur = np.ones(trial_num) * np.round(reach_distance / ave_vel, 2)
        move_ntps = np.array([int(n) for n in (move_dur / self.dt)])
        traj_list, vel_list = comMiniJerk(start_postion, goal, move_dur, move_ntps)

        go = int(round(delay_durtion / self.dt, 2))
        tt = go + move_ntps[0]
        end = tt + int(round(hold_durtion / self.dt, 2))
        marker = np.zeros((trial_num, 3), dtype=int)
        marker[:, 0] = go
        marker[:, 1] = tt
        marker[:, 2] = end

        trajactory = np.zeros((trial_num, end, 2))
        trajactory[:, :go, :] = start_postion[:, None, :]
        trajactory[:, go:tt, :] = np.stack(traj_list)
        trajactory[:, tt:, :] = goal[:, None, :]
        velocity = np.zeros_like(trajactory)
        velocity[:, go:tt, :] = np.stack(vel_list)
        gocue = np.concatenate(
            (np.zeros((trial_num, end, 1)), np.ones((trial_num, end, 1))), axis=2
        )
        gocue[:, go:tt, 0] = ave_vel
        gocue[:, go:tt, 1] = 0

        return trajactory, velocity, marker, gocue, joint_state, goal


class CenterOutTask:
    def __init__(self, **kwargs):
        self.joint_range = [[0.0, 140.0], [0.0, 160.0]]
        self.aparams = {"l1": 0.309, "l2": 0.333}
        self.dt = 0.01
        self.workspace = [-0.3, 0.3, 0.2, 0.5]

    def genTargets(self, center_joints, angle_intervals, target_radius):
        target_set = []
        center_set = []
        joint_set = []
        for c_j in center_joints:
            center_joint = np.deg2rad(np.array(c_j)[None, :]).astype(np.float32)
            center_cartesian = joints_to_hand(center_joint, self.aparams)[0][0]
            for angle_interval in angle_intervals:
                angle_set = np.deg2rad(np.arange(0, 360, angle_interval))
                for radius in target_radius:
                    end_cp = radius * np.stack([np.cos(angle_set), np.sin(angle_set)], axis=-1)
                    joint_set.append(np.tile(center_joint, reps=(len(end_cp), 1)))
                    center_set.append(np.tile(center_cartesian, reps=(len(end_cp), 1)))
                    target_set.append(center_cartesian + end_cp)

        targets = np.concatenate(target_set)
        joints = np.concatenate(joint_set)
        centers = np.concatenate(center_set)

        x_min, x_max, y_min, y_max = self.workspace
        if not (
            (targets[:, 0] >= x_min).all()
            & (targets[:, 0] <= x_max).all()
            & (targets[:, 1] >= y_min).all()
            & (targets[:, 1] <= y_max).all()
        ):
            raise ValueError("Target exceeds workspace range!")

        g_joint = hand_to_joints(targets, self.aparams)
        H, _ = joints_to_hand(g_joint, self.aparams)
        if not (np.round(H, 3) == np.round(targets, 3)).all():
            raise ValueError("Target exceeds arm range!")

        return targets, joints, centers

    def genReach(self, options=None, **kwargs):
        options = {} if options is None else options
        batch_size = options.get("batch_size", 1)
        center_joints = options.get("center_joint", [[38.0, 113.3]])
        angle_intervals = options.get("angle_interval", [45])
        target_radius = options.get("target_radius", [0.12])
        target_on = options.get("target_on", 0.2)
        delay_durtion = options.get("delay_durtion", 0.8)
        reach_durtion = options.get("reach_durtion", 0.4)
        hold_durtion = options.get("hold_durtion", 0.1)
        catch_trial_proportion = options.get("catch_trial_proportion", 0)

        targets, joints, centers = self.genTargets(center_joints, angle_intervals, target_radius)
        reps = int(np.ceil(batch_size / len(targets)))
        goal = np.tile(targets, reps=(reps, 1))
        joint_state = np.tile(joints, reps=(reps, 1))
        start_postion = np.tile(centers, reps=(reps, 1))
        trial_num = goal.shape[0]
        move_dur = np.tile(reach_durtion, (trial_num, 1))
        move_ntps = np.array([int(n) for n in (move_dur / self.dt)])
        dis = start_postion - goal
        dis = np.sqrt(dis[:, 0] ** 2 + dis[:, 1] ** 2)
        ave_vel = dis / reach_durtion
        traj_list, vel_list = comMiniJerk(start_postion, goal, move_dur, move_ntps)
        to = int(round(target_on / self.dt, 2))
        go = to + int(round(delay_durtion / self.dt, 2))
        tt = go + move_ntps[0]
        end = tt + int(round(hold_durtion / self.dt, 2))
        marker = np.zeros((goal.shape[0], 4), dtype=int)
        marker[:, 0] = to
        marker[:, 1] = go
        marker[:, 2] = tt
        marker[:, 3] = end

        trajactory = np.zeros((trial_num, end, 2))
        trajactory[:, :go] = start_postion[:, None, :]
        trajactory[:, go:tt] = np.stack(traj_list)
        trajactory[:, tt:end] = goal[:, None, :]
        velocity = np.zeros_like(trajactory)
        velocity[:, go:tt] = np.stack(vel_list)
        gocue = np.concatenate(
            (np.zeros((trial_num, end, 1)), np.ones((trial_num, end, 1))), axis=2
        )
        gocue[:, go:tt, 0] = ave_vel[:, None]
        gocue[:, go:tt, 1] = 0
        goal_for_train = np.zeros_like(trajactory)
        goal_for_train[:, :to, :] = start_postion[:, None, :]
        goal_for_train[:, to:end, :] = goal[:, None, :]

        catch_trial_num = int(round(catch_trial_proportion * trial_num, 2))
        catch_trial = np.random.choice(range(trial_num), catch_trial_num, replace=False)
        trajactory[catch_trial] = start_postion[catch_trial, None, :]
        velocity[catch_trial] = 0
        gocue[catch_trial, :, 0] = 0
        gocue[catch_trial, :, 1] = 1
        goal_for_train[catch_trial] = start_postion[catch_trial, None]

        return trajactory, velocity, marker, gocue, joint_state, goal, goal_for_train


class BasicTaskEnv(mn.environment.Environment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__name__ = "BasicTask"

    def reset(self, *, seed=None, options=None):
        self._set_generator(seed)
        options = {} if options is None else options
        batch_size = options.get("batch_size", 1)
        joint_state = options.get("joint_state", th.ones((batch_size, 2)))
        move_ntps = options.get("movement_timepoints", th.ones(batch_size))
        goal = options.get("goal", th.ones((batch_size, 1, 2)))
        gocue = options.get("gocue", th.zeros((batch_size, 1, 2)))
        deterministic = options.get("deterministic", False)
        obs_class = options.get("obs_class", "prop")

        self.effector.reset(
            options={"batch_size": batch_size, "joint_state": th.as_tensor(joint_state)}
        )
        self.elapsed = 0.0
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
        obs_as_list.append(self.goal[:, t])
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
