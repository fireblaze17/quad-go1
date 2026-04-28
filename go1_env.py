import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


class Go1Env(gym.Env):
    def __init__(
        self,
        reset_joint_noise=0.1,
        reset_velocity_noise=0.1,
    ):
        self.model = mujoco.MjModel.from_xml_path(
            r"c:\Learning code\mujoco_menagerie\unitree_go1\scene.xml"
        )
        self.data = mujoco.MjData(self.model)
        self.home_key_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_KEY,
            "home",
        )

        if self.home_key_id == -1:
            raise ValueError("Could not find the 'home' keyframe in the Go1 model.")

        self.step_count = 0
        self.max_steps = 1000
        self.reset_joint_noise = reset_joint_noise
        self.reset_velocity_noise = reset_velocity_noise
        self.action_scale = 0.25
        self.ctrl_low = self.model.actuator_ctrlrange[:, 0].astype(np.float32)
        self.ctrl_high = self.model.actuator_ctrlrange[:, 1].astype(np.float32)
        self.default_ctrl = self.model.key_ctrl[self.home_key_id].astype(np.float32)
        self.home_joint_angles = self.model.key_qpos[self.home_key_id][7:].astype(
            np.float32
        )

        obs_size = self.model.nq + self.model.nv
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.model.nu,),
            dtype=np.float32,
        )

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)

    def _upright_score(self):
        quat = self.data.qpos[3:7]
        mat = np.zeros(9)

        mujoco.mju_quat2Mat(mat, quat)

        body_z_axis = mat.reshape(3, 3)[:, 2]
        world_z_axis = np.array([0.0, 0.0, 1.0])

        return np.dot(body_z_axis, world_z_axis)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key_id)
        self.step_count = 0

        if self.reset_joint_noise > 0.0:
            joint_noise = self.np_random.uniform(
                low=-self.reset_joint_noise,
                high=self.reset_joint_noise,
                size=self.model.nu,
            )
            self.data.qpos[7:] += joint_noise

        if self.reset_velocity_noise > 0.0:
            velocity_noise = self.np_random.uniform(
                low=-self.reset_velocity_noise,
                high=self.reset_velocity_noise,
                size=self.model.nv,
            )
            self.data.qvel[:] += velocity_noise

        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        info = {"base_height": self.data.qpos[2]}

        return obs, info

    def step(self, action):
        self.step_count += 1

        action = np.clip(action, self.action_space.low, self.action_space.high)
        ctrl = self.default_ctrl + self.action_scale * action
        ctrl = np.clip(ctrl, self.ctrl_low, self.ctrl_high)
        self.data.ctrl[:] = ctrl

        mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        base_height = self.data.qpos[2]
        upright_score = self._upright_score()
        terminated = base_height < 0.1755

        alive_reward = 0.0 if terminated else 1.0
        upright_reward = max(0.0, upright_score)
        control_penalty = 0.01 * np.sum(np.square(action))
        joint_angles = self.data.qpos[7:]
        pose_error = joint_angles - self.home_joint_angles
        pose_penalty = 0.1 * np.sum(np.square(pose_error))
        base_xy_position = self.data.qpos[0:2]
        position_penalty = 0.5 * np.sum(np.square(base_xy_position))
        base_xy_velocity = self.data.qvel[0:2]
        velocity_penalty = 0.1 * np.sum(np.square(base_xy_velocity))
        base_angular_velocity = self.data.qvel[3:6]
        angular_velocity_penalty = 0.05 * np.sum(np.square(base_angular_velocity))
        reward = (
            alive_reward
            + upright_reward
            - control_penalty
            - pose_penalty
            - position_penalty
            - velocity_penalty
            - angular_velocity_penalty
        )

        truncated = self.step_count >= self.max_steps
        info = {
            "base_height": base_height,
            "upright_score": upright_score,
            "alive_reward": alive_reward,
            "upright_reward": upright_reward,
            "control_penalty": control_penalty,
            "pose_penalty": pose_penalty,
            "position_penalty": position_penalty,
            "velocity_penalty": velocity_penalty,
            "angular_velocity_penalty": angular_velocity_penalty,
            "ctrl": ctrl,
        }

        return obs, reward, terminated, truncated, info

    def close(self):
        pass
