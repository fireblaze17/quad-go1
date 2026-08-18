"""Small PPO loading helpers shared by training and diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv


class EnvClippedActionPPO(PPO):
    """PPO that lets Go1Env own the physical action clip.

    Stable-Baselines3 normally clips continuous Box actions to the env action
    bounds before calling env.step() and also clips in model.predict(). Go1Env
    performs the physical action clip internally so actuator diagnostics and
    execution use one code path.
    """

    def predict(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        self.policy.set_training_mode(False)
        obs_tensor, vectorized_env = self.policy.obs_to_tensor(observation)

        with torch.no_grad():
            actions = self.policy._predict(obs_tensor, deterministic=deterministic)
        actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))

        if isinstance(self.action_space, spaces.Box) and self.policy.squash_output:
            actions = self.policy.unscale_action(actions)

        if not vectorized_env:
            actions = actions.squeeze(axis=0)

        return actions, state

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with torch.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            env_actions = actions
            if isinstance(self.action_space, spaces.Box) and self.policy.squash_output:
                env_actions = self.policy.unscale_action(env_actions)

            new_obs, rewards, dones, infos = env.step(env_actions)

            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with torch.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with torch.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())
        callback.on_rollout_end()

        return True


def _same_box_shape(left: spaces.Space, right: spaces.Space) -> bool:
    return (
        isinstance(left, spaces.Box)
        and isinstance(right, spaces.Box)
        and left.shape == right.shape
    )


def load_ppo_same_shape_action_space(
    model_cls: type,
    path: str | Path,
    *,
    env=None,
    device: str = "auto",
    **kwargs: Any,
):
    """Load PPO while allowing same-shape checkpoint action bounds.

    Stable-Baselines3 stores exact action-space bounds in checkpoints. The
    current environment may use different bounds with the same action shape;
    observation or action-dimension changes must still fail.
    """
    try:
        return model_cls.load(path, env=env, device=device, **kwargs)
    except ValueError as exc:
        if env is None or "Action spaces do not match" not in str(exc):
            raise

        probe = model_cls.load(path, env=None, device=device, **kwargs)
        saved_obs_space = getattr(probe, "observation_space", None)
        saved_action_space = getattr(probe, "action_space", None)
        if not _same_box_shape(saved_action_space, env.action_space):
            raise
        if saved_obs_space != env.observation_space:
            raise

        return model_cls.load(
            path,
            env=env,
            device=device,
            custom_objects={"action_space": env.action_space},
            **kwargs,
        )
