from typing import Any

import gymnasium
import pytest
import torch
import torch.utils
import torch.utils.data
from torch import Tensor

from project.datamodules.rl.envs import make_torch_env, make_torch_vectorenv
from project.datamodules.rl.rl_types import (
    Episode,
    EpisodeBatch,
    VectorEnv,
)
from project.datamodules.rl.wrappers.tensor_spaces import TensorBox, TensorDiscrete
from project.utils.tensor_regression import TensorRegressionFixture
from project.utils.types import NestedDict
from project.utils.utils import get_shape_ish

pytest.register_assert_rewrite(__file__)


class EnvTests:
    """Tests for the RL environments whose observations / actions are on the GPU."""

    @pytest.fixture(scope="class")
    def env_id(self, request: pytest.FixtureRequest):
        env_id_str = getattr(request, "param", None)
        if not env_id_str:
            raise RuntimeError(
                "You are supposed to pass the env_id via an indirect parametrization!"
            )
        return env_id_str

    @pytest.fixture(scope="class", params=[123])
    def seed(self, request: pytest.FixtureRequest):
        return request.param

    @pytest.fixture(scope="class")
    def env(self, env_id: str, seed: int, device: torch.device):
        return make_torch_env(env_id, device=device, seed=seed)

    @pytest.fixture(scope="class", params=[1, 11, 16])
    def num_envs(self, request: pytest.FixtureRequest) -> int:
        return request.param

    @pytest.fixture(scope="class")
    def vectorenv(self, env_id: str, seed: int, num_envs: int, device: torch.device):
        return make_torch_vectorenv(env_id=env_id, num_envs=num_envs, seed=seed, device=device)

    @pytest.mark.timeout(30)
    def test_env(
        self,
        env: gymnasium.Env[torch.Tensor, torch.Tensor],
        seed: int,
        device: torch.device,
        tensor_regression: TensorRegressionFixture,
    ):
        observation_from_reset, info_from_reset = env.reset(seed=seed)

        def _check_observation(obs: Any):
            assert isinstance(obs, Tensor) and obs.device == device
            # todo: fix issue with `inf` in the cartpole observations of Gymnax.
            assert obs in env.observation_space, (obs, type(obs), env.observation_space)

        def _check_dict(d: NestedDict[str, Tensor | Any]):
            for k, value in d.items():
                if isinstance(value, dict):
                    _check_dict(value)
                elif value is not None:
                    assert isinstance(value, Tensor) and value.device == device, k

        _check_observation(observation_from_reset)
        _check_dict(info_from_reset)

        observation_from_space = env.observation_space.sample()
        _check_observation(observation_from_space)

        action_from_space = env.action_space.sample()
        assert isinstance(action_from_space, torch.Tensor) and action_from_space.device == device
        assert action_from_space in env.action_space

        observation_from_step, reward, terminated, truncated, info_from_step = env.step(
            action_from_space
        )
        _check_observation(observation_from_step)
        assert (
            isinstance(reward, torch.Tensor)
            and reward.device == device
            and reward.dtype == torch.float32
        )
        assert (
            isinstance(terminated, torch.Tensor)
            and terminated.device == device
            and terminated.dtype == torch.bool
        )
        assert (
            isinstance(truncated, torch.Tensor)
            and truncated.device == device
            and truncated.dtype == torch.bool
        )
        _check_dict(info_from_step)

        tensor_regression.check(
            {
                "obs_from_reset": observation_from_reset,
                "info_from_reset": info_from_reset,
                "obs_from_space": observation_from_space,
                "action_from_space": action_from_space,
                "obs_from_step": observation_from_step,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "info_from_step": info_from_step,
            }
        )

    @pytest.mark.timeout(60)
    def test_vectorenv(
        self,
        vectorenv: VectorEnv[torch.Tensor, torch.Tensor],
        num_envs: int,
        device: torch.device,
        seed: int,
        tensor_regression: TensorRegressionFixture,
    ):
        assert vectorenv.num_envs == num_envs

        obs_batch_from_reset, info_batch_from_reset = vectorenv.reset(seed=seed)

        def _check_obs(obs: Any):
            assert isinstance(obs, Tensor) and obs.device == device and obs.shape[0] == num_envs
            if not obs.isnan().any():
                assert obs in vectorenv.observation_space
                assert all(obs_i in vectorenv.single_observation_space for obs_i in obs)

        def _check_dict(d: NestedDict[str, Tensor | Any]):
            for k, value in d.items():
                if isinstance(value, dict):
                    _check_dict(value)
                elif value is not None:
                    assert (
                        isinstance(value, Tensor)
                        and value.device == device
                        and value.shape[0] == num_envs
                    ), k

        _check_obs(obs_batch_from_reset)
        _check_dict(info_batch_from_reset)

        obs_from_space = vectorenv.observation_space.sample()
        _check_obs(obs_from_space)

        action_from_space = vectorenv.action_space.sample()
        assert isinstance(action_from_space, torch.Tensor) and action_from_space.device == device
        assert action_from_space in vectorenv.action_space

        obs_from_step, reward, terminated, truncated, info_from_step = vectorenv.step(
            action_from_space
        )
        _check_obs(obs_from_step)
        assert (
            isinstance(reward, torch.Tensor)
            and reward.device == device
            and reward.dtype == torch.float32
            and reward.shape == (num_envs,)
        )

        assert (
            isinstance(terminated, torch.Tensor)
            and terminated.device == device
            and terminated.dtype == torch.bool
            and terminated.shape == (num_envs,)
        )
        assert (
            isinstance(truncated, torch.Tensor)
            and truncated.device == device
            and truncated.dtype == torch.bool
            and truncated.shape == (num_envs,)
        )
        _check_dict(info_from_step)

        tensor_regression.check(
            {
                "obs_from_reset": obs_batch_from_reset,
                "info_from_reset": info_batch_from_reset,
                "obs_from_space": obs_from_space,
                "action_from_space": action_from_space,
                "obs_from_step": obs_from_step,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "info_from_step": info_from_step,
            }
        )


def _check_episode_tensor(
    v: Any,
    device: torch.device,
    space: gymnasium.Space[Tensor] | None = None,
    nested: bool = False,
    dtype: torch.dtype | None = None,
):
    assert isinstance(v, Tensor) and v.device == device
    assert not v.is_nested
    if space:
        assert all(v_i in space for v_i in v), (len(v), v[0], space)
    if dtype:
        assert v.dtype == dtype


def check_episode(episode: Episode, env: gymnasium.Env[Tensor, Any], device: torch.device):
    assert episode["observations"] is episode.observations

    if isinstance(env, VectorEnv):
        observation_space = env.single_observation_space
    else:
        observation_space = env.observation_space
    _check_episode_tensor(episode.observations, device=device, space=observation_space)

    assert episode["actions"] is episode.actions
    if isinstance(env, VectorEnv):
        action_space = env.single_action_space
    else:
        action_space = env.action_space
    _check_episode_tensor(episode.actions, device=device, space=action_space)

    assert episode["rewards"] is episode.rewards
    _check_episode_tensor(episode.rewards, device=device, dtype=torch.float32)

    assert episode["terminated"] is episode.terminated
    _check_episode_tensor(episode.terminated, device=device, dtype=torch.bool)

    assert episode["truncated"] is episode.truncated
    _check_episode_tensor(episode.truncated, device=device, dtype=torch.bool)


def check_episode_batch(
    episode: EpisodeBatch, env: VectorEnv[Tensor, Any], batch_size: int, device: torch.device
):
    obs = episode.observations
    assert isinstance(obs, Tensor) and obs.device == device

    def _check_episode_batch_tensor(
        v: Tensor,
        single_space: TensorBox | TensorDiscrete | gymnasium.Space[Tensor] | None = None,
        dtype: torch.dtype | None = None,
    ):
        shape = get_shape_ish(v)
        assert shape[0] == batch_size
        if v.is_nested:
            assert shape[1] == "?"
        elif len(shape) > 1:
            assert isinstance(shape[1], int)
        if single_space:
            assert shape[2:] == single_space.shape
            if dtype is None:
                dtype = single_space.dtype
        assert v.device == device
        if dtype:
            assert v.dtype == dtype

    _check_episode_batch_tensor(obs, single_space=env.single_observation_space)
    _check_episode_batch_tensor(episode.actions, single_space=env.single_action_space)
    _check_episode_batch_tensor(episode.rewards, dtype=torch.float32)
    _check_episode_batch_tensor(episode.terminated, dtype=torch.bool)
    _check_episode_batch_tensor(episode.truncated, dtype=torch.bool)
