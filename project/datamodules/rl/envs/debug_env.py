from typing import Any, SupportsFloat, TypedDict

import gymnasium
import gymnasium.envs.registration
import torch

from project.datamodules.rl.rl_types import VectorEnv
from project.datamodules.rl.wrappers.tensor_spaces import (
    TensorBox,
    TensorDiscrete,
    TensorMultiDiscrete,
)


class DebugEnvInfo(TypedDict):
    episode_length: torch.Tensor
    target: torch.Tensor


class DebugEnv(gymnasium.Env[torch.Tensor, torch.Tensor]):
    """A simple environment for debugging.

    The goal is to match the state with a hidden target state by adding 1, subtracting 1, or
    staying the same.
    """

    def __init__(
        self,
        min: int = -10,
        max: int = 10,
        target: int = 5,
        initial_state: int = 0,
        max_episode_length: int = 20,
        randomize_target: bool = False,
        randomize_initial_state: bool = False,
        wrap_around_state: bool = False,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.int32,
    ):
        # Don't call super().__init__ because it would cause an error below.
        # super().__init__()

        self.min = min
        self.max = max
        self.max_episode_length = max_episode_length

        self.randomize_target = randomize_target
        self.randomize_initial_state = randomize_initial_state

        self.wrap_around_state = wrap_around_state
        self.device = device
        self.dtype = dtype
        self.rng = torch.Generator(device=self.device)
        self.observation_space: TensorBox = TensorBox(
            low=self.min, high=self.max, shape=(), dtype=self.dtype, device=self.device
        )
        # todo: make this a TensorBox(-1, 1) for a version with a continuous action space.
        self.action_space: TensorDiscrete = TensorDiscrete(
            start=-1, n=3, dtype=self.dtype, device=self.device
        )
        self._episode_length = torch.zeros(
            self.observation_space.shape, dtype=torch.int32, device=device
        )

        self._target = torch.as_tensor(target, dtype=dtype, device=device)
        assert self._target in self.observation_space, "invalid target!"

        self._initial_state = torch.as_tensor(initial_state, dtype=dtype, device=device)
        assert self._initial_state in self.observation_space, "invalid initial state!"
        self._state = self._initial_state.clone()

        self.spec = gymnasium.envs.registration.EnvSpec(
            id="DebugEnv-v0",
            entry_point="project.datamodules.rl.envs.debug_env:DebugEnv",
            max_episode_steps=max_episode_length,
            vector_entry_point="project.datamodules.rl.envs.debug_env:DebugVectorEnv",
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[torch.Tensor, DebugEnvInfo]:
        if seed:
            self.rng.manual_seed(seed)
            self.observation_space.seed(seed)
            self.action_space.seed(seed)

        if self.randomize_target:
            # Set a new target for the next episode.
            self._target = self.observation_space.sample()

        if self.randomize_initial_state:
            # Set a new initial state for the next episode.
            self._state = self.observation_space.sample()
        else:
            self._state = self._initial_state.clone()

        self._episode_length = torch.zeros_like(self._episode_length)
        return (
            self._state,
            {"episode_length": self._episode_length, "target": self._target},
        )

    def step(
        self, action: torch.Tensor
    ) -> tuple[torch.Tensor, SupportsFloat, torch.BoolTensor, torch.BoolTensor, DebugEnvInfo]:
        if action not in self.action_space:
            raise RuntimeError(
                f"Invalid action: {action} of type {type(action)} , {action.dtype=}, {action.device=} "
                f"is not in {self.action_space}."
                + (
                    f" (wrong device: {action.device}!={self.action_space.device})"
                    if action.device != self.action_space.device
                    else ""
                )
                + (
                    f" (wrong dtype: {action.dtype} can't be casted to {self.action_space.dtype})"
                    if not torch.can_cast(action.dtype, self.action_space.dtype)
                    else ""
                )
            )
        assert torch.can_cast(action.dtype, self.action_space.dtype), (
            action.dtype,
            self.action_space.dtype,
        )

        self._state += action
        # Two options: Either we wrap around, or we clamp to the range.
        if self.wrap_around_state:
            self._state %= self.max
        else:
            self._state = torch.clamp(
                self._state,
                min=torch.zeros_like(self._state),
                max=self.max * torch.ones_like(self._state),
            )
        assert self._target is not None
        reward = -(self._state - self._target).abs().to(dtype=torch.float32)
        self._episode_length += 1
        episode_ended = self._episode_length == self.max_episode_length
        at_target = self._state == self._target
        terminated: torch.BoolTensor = episode_ended & at_target  # type: ignore
        truncated: torch.BoolTensor = episode_ended & ~at_target  # type: ignore
        return (
            self._state,
            reward,
            terminated,
            truncated,
            {"episode_length": self._episode_length, "target": self._target},
        )


class DebugVectorEnv(DebugEnv, VectorEnv[torch.Tensor, torch.Tensor]):
    """Same as DebugEnv, but vectorized."""

    def __init__(
        self,
        num_envs: int,
        min: int = -10,
        max: int = 10,
        target: int = 5,
        initial_state: int = 0,
        max_episode_length: int = 20,
        randomize_target: bool = False,
        randomize_initial_state: bool = False,
        wrap_around_state: bool = False,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.int32,
    ):
        self.num_envs = num_envs
        super().__init__(
            min=min,
            max=max,
            target=target,
            initial_state=initial_state,
            max_episode_length=max_episode_length,
            randomize_target=randomize_target,
            randomize_initial_state=randomize_initial_state,
            wrap_around_state=wrap_around_state,
            device=device,
            dtype=dtype,
        )
        single_observation_space = self.observation_space
        single_action_space = self.action_space
        # todo: double-check that 1 is in this this space (bounds are included).
        VectorEnv.__init__(
            self,
            num_envs=num_envs,
            observation_space=single_observation_space,
            action_space=single_action_space,
        )
        expected_observation_space = TensorBox(
            low=self.min,
            high=self.max,
            shape=(self.num_envs,),
            dtype=self.dtype,
            device=self.device,
        )
        assert self.observation_space == expected_observation_space, (
            expected_observation_space,
            self.observation_space,
        )
        expected_action_space = TensorMultiDiscrete(
            start=torch.full(
                (self.num_envs,), fill_value=-1, dtype=self.dtype, device=self.device
            ),
            nvec=torch.full((self.num_envs,), fill_value=3, dtype=self.dtype, device=self.device),
            dtype=self.dtype,
            device=self.device,
        )
        assert self.action_space == expected_action_space, (
            expected_action_space,
            self.action_space,
        )
        self._episode_length = self._episode_length.expand((self.num_envs,))
        self._state = self._state.expand(self.observation_space.shape)
        self._target = self._target.expand(self.observation_space.shape)
        self._initial_state = self._initial_state.expand(self.observation_space.shape)
