"""
Performance Gain Reward Worker

This reward worker directly uses pre-computed performance_gain from the dataset as reward.
Designed for AFlow workflow optimization where rewards are already calculated.
"""

import numpy as np
import torch
from typing import Dict

from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.utils.logging import get_logger

logger = get_logger()


class PerformanceGainRewardWorker(Worker):
    """
    Reward worker that uses pre-computed performance_gain from dataset.

    This is designed for datasets where rewards are already calculated offline,
    such as AFlow workflow optimization data with parent/child score comparisons.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Get configuration
        worker_config = kwargs.get('worker_config', {})

        # Reward scaling factor (default 10.0 to match GRPO scale)
        self.reward_scale = getattr(worker_config, 'reward_scale', 10.0)

        # Whether to clip rewards
        self.clip_rewards = getattr(worker_config, 'clip_rewards', True)
        self.reward_min = getattr(worker_config, 'reward_min', -1.0)
        self.reward_max = getattr(worker_config, 'reward_max', 1.0)

        logger.info(f"PerformanceGainRewardWorker initialized: "
                   f"scale={self.reward_scale}, clip={self.clip_rewards}")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        pass

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE, clear_cache=False)
    def compute_rewards(self, data: DataProto) -> DataProto:
        """
        Compute rewards from pre-calculated performance_gain in dataset.

        Input DataProto must contain:
            - non_tensor_batch["performance_gain"]: Pre-computed rewards

        Returns:
            DataProto with response_level_rewards tensor
        """
        # Extract performance_gain from data
        if "performance_gain" not in data.non_tensor_batch:
            logger.error("performance_gain field not found in data.non_tensor_batch")
            logger.error(f"Available fields: {list(data.non_tensor_batch.keys())}")
            raise KeyError("performance_gain field is required but not found in data")

        performance_gains = data.non_tensor_batch["performance_gain"]

        # Convert to tensor and scale
        if isinstance(performance_gains, (list, tuple)):
            rewards = torch.tensor(performance_gains, dtype=torch.float32)
        elif isinstance(performance_gains, torch.Tensor):
            rewards = performance_gains.float()
        elif isinstance(performance_gains, np.ndarray):
            # Convert to float32 first to handle object dtype arrays
            rewards = torch.from_numpy(performance_gains.astype(np.float32))
        else:
            raise TypeError(f"Unexpected type for performance_gain: {type(performance_gains)}")

        # Scale rewards
        rewards = rewards * self.reward_scale

        # Clip if requested
        if self.clip_rewards:
            rewards = torch.clamp(rewards, min=self.reward_min, max=self.reward_max)

        # Log statistics
        logger.info(f"Computed rewards for {len(rewards)} samples: "
                   f"mean={rewards.mean().item():.4f}, "
                   f"std={rewards.std().item():.4f}, "
                   f"min={rewards.min().item():.4f}, "
                   f"max={rewards.max().item():.4f}")

        # Create token-level rewards (all zeros, only response-level matters)
        batch_size = len(rewards)
        if "responses" in data.batch:
            response_shape = data.batch["responses"].shape
            token_level_rewards = torch.zeros(response_shape, dtype=torch.float32)
        else:
            # Fallback: assume some reasonable sequence length
            token_level_rewards = torch.zeros((batch_size, 512), dtype=torch.float32)

        # Return rewards (must include scores field for ROLL framework)
        return DataProto.from_dict(
            tensors={
                "token_level_rewards": token_level_rewards,
                "response_level_rewards": rewards,
                "scores": rewards,  # Same as response_level_rewards
            },
            non_tensors={}
        )


# Register worker
__all__ = ["PerformanceGainRewardWorker"]
