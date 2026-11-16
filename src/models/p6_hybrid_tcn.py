"""P6: permutation-invariant player sets plus temporal convolution.

The model consumes the leak-safe schema-03 flat view, anonymous player slots,
and the separate ball trajectory from ``classification_inputs``.  Player
slots share one encoder and are mean-pooled by team, so arbitrary slot order
cannot affect the prediction.  A small residual TCN then supplies offline
temporal context before the per-frame H/A/D head.
"""

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class HybridBatch:
    """Typed tensor boundary for one or more continuous frame sequences."""

    flat: Tensor
    player_values: Tensor
    player_observed: Tensor
    player_inside: Tensor
    player_age_s: Tensor
    player_keeper: Tensor
    player_team: Tensor
    ball_values: Tensor
    ball_observed: Tensor
    ball_inside: Tensor
    ball_since_s: Tensor
    ball_until_s: Tensor

    def to(self, device: torch.device) -> "HybridBatch":
        return HybridBatch(**{name: value.to(device) for name, value in self.__dict__.items()})


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(values.dtype).unsqueeze(-1)
    return (values * weights).sum(dim=-2) / weights.sum(dim=-2).clamp_min(1.0)


def _age_summary(age_s: Tensor, mask: Tensor) -> Tensor:
    valid = mask & torch.isfinite(age_s)
    age = torch.log1p(torch.nan_to_num(age_s, nan=0.0).clamp(0.0, 30.0))
    weights = valid.to(age.dtype)
    return (age * weights).sum(dim=-1, keepdim=True) / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=1),
        )
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, values: Tensor) -> Tensor:
        return self.norm(values + self.net(values))


class HybridTCN(nn.Module):
    """Small P6 baseline with a 127-frame (about 5.1 s) receptive field."""

    def __init__(
        self,
        flat_features: int,
        hidden: int = 64,
        object_hidden: int = 16,
        dilations: Tuple[int, ...] = (1, 2, 4, 8, 16, 32),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if flat_features <= 0:
            raise ValueError("flat_features must be positive")
        if not dilations or min(dilations) <= 0:
            raise ValueError("dilations must be positive")

        self.player_encoder = nn.Sequential(
            nn.Linear(7, object_hidden),
            nn.GELU(),
            nn.Linear(object_hidden, object_hidden),
            nn.GELU(),
        )
        self.ball_encoder = nn.Sequential(
            nn.Linear(8, object_hidden),
            nn.GELU(),
            nn.Linear(object_hidden, object_hidden),
            nn.GELU(),
        )
        fused = flat_features + 2 * object_hidden + 2 + object_hidden
        self.input_projection = nn.Sequential(nn.Linear(fused, hidden), nn.GELU())
        self.temporal = nn.Sequential(
            *(ResidualTemporalBlock(hidden, dilation, dropout) for dilation in dilations)
        )
        self.head = nn.Linear(hidden, 3)

    @property
    def receptive_field_frames(self) -> int:
        return 1 + 2 * sum(block.net[0].dilation[0] for block in self.temporal)

    @property
    def history_frames(self) -> int:
        return self.receptive_field_frames // 2

    @property
    def future_frames(self) -> int:
        return self.receptive_field_frames // 2

    def forward(self, batch: HybridBatch) -> Tensor:
        """Return logits with shape ``[batch, frames, 3]`` in H/A/D order."""
        player = torch.nan_to_num(batch.player_values, nan=0.0).clone()
        player[..., :2] = player[..., :2].clamp(-1.5, 1.5)
        player[..., 2:] = (player[..., 2:].clamp(-12.0, 12.0) / 12.0)
        player_raw = torch.cat(
            (
                player,
                batch.player_inside.unsqueeze(-1).to(player.dtype),
                batch.player_keeper.unsqueeze(-1).to(player.dtype),
            ),
            dim=-1,
        )
        encoded = self.player_encoder(player_raw)
        team = batch.player_team
        if team.ndim == 1:
            team = team.unsqueeze(0).expand(player.shape[0], -1)
        home = (team[:, None, :] == 0) & batch.player_observed
        away = (team[:, None, :] == 1) & batch.player_observed
        home_pool = _masked_mean(encoded, home)
        away_pool = _masked_mean(encoded, away)
        missing = ~batch.player_observed
        home_age = _age_summary(batch.player_age_s, missing & (team[:, None, :] == 0))
        away_age = _age_summary(batch.player_age_s, missing & (team[:, None, :] == 1))

        ball = torch.nan_to_num(batch.ball_values, nan=0.0).clone()
        ball[..., :2] = ball[..., :2].clamp(-1.5, 1.5)
        ball[..., 2:] = ball[..., 2:].clamp(-30.0, 30.0) / 30.0
        ball_since = torch.log1p(torch.nan_to_num(batch.ball_since_s, nan=30.0).clamp(0.0, 30.0))
        ball_until = torch.log1p(torch.nan_to_num(batch.ball_until_s, nan=30.0).clamp(0.0, 30.0))
        ball_raw = torch.cat(
            (
                ball,
                batch.ball_observed.unsqueeze(-1).to(ball.dtype),
                batch.ball_inside.unsqueeze(-1).to(ball.dtype),
                ball_since.unsqueeze(-1),
                ball_until.unsqueeze(-1),
            ),
            dim=-1,
        )
        ball_context = self.ball_encoder(ball_raw)

        fused = torch.cat(
            (batch.flat, home_pool, away_pool, home_age, away_age, ball_context), dim=-1
        )
        temporal_input = self.input_projection(fused).transpose(1, 2)
        encoded_time = self.temporal(temporal_input).transpose(1, 2)
        return self.head(encoded_time)


class AsymmetricTemporalBlock(nn.Module):
    """Residual convolution with a declared past/future allocation."""

    def __init__(self, channels: int, dilation: int, left: int, dropout: float) -> None:
        super().__init__()
        total = 2 * dilation
        if left < 0 or left > total:
            raise ValueError("left padding must be between 0 and 2*dilation")
        self.left = left
        self.right = total - left
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation)
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, values: Tensor) -> Tensor:
        encoded = self.conv(F.pad(values, (self.left, self.right)))
        encoded = self.pointwise(self.dropout(F.gelu(encoded)))
        return self.norm(values + encoded)


class ObjectTrajectoryEncoder(nn.Module):
    """Apply the same acausal TCN independently to every object trajectory."""

    DILATIONS = (1, 2, 4, 8, 16, 19)
    LEFT_PADDING = (2, 3, 6, 12, 24, 28)

    def __init__(self, input_features: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.projection = nn.Conv1d(input_features, hidden, kernel_size=1)
        self.blocks = nn.Sequential(
            *(
                AsymmetricTemporalBlock(hidden, dilation, left, dropout)
                for dilation, left in zip(self.DILATIONS, self.LEFT_PADDING)
            )
        )

    @property
    def history_frames(self) -> int:
        return sum(self.LEFT_PADDING)

    @property
    def future_frames(self) -> int:
        return sum(2 * dilation - left for dilation, left in zip(self.DILATIONS, self.LEFT_PADDING))

    def forward(self, values: Tensor) -> Tensor:
        return self.blocks(self.projection(values))


class TrajectorySetTCN(nn.Module):
    """P6 v1: temporal encoding per object before permutation-invariant pooling."""

    def __init__(
        self,
        flat_features: int,
        hidden: int = 64,
        object_hidden: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if flat_features <= 0:
            raise ValueError("flat_features must be positive")
        self.player_temporal = ObjectTrajectoryEncoder(9, object_hidden, dropout)
        self.ball_temporal = ObjectTrajectoryEncoder(8, object_hidden, dropout)
        self.flat_projection = nn.Sequential(nn.Linear(flat_features, 32), nn.GELU())
        self.head = nn.Sequential(
            nn.Linear(32 + 3 * object_hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3),
        )

    @property
    def history_frames(self) -> int:
        return self.player_temporal.history_frames

    @property
    def future_frames(self) -> int:
        return self.player_temporal.future_frames

    @property
    def receptive_field_frames(self) -> int:
        return self.history_frames + 1 + self.future_frames

    def forward(self, batch: HybridBatch) -> Tensor:
        """Return H/A/D logits after object-level temporal encoding and team pooling."""
        player = torch.nan_to_num(batch.player_values, nan=0.0).clone()
        player[..., :2] = player[..., :2].clamp(-1.5, 1.5)
        player[..., 2:] = player[..., 2:].clamp(-12.0, 12.0) / 12.0
        player_age = torch.log1p(
            torch.nan_to_num(batch.player_age_s, nan=30.0).clamp(0.0, 30.0)
        )
        player_raw = torch.cat(
            (
                player,
                batch.player_observed.unsqueeze(-1).to(player.dtype),
                batch.player_inside.unsqueeze(-1).to(player.dtype),
                player_age.unsqueeze(-1),
                batch.player_keeper.unsqueeze(-1).to(player.dtype),
            ),
            dim=-1,
        )
        batch_size, frames, slots, channels = player_raw.shape
        player_sequence = player_raw.permute(0, 2, 3, 1).reshape(batch_size * slots, channels, frames)
        player_encoded = self.player_temporal(player_sequence)
        player_encoded = player_encoded.reshape(batch_size, slots, -1, frames).permute(0, 3, 1, 2)

        observed_sequence = batch.player_observed.permute(0, 2, 1).reshape(batch_size * slots, 1, frames)
        observed_window = F.conv1d(
            F.pad(observed_sequence.to(player.dtype), (self.history_frames, self.future_frames)),
            torch.ones(1, 1, self.receptive_field_frames, device=player.device),
        ) > 0
        observed_window = observed_window.reshape(batch_size, slots, frames).permute(0, 2, 1)
        team = batch.player_team
        if team.ndim == 1:
            team = team.unsqueeze(0).expand(batch_size, -1)
        home = observed_window & (team[:, None, :] == 0)
        away = observed_window & (team[:, None, :] == 1)
        home_pool = _masked_mean(player_encoded, home)
        away_pool = _masked_mean(player_encoded, away)

        ball = torch.nan_to_num(batch.ball_values, nan=0.0).clone()
        ball[..., :2] = ball[..., :2].clamp(-1.5, 1.5)
        ball[..., 2:] = ball[..., 2:].clamp(-30.0, 30.0) / 30.0
        ball_since = torch.log1p(torch.nan_to_num(batch.ball_since_s, nan=30.0).clamp(0.0, 30.0))
        ball_until = torch.log1p(torch.nan_to_num(batch.ball_until_s, nan=30.0).clamp(0.0, 30.0))
        ball_raw = torch.cat(
            (
                ball,
                batch.ball_observed.unsqueeze(-1).to(ball.dtype),
                batch.ball_inside.unsqueeze(-1).to(ball.dtype),
                ball_since.unsqueeze(-1),
                ball_until.unsqueeze(-1),
            ),
            dim=-1,
        )
        ball_encoded = self.ball_temporal(ball_raw.transpose(1, 2)).transpose(1, 2)
        fused = torch.cat(
            (self.flat_projection(batch.flat), home_pool, away_pool, ball_encoded), dim=-1
        )
        return self.head(fused)

