"""Independent TR/RD branches with trajectory-level soft-cascade fusion."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


CLASS_NAMES = ["drone", "bird", "balloon", "clutter", "other"]
B01_LABELS = [
    "DroneTarget",
    "BirdTarget",
    "BalloonTarget",
    "ClutterTarget",
    "UnknownTarget",
]


class TransformerTrackEncoder(nn.Module):
    def __init__(self, input_dim: int = 15, model_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=4,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, features: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        hidden = self.input_proj(features)
        if padding_mask is None:
            return self.norm(self.encoder(hidden))
        padding_mask = padding_mask.bool()
        valid_rows = (~padding_mask).any(dim=1)
        if not bool(valid_rows.any()):
            return torch.zeros_like(hidden)
        encoded = torch.zeros_like(hidden)
        valid_hidden = self.encoder(hidden[valid_rows], src_key_padding_mask=padding_mask[valid_rows])
        valid_hidden = self.norm(valid_hidden).masked_fill(padding_mask[valid_rows].unsqueeze(-1), 0.0)
        encoded[valid_rows] = valid_hidden
        return encoded


class LearnableAttentionPool(nn.Module):
    def __init__(self, dim: int = 128) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.scale = dim**0.5

    def forward(self, values: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        scores = torch.matmul(values, self.query.squeeze(0).transpose(0, 1)).squeeze(-1) / self.scale
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask, -1e9)
        weights = torch.softmax(scores, dim=-1)
        if padding_mask is not None:
            weights = torch.where(
                (~padding_mask).any(dim=-1, keepdim=True),
                weights,
                torch.zeros_like(weights),
            )
        return torch.sum(weights.unsqueeze(-1) * values, dim=1)


class TrackRepresentationBuilder(nn.Module):
    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.sequence_encoder = TransformerTrackEncoder(dropout=dropout)
        self.physical_norm = nn.LayerNorm(22)
        self.physical_proj = nn.Sequential(
            nn.Linear(22, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 32),
        )
        self.attention_pool = LearnableAttentionPool(128)
        self.fusion = nn.Sequential(
            nn.Linear(160, 160),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(160),
        )

    def forward(self, sequence: Tensor, physical: Tensor, padding_mask: Tensor) -> Tensor:
        encoded = self.sequence_encoder(sequence, padding_mask=padding_mask)
        deep = self.attention_pool(encoded, padding_mask=padding_mask)
        physical_encoded = self.physical_proj(self.physical_norm(physical))
        return self.fusion(torch.cat([deep, physical_encoded], dim=-1))


class FusedFeatureMLPBackbone(nn.Module):
    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(160, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(512),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.network(values)


class CosineMarginHead(nn.Module):
    def __init__(self, scale: float = 16.0, margin: float = 0.2) -> None:
        super().__init__()
        self.scale = float(scale)
        self.margin = float(margin)
        self.weight = nn.Parameter(torch.empty(5, 512))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features: Tensor, labels: Tensor | None = None) -> Tensor:
        cosine = F.normalize(features, dim=-1) @ F.normalize(self.weight, dim=-1).transpose(0, 1)
        if labels is not None:
            one_hot = torch.zeros_like(cosine)
            valid = labels >= 0
            one_hot[valid, labels[valid]] = 1.0
            cosine = cosine - one_hot * self.margin
        return cosine * self.scale

class TrajectoryBranch(nn.Module):
    """Standalone B01-compatible TR-only five-class branch."""

    def __init__(self, dropout: float = 0.1, scale: float = 16.0, margin: float = 0.2) -> None:
        super().__init__()
        self.representation_builder = TrackRepresentationBuilder(dropout=dropout)
        self.traj_backbone = FusedFeatureMLPBackbone(dropout=dropout)
        self.traj_head = CosineMarginHead(scale=scale, margin=margin)
        self.register_buffer("source_center", torch.zeros(15), persistent=True)
        self.register_buffer("source_scale", torch.ones(15), persistent=True)

    def forward(
        self,
        sequence: Tensor,
        physical: Tensor,
        padding_mask: Tensor,
        labels: Tensor | None = None,
    ) -> Tensor:
        normalized = (sequence - self.source_center) / self.source_scale.clamp_min(1e-3)
        fused = self.representation_builder(normalized, physical, padding_mask)
        return self.traj_head(self.traj_backbone(fused), labels=labels)


def load_b01_trajectory_branch(
    checkpoint_path: Path | str,
    *,
    map_location: str | torch.device = "cpu",
) -> TrajectoryBranch:
    """Load only the TR-owned tensors from the authoritative B01 checkpoint."""
    checkpoint = torch.load(Path(checkpoint_path), map_location=map_location, weights_only=False)
    if checkpoint.get("labels") != B01_LABELS:
        raise ValueError(f"unexpected B01 label order: {checkpoint.get('labels')}")
    config = checkpoint.get("config", {})
    if config.get("sequence_encoder_type") != "transformer":
        raise ValueError("checkpoint is not the B01 Transformer trajectory branch")
    model = TrajectoryBranch(
        dropout=float(config.get("dropout", 0.1)),
        scale=float(config.get("cosface_scale", 16.0)),
        margin=float(config.get("cosface_margin", 0.2)),
    )
    model.representation_builder.load_state_dict(checkpoint["representation_builder_state_dict"], strict=True)
    classifier_state = {
        key[len("traj_backbone."):]: value
        for key, value in checkpoint["model_state_dict"].items()
        if key.startswith("traj_backbone.")
    }
    model.traj_backbone.load_state_dict(classifier_state, strict=True)
    model.traj_head.load_state_dict(
        {
            key[len("traj_head."):]: value
            for key, value in checkpoint["model_state_dict"].items()
            if key.startswith("traj_head.")
        },
        strict=True,
    )
    stats = checkpoint.get("source_feature_stats", {}).get("cq08_track")
    if not isinstance(stats, dict) or "center" not in stats or "scale" not in stats:
        raise ValueError("B01 checkpoint lacks cq08_track source normalization")
    model.source_center.copy_(torch.as_tensor(stats["center"], dtype=torch.float32))
    model.source_scale.copy_(torch.as_tensor(stats["scale"], dtype=torch.float32))
    return model


def load_checkpoint_metadata(
    checkpoint_path: Path | str,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, object]:
    """Read checkpoint metadata without constructing a branch.

    B01 stores the selected epoch at the top level.  Keeping this helper
    separate prevents evaluation/reporting code from assuming a fixed epoch.
    """
    checkpoint = torch.load(Path(checkpoint_path), map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"checkpoint must contain a mapping: {checkpoint_path}")
    return checkpoint


def _entropy(probabilities: Tensor) -> Tensor:
    return -(probabilities.clamp_min(1e-8).log() * probabilities).sum(dim=-1) / math.log(probabilities.shape[-1])


def _margin(probabilities: Tensor) -> Tensor:
    top_two = probabilities.topk(2, dim=-1).values
    return top_two[:, 0] - top_two[:, 1]


@dataclass(frozen=True)
class AggregatedRDEvidence:
    frame_logits: Tensor
    probabilities: Tensor
    predictions: Tensor
    available: Tensor
    frame_count: Tensor
    consistency: Tensor


def aggregate_rd_evidence(frame_logits: Tensor, frame_to_track: Tensor, track_count: int) -> AggregatedRDEvidence:
    if frame_logits.ndim != 2 or frame_logits.shape[1] != len(CLASS_NAMES):
        raise ValueError("RD frame logits must have shape [frames, 5]")
    if frame_to_track.ndim != 1 or frame_to_track.shape[0] != frame_logits.shape[0]:
        raise ValueError("frame_to_track must contain one track index per RD frame")
    if frame_to_track.numel() and (int(frame_to_track.min()) < 0 or int(frame_to_track.max()) >= track_count):
        raise ValueError("frame_to_track contains an out-of-range trajectory index")
    frame_probabilities = torch.softmax(frame_logits, dim=-1)
    sums = frame_probabilities.new_zeros((track_count, len(CLASS_NAMES)))
    counts = frame_probabilities.new_zeros(track_count)
    sums.index_add_(0, frame_to_track, frame_probabilities)
    counts.index_add_(0, frame_to_track, torch.ones_like(frame_to_track, dtype=frame_probabilities.dtype))
    available = counts > 0
    probabilities = sums / counts.clamp_min(1.0).unsqueeze(-1)
    probabilities = torch.where(
        available.unsqueeze(-1),
        probabilities,
        torch.full_like(probabilities, 1.0 / len(CLASS_NAMES)),
    )
    total_variation = 0.5 * (frame_probabilities - probabilities[frame_to_track]).abs().sum(dim=-1)
    variation_sum = frame_probabilities.new_zeros(track_count)
    variation_sum.index_add_(0, frame_to_track, total_variation)
    consistency = 1.0 - variation_sum / counts.clamp_min(1.0)
    consistency = torch.where(available, consistency.clamp(0.0, 1.0), torch.zeros_like(consistency))
    return AggregatedRDEvidence(
        frame_logits=frame_logits,
        probabilities=probabilities,
        predictions=probabilities.argmax(dim=-1),
        available=available,
        frame_count=counts,
        consistency=consistency,
    )


class QualityAwareClassGate(nn.Module):
    """Predict a separate RD contribution for every class and trajectory."""

    def __init__(self, classes: int = 5, hidden_dim: int = 32, initial_rd_weight: float = 0.2) -> None:
        super().__init__()
        if not 0.0 < initial_rd_weight < 1.0:
            raise ValueError("initial_rd_weight must be between zero and one")
        self.network = nn.Sequential(
            nn.Linear(classes * 2 + 6, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, classes),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.constant_(self.network[-1].bias, math.log(initial_rd_weight / (1.0 - initial_rd_weight)))

    def forward(self, tr_probabilities: Tensor, rd: AggregatedRDEvidence) -> Tensor:
        quality = torch.stack(
            [
                _entropy(tr_probabilities),
                _entropy(rd.probabilities),
                _margin(tr_probabilities),
                _margin(rd.probabilities),
                rd.consistency,
                torch.log1p(rd.frame_count) / 8.0,
            ],
            dim=-1,
        )
        weights = torch.sigmoid(self.network(torch.cat([tr_probabilities, rd.probabilities, quality], dim=-1)))
        return weights * rd.available.unsqueeze(-1)


@dataclass(frozen=True)
class SoftCascadeOutput:
    tr_logits: Tensor
    tr_probabilities: Tensor
    tr_predictions: Tensor
    rd_frame_logits: Tensor
    rd_probabilities: Tensor
    rd_predictions: Tensor
    rd_available: Tensor
    rd_frame_count: Tensor
    rd_consistency: Tensor
    rd_class_weights: Tensor
    fused_probabilities: Tensor
    fused_predictions: Tensor


class SoftCascadeFusion(nn.Module):
    """Fuse independent branch probabilities without discarding either decision."""

    def __init__(
        self,
        *,
        mode: str = "quality_classwise",
        fixed_rd_weight: float | Sequence[float] = 0.2,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        if mode not in {"fixed", "quality_classwise"}:
            raise ValueError(f"unsupported fusion mode: {mode}")
        self.mode = mode
        fixed = torch.as_tensor(fixed_rd_weight, dtype=torch.float32)
        if fixed.ndim == 0:
            fixed = fixed.repeat(len(CLASS_NAMES))
        if tuple(fixed.shape) != (len(CLASS_NAMES),) or bool(((fixed < 0.0) | (fixed > 1.0)).any()):
            raise ValueError("fixed_rd_weight must be a scalar or five values in [0, 1]")
        self.register_buffer("fixed_rd_weight", fixed)
        initial = float(fixed.mean().clamp(1e-4, 1.0 - 1e-4))
        self.gate = QualityAwareClassGate(hidden_dim=hidden_dim, initial_rd_weight=initial)

    def forward(self, tr_logits: Tensor, rd_frame_logits: Tensor, frame_to_track: Tensor) -> SoftCascadeOutput:
        if tr_logits.ndim != 2 or tr_logits.shape[1] != len(CLASS_NAMES):
            raise ValueError("TR logits must have shape [tracks, 5]")
        tr_probabilities = torch.softmax(tr_logits, dim=-1)
        rd = aggregate_rd_evidence(rd_frame_logits, frame_to_track, tr_logits.shape[0])
        if self.mode == "fixed":
            rd_weights = self.fixed_rd_weight.unsqueeze(0).expand_as(tr_probabilities)
            rd_weights = rd_weights * rd.available.unsqueeze(-1)
        else:
            rd_weights = self.gate(tr_probabilities, rd)
        fused = rd_weights * rd.probabilities + (1.0 - rd_weights) * tr_probabilities
        fused = fused / fused.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return SoftCascadeOutput(
            tr_logits=tr_logits,
            tr_probabilities=tr_probabilities,
            tr_predictions=tr_probabilities.argmax(dim=-1),
            rd_frame_logits=rd.frame_logits,
            rd_probabilities=rd.probabilities,
            rd_predictions=rd.predictions,
            rd_available=rd.available,
            rd_frame_count=rd.frame_count,
            rd_consistency=rd.consistency,
            rd_class_weights=rd_weights,
            fused_probabilities=fused,
            fused_predictions=fused.argmax(dim=-1),
        )

    def fuse_probabilities(
        self,
        tr_probabilities: Tensor,
        rd: AggregatedRDEvidence,
    ) -> tuple[Tensor, Tensor]:
        """Fuse already aggregated trajectory probabilities.

        This is used by OOF gate fitting.  It deliberately reuses the same
        weight calculation as :meth:`forward` while retaining RD availability
        and frame-quality features from the caller.
        """
        if tr_probabilities.ndim != 2 or tr_probabilities.shape[1] != len(CLASS_NAMES):
            raise ValueError("TR probabilities must have shape [tracks, 5]")
        if tr_probabilities.shape[0] != rd.probabilities.shape[0]:
            raise ValueError("TR/RD trajectory counts do not match")
        if self.mode == "fixed":
            weights = self.fixed_rd_weight.unsqueeze(0).expand_as(tr_probabilities)
            weights = weights * rd.available.unsqueeze(-1)
        else:
            weights = self.gate(tr_probabilities, rd)
        fused = weights * rd.probabilities + (1.0 - weights) * tr_probabilities
        fused = fused / fused.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return fused, weights


class TrajectoryRDFusionModel(nn.Module):
    """End-to-end container while retaining explicit ownership by each branch."""

    def __init__(self, trajectory_branch: nn.Module, rd_branch: nn.Module, fusion: SoftCascadeFusion) -> None:
        super().__init__()
        self.trajectory_branch = trajectory_branch
        self.rd_branch = rd_branch
        self.fusion = fusion

    def freeze_branches(self) -> None:
        for branch in (self.trajectory_branch, self.rd_branch):
            branch.eval()
            for parameter in branch.parameters():
                parameter.requires_grad_(False)

    def forward(
        self,
        sequence: Tensor,
        physical: Tensor,
        padding_mask: Tensor,
        rd_images: Tensor,
        rd_frame_to_track: Tensor,
    ) -> SoftCascadeOutput:
        tr_logits = self.trajectory_branch(sequence, physical, padding_mask)
        rd_frame_logits = self.rd_branch(rd_images)
        return self.fusion(tr_logits, rd_frame_logits, rd_frame_to_track)
