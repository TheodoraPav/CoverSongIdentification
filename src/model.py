"""Frozen backbones + trainable projection head + temporal pooling.

Design notes:

*   Every backbone is loaded once (eval mode, all parameters frozen) during
    `extract_features.py` caching.
*   `last_hidden_state` is the single source of truth across MERT families.
    Pooling collapses the time axis into one vector.
*   Length correction (`augmentations.length_fix`) keeps every segment at the
    same number of samples per batch, so an attention mask is usually not
    needed. The pooling helper still supports a mask for variable-length
    (mixed) segments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Backbone registry
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class BackboneSpec:

    name: str
    checkpoint: str
    sample_rate: int
    hidden_dim: int
    input_type: str            # "waveform" or "mel"
    trust_remote_code: bool = False


BACKBONE_REGISTRY: dict[str, BackboneSpec] = {
    "mert": BackboneSpec(
        name="mert",
        checkpoint="m-a-p/MERT-v1-95M",
        sample_rate=24000,
        hidden_dim=768,
        input_type="waveform",
        trust_remote_code=True,
    ),
    "mert_large": BackboneSpec(
        name="mert_large",
        checkpoint="m-a-p/MERT-v1-330M",
        sample_rate=24000,
        hidden_dim=1024,
        input_type="waveform",
        trust_remote_code=True,
    ),
}


def get_backbone_spec(name: str) -> BackboneSpec:
    if name not in BACKBONE_REGISTRY:
        allowed = sorted(BACKBONE_REGISTRY.keys())
        raise KeyError(f"Unknown backbone {name!r}. Allowed: {allowed}")
    return BACKBONE_REGISTRY[name]


# -----------------------------------------------------------------------------
# Loading + forward
# -----------------------------------------------------------------------------


def load_backbone(
        name: str,
        checkpoint: str | None = None,
        device: torch.device | str | None = None,
) -> tuple[nn.Module, object, BackboneSpec]:
    """Load a frozen pretrained backbone + its HuggingFace processor.

    Returns (model, processor, spec). The model is moved to `device`, set to
    eval mode, and every parameter has `requires_grad=False`.
    """
    from transformers import AutoFeatureExtractor, AutoModel

    spec = get_backbone_spec(name)
    ckpt = checkpoint or spec.checkpoint

    processor = AutoFeatureExtractor.from_pretrained(
        ckpt, trust_remote_code=spec.trust_remote_code
    )
    model = AutoModel.from_pretrained(
        ckpt, trust_remote_code=spec.trust_remote_code
    )

    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    if device is not None:
        model.to(device)

    return model, processor, spec


def _prepare_processor_inputs(
        processor: object,
        waveforms: Sequence[np.ndarray] | np.ndarray,
        sample_rate: int,
) -> dict:
    """Run the HuggingFace processor and return a dict of tensors."""
    if isinstance(waveforms, np.ndarray):
        batch = list(waveforms) if waveforms.ndim == 2 else [waveforms]
    else:
        batch = list(waveforms)

    inputs = processor(
        batch,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
    )
    return dict(inputs)


def backbone_forward(
        model: nn.Module,
        processor: object,
        waveforms: Sequence[np.ndarray] | np.ndarray,
        sample_rate: int,
        spec: BackboneSpec,
        device: torch.device | str | None = None,
        mel_transform=None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run one frozen forward pass.

    If `mel_transform` is given and the backbone consumes mel input (AST), it
    is applied to `input_values` before the model. Used for SpecAugment.

    Returns `(hidden (B, T, D), mask (B, T) or None)`.
    """
    if device is None:
        device = next(model.parameters()).device

    inputs = _prepare_processor_inputs(processor, waveforms, sample_rate)
    if mel_transform is not None and spec.input_type == "mel":
        inputs["input_values"] = mel_transform(inputs["input_values"])

    inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)

    hidden = out.last_hidden_state  # (B, T, D)
    all_hidden = getattr(out, "hidden_states", None)  # tuple of 13 × (B, T, D) or None
    mask = _frame_level_mask(model, inputs.get("attention_mask"), hidden.shape[1])
    return hidden, mask, all_hidden


def _frame_level_mask(
        model: nn.Module,
        sample_mask: torch.Tensor | None,
        n_frames: int,
) -> torch.Tensor | None:
    """Map a sample level attention_mask to frame level (B, T).

    MERT exposes `_get_feature_vector_attention_mask`
    which accounts for the convolutional downsampling. If that helper is
    missing, or if the mask already matches `n_frames`, we just
    return what we have.
    """
    if sample_mask is None:
        return None
    if sample_mask.shape[-1] == n_frames:
        return sample_mask
    helper = getattr(model, "_get_feature_vector_attention_mask", None)
    if helper is None:
        return None
    try:
        return helper(n_frames, sample_mask)
    except Exception:  # noqa: BLE001 - frame mask is best effort
        return None


# -----------------------------------------------------------------------------
# Temporal pooling
# -----------------------------------------------------------------------------


def pool_backbone_output(
        hidden: torch.Tensor,
        mask: torch.Tensor | None = None,
        method: str = "mean",
) -> torch.Tensor:
    """Pool a `(B, T, D)` sequence into a single `(B, D)` vector and L2 normalize.

    `mask=None` means all frames are valid (the default for
    same length segments). Pass `mask` for the `mixed` sampling case so that
    padded silence does not bias the mean.
    """
    if hidden.dim() != 3:
        raise ValueError(f"Expected (B, T, D), got {tuple(hidden.shape)}")

    if method == "mean":
        if mask is not None:
            m = mask.to(hidden.dtype).unsqueeze(-1)
            summed = (hidden * m).sum(dim=1)
            counts = m.sum(dim=1).clamp_min(1.0)
            pooled = summed / counts
        else:
            pooled = hidden.mean(dim=1)
    elif method == "max":
        if mask is not None:
            invalid = (mask == 0).unsqueeze(-1)
            hidden = hidden.masked_fill(invalid, float("-inf"))
        pooled = hidden.max(dim=1).values
    else:
        raise ValueError(f"Unknown pool method: {method!r}. Use 'mean' or 'max'.")

    return F.normalize(pooled, p=2, dim=-1)


def pool_all_layers(
        all_hidden_states: tuple[torch.Tensor, ...],
        mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pool each layer's ``(B, T, D)`` to ``(B, D)`` and stack to ``(B, N_layers, D)``.

    Uses mean temporal pooling (same as the default ``pool_backbone_output``).
    The result is **not** L2-normalized — normalization is left to the
    projection head so that the layer-wise weighting operates on raw magnitudes.
    """
    pooled = []
    for h in all_hidden_states:
        if mask is not None:
            m = mask.to(h.dtype).unsqueeze(-1)
            p = (h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        else:
            p = h.mean(dim=1)
        pooled.append(p)
    return torch.stack(pooled, dim=1)  # (B, N_layers, D)


# -----------------------------------------------------------------------------
# Projection head
# -----------------------------------------------------------------------------


class LayerPooler(nn.Module):
    """Combine multiple MERT hidden-state layers into a single embedding.

    Modes (matching ``LAYER_POOLINGS`` in ``utils.py``):

    =============  ===========  =============================================
    Mode           Layers used  Description
    =============  ===========  =============================================
    last           12 only      Current baseline — last Transformer layer
    acoustic       0–3          Low-level spectral / timbre features
    early_mid      3–5          Transition zone (onset of beat tracking)
    mid_beat       4–6          Peak beat / rhythm features
    mid_pitch      6–8          Peak pitch / melody / key features
    semantic       9–11         High-level genre / instrument features
    musical_core   4–8          Combined beat + pitch + harmony
    mean_all       0–12         Equal-weight mean of all 13 layers
    learned_mix    0–12         13 learnable scalar weights + Softmax
    =============  ===========  =============================================
    """

    LAYER_RANGES: dict[str, tuple[int, int]] = {
        "acoustic":     (0, 4),   # layers 0, 1, 2, 3
        "early_mid":    (3, 6),   # layers 3, 4, 5
        "mid_beat":     (4, 7),   # layers 4, 5, 6
        "mid_pitch":    (6, 9),   # layers 6, 7, 8
        "semantic":     (9, 12),  # layers 9, 10, 11
        "musical_core": (4, 9),   # layers 4, 5, 6, 7, 8
    }

    def __init__(self, mode: str = "last", n_layers: int = 13) -> None:
        super().__init__()
        self.mode = mode
        self.n_layers = n_layers
        if mode == "learned_mix":
            # Initialised to zeros → uniform Softmax (1/13 each)
            self.layer_logits = nn.Parameter(torch.zeros(n_layers))

    def forward(self, all_hidden: torch.Tensor) -> torch.Tensor:
        """``all_hidden``: ``(B, N_layers, D)`` → output: ``(B, D)``."""
        if self.mode == "last":
            return all_hidden[:, -1, :]
        if self.mode in self.LAYER_RANGES:
            lo, hi = self.LAYER_RANGES[self.mode]
            return all_hidden[:, lo:hi, :].mean(dim=1)
        if self.mode == "mean_all":
            return all_hidden.mean(dim=1)
        if self.mode == "learned_mix":
            weights = F.softmax(self.layer_logits, dim=0)  # (N_layers,)
            return (all_hidden * weights.view(1, -1, 1)).sum(dim=1)
        raise ValueError(f"Unknown layer_pooling mode: {self.mode!r}")

    @torch.no_grad()
    def get_layer_weights(self) -> list[float]:
        """Return current Softmax weights (useful for visualisation after training)."""
        if self.mode != "learned_mix":
            return []
        return F.softmax(self.layer_logits, dim=0).cpu().tolist()


class ProjectionHead(nn.Module):
    """Trainable MLP projection head.

    Architecture (standard):
        Linear(D_in, hidden_dim) → (BatchNorm1d) → ReLU → Dropout(p) → Linear(hidden_dim, output_dim)
        → L2 normalize.

    Architecture (with chroma bottleneck, chroma_dim > 0):
        Linear(D_in, chroma_dim) → ReLU → Linear(chroma_dim, hidden_dim)
        → (BatchNorm1d) → ReLU → Dropout(p) → Linear(hidden_dim, output_dim)
        → L2 normalize.

    The chroma bottleneck forces the representation through a narrow
    pitch-class space (e.g. 24 = 2×12 semitones), stripping timbre
    information and retaining only harmonic content.

    The output sits on the unit sphere so that cosine similarity equals dot
    product, which keeps the triplet and NT-Xent losses well behaved.
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 512,
            output_dim: int = 128,
            dropout: float = 0.1,
            use_batchnorm: bool = True,
            chroma_dim: int = 0,
            layer_pooler: LayerPooler | None = None,
            feature_noise: float = 0.0,
            feature_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.chroma_dim = chroma_dim
        self.layer_pooler = layer_pooler
        self.feature_noise = float(feature_noise)
        self.feature_dropout = float(feature_dropout)
        
        if self.feature_dropout > 0.0:
            self.input_dropout = nn.Dropout(self.feature_dropout)
        else:
            self.input_dropout = None
        layers: list[nn.Module] = []
        if chroma_dim > 0:
            # Chroma bottleneck: input_dim → chroma_dim → hidden_dim
            layers.extend([
                nn.Linear(input_dim, chroma_dim),
                nn.ReLU(inplace=True),
                nn.Linear(chroma_dim, hidden_dim),
            ])
        else:
            layers.append(nn.Linear(input_dim, hidden_dim))

        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.extend([
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        ])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D) for single-layer mode, (B, N_layers, D) for multi-layer
        if self.layer_pooler is not None and x.dim() == 3:
            x = self.layer_pooler(x)  # (B, N_layers, D) → (B, D)
        if self.training:
            if self.input_dropout is not None:
                x = self.input_dropout(x)
            if self.feature_noise > 0.0:
                x = x + torch.randn_like(x) * self.feature_noise
        z = self.net(x)
        return F.normalize(z, p=2, dim=-1)


def resolve_projection_input_dim(cfg) -> int:
    """Pick `projection.input_dim` from the YAML, fall back to backbone metadata."""
    declared = getattr(cfg.projection, "input_dim", None)
    if declared is not None:
        return int(declared)
    return get_backbone_spec(cfg.backbone).hidden_dim


def build_projection_head(cfg) -> ProjectionHead:
    """Construct a `ProjectionHead` from an `ExperimentConfig`."""
    layer_pooling = getattr(cfg, "layer_pooling", "last")
    pooler = None
    if layer_pooling != "last":
        pooler = LayerPooler(mode=layer_pooling, n_layers=13)
    return ProjectionHead(
        input_dim=resolve_projection_input_dim(cfg),
        hidden_dim=cfg.projection.hidden_dim,
        output_dim=cfg.projection.output_dim,
        dropout=cfg.projection.dropout,
        use_batchnorm=getattr(cfg.projection, "batchnorm", True),
        chroma_dim=getattr(cfg.projection, "chroma_dim", 0),
        layer_pooler=pooler,
        feature_noise=getattr(cfg.projection, "feature_noise", 0.0),
        feature_dropout=getattr(cfg.projection, "feature_dropout", 0.0),
    )