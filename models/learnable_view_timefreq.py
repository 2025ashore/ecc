# -*- coding: utf-8 -*-
"""learnable_view_timefreq.py

Route A implementation:
- Learn K time-domain views via a soft lead->view routing matrix W (K x 12).
- Add 1 time-frequency view from all 12 leads (LearnableSTFT + SpectrogramCNN).
- Fuse (K+1) view tokens via a lightweight view-level Transformer.
- Optional label-conditional view gating using y_soft (self-training style).
- Optional label graph refiner on logits.

This is designed to plug into main_train_timefreq.py.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model import MyNet
from models.timefreq_fusion import LearnableSTFT, SpectrogramCNN
from models.label_graph import LabelGraphRefiner


class LeadViewRouter(nn.Module):
    """Learnable soft routing from 12 leads -> K time-domain views."""

    def __init__(
        self,
        num_views: int,
        num_leads: int = 12,
        temperature: float = 1.0,
        init: str = "uniform",
    ):
        super().__init__()
        self.num_views = int(num_views)
        self.num_leads = int(num_leads)
        self.temperature = float(temperature)

        if init not in {"uniform", "random"}:
            raise ValueError(f"Unsupported init='{init}'")

        if init == "uniform":
            logits = torch.zeros(self.num_views, self.num_leads)
        else:
            logits = torch.randn(self.num_views, self.num_leads) * 0.01

        self.logits = nn.Parameter(logits)

    def weights(self) -> torch.Tensor:
        temp = max(self.temperature, 1e-6)
        return F.softmax(self.logits / temp, dim=-1)  # (K, 12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape (B, 12, L), got {tuple(x.shape)}")
        if x.shape[1] != self.num_leads:
            raise ValueError(f"Expected {self.num_leads} leads, got {x.shape[1]}")
        W = self.weights()
        return torch.einsum("bcl,kc->bkl", x, W)  # (B, K, L)


class TimeFreqView(nn.Module):
    """Time-frequency view over all 12 leads -> 128-d feature."""

    def __init__(
        self,
        input_channels: int = 12,
        output_dim: int = 128,
        n_fft: int = 512,
        hop_length: int = 128,
        n_scales: int = 3,
        normalize_spectrogram: bool = True,
        spectrogram_norm_type: str = "instance",
    ):
        super().__init__()
        self.normalize_spectrogram = bool(normalize_spectrogram)
        self.spectrogram_norm_type = str(spectrogram_norm_type)

        self.stft = LearnableSTFT(n_fft=n_fft, hop_length=hop_length, n_scales=n_scales)
        self.spectrogram_cnn = SpectrogramCNN(
            input_channels=input_channels,
            num_classes=output_dim,  # logits head not used; we take features
            hidden_dims=[64, 128, 256, 512],
        )
        self.proj = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim),
            nn.BatchNorm1d(output_dim),
        )

    def _normalize(self, spec: torch.Tensor) -> torch.Tensor:
        # spec: (B, C, F, T)
        if not self.normalize_spectrogram:
            return spec

        if self.spectrogram_norm_type == "instance":
            mean = spec.mean(dim=(2, 3), keepdim=True)
            std = spec.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
            return (spec - mean) / std

        if self.spectrogram_norm_type == "layer":
            mean = spec.mean(dim=(1, 2, 3), keepdim=True)
            std = spec.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
            return (spec - mean) / std

        return spec

    def forward(self, x: torch.Tensor, spec_augmentation=None) -> torch.Tensor:
        # x: (B, 12, L)
        fused_spec, _ = self.stft(x)  # (B, 12, F, T)
        fused_spec = self._normalize(fused_spec)

        if self.training and spec_augmentation is not None:
            fused_spec = spec_augmentation(fused_spec)

        feats, _, _ = self.spectrogram_cnn(fused_spec)  # (B, 512)
        return self.proj(feats)  # (B, 128)


class MyNetLearnableKViewTimeFreq(nn.Module):
    """Learnable K time-domain views + 1 time-frequency view + Transformer fusion."""

    def __init__(
        self,
        num_classes: int,
        num_views: int = 6,
        routing_temperature: float = 1.0,
        routing_init: str = "uniform",
        reg_entropy_weight: float = 0.0,
        reg_ortho_weight: float = 0.0,
        use_view_transformer_fusion: bool = True,
        view_transformer_layers: int = 1,
        view_transformer_heads: int = 4,
        view_transformer_dropout: float = 0.1,
        view_transformer_residual_scale: float = 0.2,
        use_label_conditional_view_gating: bool = True,
        fusion_temperature: float = 1.0,
        # time-freq params
        stft_n_fft: int = 512,
        stft_hop_length: int = 128,
        stft_n_scales: int = 3,
        normalize_spectrogram: bool = True,
        spectrogram_norm_type: str = "instance",
        # label graph
        use_label_graph_refiner: bool = False,
        label_graph_hidden: int = 64,
        label_graph_learnable_adj: bool = True,
        label_graph_dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_views = int(num_views)
        self.reg_entropy_weight = float(reg_entropy_weight)
        self.reg_ortho_weight = float(reg_ortho_weight)
        self.use_view_transformer_fusion = bool(use_view_transformer_fusion)
        self.view_transformer_residual_scale = float(view_transformer_residual_scale)
        self.use_label_conditional_view_gating = bool(use_label_conditional_view_gating)
        self.fusion_temperature = float(fusion_temperature)

        self.router = LeadViewRouter(
            num_views=self.num_views,
            num_leads=12,
            temperature=routing_temperature,
            init=routing_init,
        )

        # Shared encoder for each time-domain view (1 lead after mixing)
        self.time_view_encoder = MyNet(input_channels=1, single_view=True, num_classes=self.num_classes)

        # Time-frequency view
        self.timefreq_view = TimeFreqView(
            input_channels=12,
            output_dim=128,
            n_fft=stft_n_fft,
            hop_length=stft_hop_length,
            n_scales=stft_n_scales,
            normalize_spectrogram=normalize_spectrogram,
            spectrogram_norm_type=spectrogram_norm_type,
        )

        # Base view weighting (content-based)
        self.view_weight_fc = nn.Linear(128, 1)

        # Label-conditional gating: y_soft (B, C) -> (B, K+1)
        self.label_to_view = nn.Sequential(
            nn.Linear(self.num_classes, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, self.num_views + 1),
        )

        # View-level transformer
        self.view_pos_embed: Optional[nn.Parameter] = None
        self.view_transformer: Optional[nn.Module] = None
        self.view_transformer_norm: Optional[nn.Module] = None
        if self.use_view_transformer_fusion:
            self.view_pos_embed = nn.Parameter(torch.zeros(1, self.num_views + 1, 128))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=128,
                nhead=view_transformer_heads,
                dim_feedforward=256,
                dropout=view_transformer_dropout,
                batch_first=True,
                activation='gelu',
                norm_first=True,
            )
            self.view_transformer = nn.TransformerEncoder(encoder_layer, num_layers=view_transformer_layers)
            self.view_transformer_norm = nn.LayerNorm(128)

        self.fc = nn.Linear(128, self.num_classes)

        self.label_refiner: Optional[LabelGraphRefiner] = None
        if use_label_graph_refiner:
            self.label_refiner = LabelGraphRefiner(
                num_classes=self.num_classes,
                hidden=label_graph_hidden,
                dropout=label_graph_dropout,
                learnable_adj=label_graph_learnable_adj,
            )

    def get_routing_weights(self) -> torch.Tensor:
        return self.router.weights()  # (K, 12)

    def regularization_loss(self) -> torch.Tensor:
        W = self.get_routing_weights()
        loss = W.new_tensor(0.0)

        if self.reg_entropy_weight > 0:
            entropy = -(W * (W + 1e-12).log()).sum(dim=-1).mean()
            loss = loss + self.reg_entropy_weight * entropy

        if self.reg_ortho_weight > 0 and W.shape[0] > 1:
            G = W @ W.t()
            I = torch.eye(G.shape[0], device=G.device, dtype=G.dtype)
            loss = loss + self.reg_ortho_weight * ((G - I) ** 2).mean()

        return loss

    def _label_gate(self, y_soft: torch.Tensor) -> torch.Tensor:
        # returns (B, K+1) in [0,1]
        temp = max(self.fusion_temperature, 1e-6)
        gate_logits = self.label_to_view(y_soft)
        return torch.sigmoid(gate_logits / temp)

    def forward(
        self,
        x: torch.Tensor,
        y_soft: Optional[torch.Tensor] = None,
        spec_augmentation=None,
        return_intermediate: bool = False,
    ):
        # x: (B, 12, L)
        B, C, L = x.shape
        _ = C  # keep style

        # K time-domain views
        views = self.router(x)  # (B, K, L)
        time_in = views.reshape(B * self.num_views, 1, L)
        time_feats = self.time_view_encoder(time_in).reshape(B, self.num_views, 128)  # (B, K, 128)

        # +1 time-frequency view
        tf_feat = self.timefreq_view(x, spec_augmentation=spec_augmentation)  # (B, 128)
        tf_feat = tf_feat.unsqueeze(1)  # (B, 1, 128)

        view_tokens = torch.cat([time_feats, tf_feat], dim=1)  # (B, K+1, 128)

        # content-based weights
        w_raw = self.view_weight_fc(view_tokens)  # (B, K+1, 1)
        w = torch.sigmoid(w_raw)  # (B, K+1, 1)

        # label-conditional gating
        gate = None
        if self.use_label_conditional_view_gating and y_soft is not None:
            gate = self._label_gate(y_soft).unsqueeze(-1)  # (B, K+1, 1)
            # mild scaling to avoid instability
            token_scale = 0.5 + 0.5 * gate
            view_tokens = view_tokens * token_scale

        fused_sum = (w * view_tokens).sum(dim=1)  # (B, 128)

        fused_feat = fused_sum
        if self.use_view_transformer_fusion and self.view_transformer is not None:
            tokens = view_tokens
            if self.view_pos_embed is not None:
                tokens = tokens + self.view_pos_embed
            tokens = self.view_transformer(tokens)
            delta = tokens.mean(dim=1)
            delta = self.view_transformer_norm(delta) if self.view_transformer_norm is not None else delta
            fused_feat = fused_sum + self.view_transformer_residual_scale * delta

        logits = self.fc(fused_feat)
        if self.label_refiner is not None:
            logits = self.label_refiner(logits)

        if return_intermediate:
            intermediate: Dict[str, torch.Tensor] = {
                'routing_weights': self.get_routing_weights(),
                'time_view_features': time_feats,
                'timefreq_feature': tf_feat.squeeze(1),
                'view_weight_raw': w_raw.squeeze(-1),
                'view_weight_sigmoid': w.squeeze(-1),
                'label_gate': gate.squeeze(-1) if gate is not None else torch.empty(0),
                'fused_feat': fused_feat,
            }
            return logits, intermediate

        return logits
