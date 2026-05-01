"""Arquiteturas alternativas para o A/B do Sprint 2.

Variantes (alem do `BrandSimilarityMLP` legado em `mlp.py`):

  - `TwoTowerCrossAttention`: separa features em dois grupos ("nome" e
    "contexto"), cada um passa por um MLP, e a fusao e feita por
    cross-attention multi-head antes da head final.
  - `FTTransformer`: cada feature continua vira um token (Linear+LayerNorm),
    processado por blocos Transformer, depois CLS -> MLP -> logit.
  - `MultiTaskMLP`: backbone unico com 4 heads:
      * main: P(colide) -> BCE/Focal
      * aux1: spec_cosine_emb -> MSE
      * aux2: cls_same -> BCE
      * aux3: name_high_sim (graf_levenshtein > 0.6) -> BCE
    Em inferencia retornamos apenas a head principal.

Todas as arquiteturas expoem `forward(x)` que retorna **logits** principais
(shape `(N,1)`), compativel com `compute_metrics_at_threshold` e com as
losses BCE/Focal. Para `MultiTaskMLP`, use `forward_all_heads(x)` quando
precisar das auxiliares (treino).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from .mlp import BrandSimilarityMLP, MLPConfig, _activation


# =============================================================================
# Helpers
# =============================================================================

def _make_mlp_block(
    input_dim: int,
    hidden_dims: list[int],
    dropout: float,
    use_bn: bool,
    activation: str = "relu",
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        if use_bn:
            layers.append(nn.BatchNorm1d(h))
        layers.append(_activation(activation))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    return nn.Sequential(*layers)


def split_feature_indices(
    feature_names: list[str],
) -> tuple[list[int], list[int]]:
    """Divide features em (name_idx, context_idx) usando prefixos canonicos.

    Tower NOME (graf_, fon_, tok_, name_, contain_, radical_, lev_pure_,
    brand_, ofta_, anagram, len_).
    Tower CONTEXTO (spec_, cls_, against_).

    Features de interaction (`inter_*`) ficam no tower NOME por usarem
    blocos nominais como base.
    """
    NAME_PREFIXES = (
        "graf_", "fon_", "tok_", "name_", "contain_", "radical_",
        "lev_pure_", "brand_", "ofta_", "anagram", "len_",
        "inter_",
    )
    CONTEXT_PREFIXES = (
        "spec_", "cls_", "against_",
    )

    name_idx: list[int] = []
    ctx_idx: list[int] = []
    for i, name in enumerate(feature_names):
        if name.startswith(CONTEXT_PREFIXES):
            ctx_idx.append(i)
        elif name.startswith(NAME_PREFIXES):
            name_idx.append(i)
        else:
            ctx_idx.append(i)
    return name_idx, ctx_idx


# =============================================================================
# Two-Tower com Cross-Attention
# =============================================================================

@dataclass
class TwoTowerConfig:
    name_dim: int
    ctx_dim: int
    name_idx: list[int]
    ctx_idx: list[int]
    tower_hidden: list[int] = field(default_factory=lambda: [128, 64])
    embed_dim: int = 64
    n_heads: int = 4
    dropout: float = 0.3
    use_batchnorm: bool = True
    head_hidden: list[int] = field(default_factory=lambda: [128, 64])

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "two_tower",
            "name_dim": self.name_dim,
            "ctx_dim": self.ctx_dim,
            "name_idx": list(self.name_idx),
            "ctx_idx": list(self.ctx_idx),
            "tower_hidden": list(self.tower_hidden),
            "embed_dim": self.embed_dim,
            "n_heads": self.n_heads,
            "dropout": self.dropout,
            "use_batchnorm": self.use_batchnorm,
            "head_hidden": list(self.head_hidden),
        }


class TwoTowerCrossAttention(nn.Module):
    """Two-Tower: tower NOME e tower CONTEXTO + cross-attention final.

    forward(x): x e (B, F) inteiro - separamos internamente pelas listas de
    indices fornecidas no config.
    """

    def __init__(self, cfg: TwoTowerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer(
            "name_idx_buf",
            torch.tensor(cfg.name_idx, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "ctx_idx_buf",
            torch.tensor(cfg.ctx_idx, dtype=torch.long),
            persistent=True,
        )

        self.name_tower = _make_mlp_block(
            cfg.name_dim, cfg.tower_hidden + [cfg.embed_dim],
            dropout=cfg.dropout, use_bn=cfg.use_batchnorm,
        )
        self.ctx_tower = _make_mlp_block(
            cfg.ctx_dim, cfg.tower_hidden + [cfg.embed_dim],
            dropout=cfg.dropout, use_bn=cfg.use_batchnorm,
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=cfg.embed_dim,
            num_heads=cfg.n_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(cfg.embed_dim)

        # Head: concat(name_repr, ctx_repr, attn_out) -> MLP -> 1
        head_in = cfg.embed_dim * 3
        self.head = _make_mlp_block(
            head_in, cfg.head_hidden,
            dropout=cfg.dropout, use_bn=cfg.use_batchnorm,
        )
        self.out = nn.Linear(cfg.head_hidden[-1] if cfg.head_hidden else head_in, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_name = x.index_select(1, self.name_idx_buf)
        x_ctx = x.index_select(1, self.ctx_idx_buf)

        h_name = self.name_tower(x_name)
        h_ctx = self.ctx_tower(x_ctx)

        # cross-attention: query=name, key/value=ctx
        q = h_name.unsqueeze(1)
        kv = h_ctx.unsqueeze(1)
        attn_out, _ = self.cross_attn(q, kv, kv)
        attn_out = self.attn_norm(attn_out.squeeze(1) + h_name)

        z = torch.cat([h_name, h_ctx, attn_out], dim=1)
        z = self.head(z)
        return self.out(z)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return torch.sigmoid(self.forward(x))


# =============================================================================
# FT-Transformer
# =============================================================================

@dataclass
class FTTransformerConfig:
    input_dim: int
    embed_dim: int = 16
    n_layers: int = 2
    n_heads: int = 4
    ffn_dim: int = 64
    dropout: float = 0.1

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "ft_transformer",
            "input_dim": self.input_dim,
            "embed_dim": self.embed_dim,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "ffn_dim": self.ffn_dim,
            "dropout": self.dropout,
        }


class FeatureTokenizer(nn.Module):
    """Cada feature continua (escalar) vira um embedding (Linear + LayerNorm).

    Equivalente ao "Numerical Feature Tokenizer" do paper FT-Transformer
    (Gorishniy et al. 2021). Adiciona 1 token CLS.
    """

    def __init__(self, n_features: int, embed_dim: int) -> None:
        super().__init__()
        self.n_features = n_features
        self.embed_dim = embed_dim
        self.weights = nn.Parameter(torch.randn(n_features, embed_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_features, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F) -> (B, F, D)
        tokens = x.unsqueeze(-1) * self.weights.unsqueeze(0) + self.bias.unsqueeze(0)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        out = torch.cat([cls, tokens], dim=1)
        return self.norm(out)


class FTTransformer(nn.Module):
    """FT-Transformer simplificado (CLS no inicio, blocos Transformer padrao)."""

    def __init__(self, cfg: FTTransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tokenizer = FeatureTokenizer(cfg.input_dim, cfg.embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.embed_dim,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.embed_dim),
            nn.Linear(cfg.embed_dim, cfg.embed_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.embed_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tok = self.tokenizer(x)
        z = self.encoder(tok)
        cls = z[:, 0, :]
        return self.head(cls)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return torch.sigmoid(self.forward(x))


# =============================================================================
# Multi-Task MLP
# =============================================================================

@dataclass
class MultiTaskConfig:
    input_dim: int
    backbone_hidden: list[int] = field(default_factory=lambda: [256, 128, 64])
    dropout: float = 0.45
    use_batchnorm: bool = True
    aux_indices: dict[str, int] | None = None  # {"spec_cosine_emb": idx, ...}

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "multitask",
            "input_dim": self.input_dim,
            "backbone_hidden": list(self.backbone_hidden),
            "dropout": self.dropout,
            "use_batchnorm": self.use_batchnorm,
            "aux_indices": self.aux_indices,
        }


class MultiTaskMLP(nn.Module):
    """Backbone + 4 heads (main + 3 auxiliares).

    Em inferencia, somente `forward(x)` (head principal) e usado.
    `forward_all_heads(x)` retorna dict com todas as saidas para a loss
    multi-tarefa.
    """

    def __init__(self, cfg: MultiTaskConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = _make_mlp_block(
            cfg.input_dim, cfg.backbone_hidden,
            dropout=cfg.dropout, use_bn=cfg.use_batchnorm,
        )
        last = cfg.backbone_hidden[-1] if cfg.backbone_hidden else cfg.input_dim
        self.head_main = nn.Linear(last, 1)
        self.head_spec = nn.Linear(last, 1)
        self.head_cls = nn.Linear(last, 1)
        self.head_namesim = nn.Linear(last, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        return self.head_main(z)

    def forward_all_heads(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.backbone(x)
        return {
            "main": self.head_main(z),
            "aux_spec": self.head_spec(z),
            "aux_cls": self.head_cls(z),
            "aux_namesim": self.head_namesim(z),
        }

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return torch.sigmoid(self.forward(x))


# =============================================================================
# Factory
# =============================================================================

def build_model(
    architecture: str,
    input_dim: int,
    feature_names: list[str] | None = None,
    *,
    mlp_hidden: list[int] | None = None,
    dropout: float = 0.3,
    use_batchnorm: bool = True,
    activation: str = "relu",
) -> nn.Module:
    """Cria um modelo a partir de string.

    architecture in {"mlp", "two_tower", "ft_transformer", "multitask"}.
    Para "two_tower" e necessario `feature_names` para particionar as colunas.
    """
    arch = architecture.lower()
    if arch in ("mlp", "brand_mlp", "baseline"):
        return BrandSimilarityMLP(
            MLPConfig(
                input_dim=input_dim,
                hidden_dims=mlp_hidden or [128, 64, 32],
                dropout=dropout,
                use_batchnorm=use_batchnorm,
                activation=activation,
            )
        )
    if arch in ("two_tower", "twotower", "two-tower"):
        if not feature_names:
            raise ValueError("two_tower exige feature_names.")
        name_idx, ctx_idx = split_feature_indices(feature_names)
        if not name_idx:
            name_idx = list(range(min(8, input_dim)))
        if not ctx_idx:
            ctx_idx = [i for i in range(input_dim) if i not in name_idx]
        cfg = TwoTowerConfig(
            name_dim=len(name_idx),
            ctx_dim=len(ctx_idx),
            name_idx=name_idx,
            ctx_idx=ctx_idx,
            tower_hidden=mlp_hidden or [128, 64],
            dropout=dropout,
            use_batchnorm=use_batchnorm,
        )
        return TwoTowerCrossAttention(cfg)
    if arch in ("ft_transformer", "fttransformer", "ft-transformer", "ftt"):
        cfg = FTTransformerConfig(input_dim=input_dim, dropout=dropout)
        return FTTransformer(cfg)
    if arch in ("multitask", "multi_task", "mt"):
        aux = None
        if feature_names:
            aux = {}
            for tgt in ("spec_cosine_emb", "cls_same", "graf_levenshtein"):
                if tgt in feature_names:
                    aux[tgt] = feature_names.index(tgt)
        cfg = MultiTaskConfig(
            input_dim=input_dim,
            backbone_hidden=mlp_hidden or [256, 128, 64],
            dropout=dropout,
            use_batchnorm=use_batchnorm,
            aux_indices=aux,
        )
        return MultiTaskMLP(cfg)
    raise ValueError(f"Arquitetura desconhecida: {architecture}")


__all__ = [
    "TwoTowerConfig", "TwoTowerCrossAttention",
    "FTTransformerConfig", "FTTransformer",
    "MultiTaskConfig", "MultiTaskMLP",
    "split_feature_indices", "build_model",
]
