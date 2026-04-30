"""MLP nao-linear configuravel para similaridade aprendida de marcas.

Saida e logit (linear) - aplicar sigmoid externamente para inferencia ou usar
BCEWithLogitsLoss para treino (mais estavel numericamente).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn


@dataclass
class MLPConfig:
    input_dim: int
    hidden_dims: list[int] = field(default_factory=lambda: [128, 64, 32])
    dropout: float = 0.3
    use_batchnorm: bool = True
    activation: str = "relu"

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "hidden_dims": list(self.hidden_dims),
            "dropout": self.dropout,
            "use_batchnorm": self.use_batchnorm,
            "activation": self.activation,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MLPConfig":
        return cls(
            input_dim=int(d["input_dim"]),
            hidden_dims=list(d.get("hidden_dims", [128, 64, 32])),
            dropout=float(d.get("dropout", 0.3)),
            use_batchnorm=bool(d.get("use_batchnorm", True)),
            activation=str(d.get("activation", "relu")),
        )


def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "leakyrelu":
        return nn.LeakyReLU(0.1, inplace=True)
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Ativacao desconhecida: {name}")


class BrandSimilarityMLP(nn.Module):
    """MLP padrao -> [Linear, BN?, Act, Dropout] x N -> Linear(1)."""

    def __init__(self, config: MLPConfig) -> None:
        super().__init__()
        self.config = config
        layers: list[nn.Module] = []
        prev = config.input_dim
        for h in config.hidden_dims:
            layers.append(nn.Linear(prev, h))
            if config.use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(_activation(config.activation))
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return torch.sigmoid(self.forward(x))


__all__ = ["MLPConfig", "BrandSimilarityMLP"]
