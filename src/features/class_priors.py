"""Prior empirico bayesiano para pares de classes Nice.

Substitui a representacao esparsa de classes (one-hot top-K + cls_a_top_*)
por um sinal denso e calibrado:

- `cls_pair_prior_pos` = P(label=1 | classe_a, classe_b), calculado SOMENTE
  no conjunto de treino, com smoothing Beta(alpha=2, beta=8) bayesiano para
  evitar overfit em pares raros.
- `cls_pair_chi2_strength` = forca normalizada do chi-quadrado do par
  (classe_a, classe_b) vs label, util como sinal de "esse par e
  estatisticamente associado/dissociado de positivos".

Pares nao vistos no treino caem no prior global P(y=1) com uma penalidade
suave em direcao ao prior bayesiano default.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)


def _canonical_pair(a: int, b: int) -> tuple[int, int]:
    """Pares (a,b) e (b,a) compartilham o mesmo prior - mantemos canonical."""
    a_i, b_i = int(a), int(b)
    if a_i <= b_i:
        return a_i, b_i
    return b_i, a_i


@dataclass
class ClassPairPrior:
    """Estima P(y=1 | classe_a, classe_b) com smoothing bayesiano Beta(alpha,beta).

    Apos o `fit`, `transform` e `transform_pair` retornam:
      cls_pair_prior_pos       -> probabilidade smoothed
      cls_pair_chi2_strength   -> chi^2 normalizado (clipped em [0,1])

    Notas:
    - Os pares sao canonizados (min,max) para tornar a estatistica simetrica.
    - Pares com contagem total < `min_count` recebem o prior global da
      base; ainda assim ainda ganham smoothing pelo Beta.
    - O chi^2 e calculado sobre a tabela 2x2 contagem(par, y).
    """

    alpha: float = 2.0   # prior positivo (~1 evento positivo a priori)
    beta: float = 8.0    # prior negativo (~4 eventos negativos a priori)
    min_count: int = 1   # pares com >= min_count contagens no treino

    pos_prior: float = field(default=0.5, init=False)
    pair_pos_count: dict[tuple[int, int], int] = field(default_factory=dict, init=False)
    pair_total_count: dict[tuple[int, int], int] = field(default_factory=dict, init=False)
    chi2_lookup: dict[tuple[int, int], float] = field(default_factory=dict, init=False)
    n_total: int = field(default=0, init=False)
    n_pos_total: int = field(default=0, init=False)
    fitted: bool = field(default=False, init=False)

    def fit(
        self,
        classes_a: Iterable[int],
        classes_b: Iterable[int],
        labels: Iterable[int],
    ) -> "ClassPairPrior":
        ca = [int(x) for x in classes_a]
        cb = [int(x) for x in classes_b]
        y = [int(v) for v in labels]
        n = len(y)
        if not (len(ca) == len(cb) == n):
            raise ValueError("ClassPairPrior.fit: tamanhos divergentes.")

        self.n_total = n
        self.n_pos_total = int(sum(y))
        self.pos_prior = self.n_pos_total / max(1, n)

        pos_count: dict[tuple[int, int], int] = defaultdict(int)
        tot_count: dict[tuple[int, int], int] = defaultdict(int)
        for a, b, lbl in zip(ca, cb, y):
            pair = _canonical_pair(a, b)
            tot_count[pair] += 1
            if lbl == 1:
                pos_count[pair] += 1
        self.pair_pos_count = dict(pos_count)
        self.pair_total_count = dict(tot_count)

        max_chi2 = 0.0
        chi2_raw: dict[tuple[int, int], float] = {}
        for pair, tot in tot_count.items():
            if tot < max(1, self.min_count):
                continue
            obs_pos = pos_count.get(pair, 0)
            obs_neg = tot - obs_pos
            n_pos = self.n_pos_total
            n_neg = self.n_total - self.n_pos_total
            other_pos = n_pos - obs_pos
            other_neg = n_neg - obs_neg
            table = np.array(
                [[obs_pos, obs_neg], [max(0, other_pos), max(0, other_neg)]],
                dtype=np.float64,
            )
            row_tot = table.sum(axis=1, keepdims=True)
            col_tot = table.sum(axis=0, keepdims=True)
            grand = float(table.sum())
            if grand <= 0:
                continue
            expected = row_tot @ col_tot / grand
            with np.errstate(divide="ignore", invalid="ignore"):
                chi2 = float(np.where(expected > 0, (table - expected) ** 2 / expected, 0.0).sum())
            chi2_raw[pair] = chi2
            if chi2 > max_chi2:
                max_chi2 = chi2

        if max_chi2 > 0:
            self.chi2_lookup = {p: min(1.0, v / max_chi2) for p, v in chi2_raw.items()}
        else:
            self.chi2_lookup = {p: 0.0 for p in chi2_raw}

        self.fitted = True
        logger.info(
            "ClassPairPrior fitado: %d pares unicos, prior global P(y=1)=%.4f",
            len(self.pair_total_count), self.pos_prior,
        )
        return self

    def transform_pair(self, classe_a: int, classe_b: int) -> tuple[float, float]:
        """Retorna (cls_pair_prior_pos, cls_pair_chi2_strength) para 1 par."""
        if not self.fitted:
            return self.pos_prior, 0.0
        pair = _canonical_pair(classe_a, classe_b)
        pos_obs = self.pair_pos_count.get(pair, 0)
        tot_obs = self.pair_total_count.get(pair, 0)
        smoothed = (pos_obs + self.alpha) / (tot_obs + self.alpha + self.beta)
        chi2 = self.chi2_lookup.get(pair, 0.0)
        return float(smoothed), float(chi2)

    def transform(
        self,
        classes_a: Iterable[int],
        classes_b: Iterable[int],
    ) -> np.ndarray:
        """Vetorizado: retorna ndarray shape (N, 2) com [prior, chi2]."""
        ca = list(classes_a)
        cb = list(classes_b)
        n = len(ca)
        out = np.zeros((n, 2), dtype=np.float32)
        for i in range(n):
            p, c = self.transform_pair(int(ca[i]), int(cb[i]))
            out[i, 0] = p
            out[i, 1] = c
        return out

    def feature_names(self) -> list[str]:
        return class_pair_prior_feature_names()

    def to_state(self) -> dict:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "min_count": self.min_count,
            "pos_prior": self.pos_prior,
            "pair_pos_count": {f"{a}|{b}": v for (a, b), v in self.pair_pos_count.items()},
            "pair_total_count": {f"{a}|{b}": v for (a, b), v in self.pair_total_count.items()},
            "chi2_lookup": {f"{a}|{b}": v for (a, b), v in self.chi2_lookup.items()},
            "n_total": self.n_total,
            "n_pos_total": self.n_pos_total,
            "fitted": self.fitted,
        }

    @classmethod
    def from_state(cls, state: dict) -> "ClassPairPrior":
        obj = cls(
            alpha=float(state.get("alpha", 2.0)),
            beta=float(state.get("beta", 8.0)),
            min_count=int(state.get("min_count", 1)),
        )
        obj.pos_prior = float(state.get("pos_prior", 0.5))
        obj.pair_pos_count = {
            tuple(int(x) for x in k.split("|")): int(v)
            for k, v in state.get("pair_pos_count", {}).items()
        }
        obj.pair_total_count = {
            tuple(int(x) for x in k.split("|")): int(v)
            for k, v in state.get("pair_total_count", {}).items()
        }
        obj.chi2_lookup = {
            tuple(int(x) for x in k.split("|")): float(v)
            for k, v in state.get("chi2_lookup", {}).items()
        }
        obj.n_total = int(state.get("n_total", 0))
        obj.n_pos_total = int(state.get("n_pos_total", 0))
        obj.fitted = bool(state.get("fitted", False))
        return obj


def class_pair_prior_feature_names() -> list[str]:
    return ["cls_pair_prior_pos", "cls_pair_chi2_strength"]


__all__ = [
    "ClassPairPrior",
    "class_pair_prior_feature_names",
]
