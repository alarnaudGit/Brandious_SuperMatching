"""Calibracao de probabilidades via Platt scaling.

Treina uma regressao logistica de 1 dimensao (logit_input -> P(y=1)) sobre
a particao de validacao. Util quando o modelo eh acurado em ranking
(boa ROC-AUC) mas as probabilidades absolutas estao mal calibradas (escolha
de threshold viesado).

Uso:
    cal = PlattCalibrator().fit(val_scores, y_val)
    test_calibrated = cal.transform(test_scores)
    cal.save("calibrator.pkl")
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


def _to_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


@dataclass
class PlattCalibrator:
    """Platt scaling sobre logits do modelo (1-D logistic regression).

    Apos `fit`, `transform` e `transform_logits` aplicam a transformacao.
    """
    fit_intercept: bool = True
    a_: float = field(default=1.0, init=False)
    b_: float = field(default=0.0, init=False)
    fitted: bool = field(default=False, init=False)

    def fit(self, scores: np.ndarray, y_true: np.ndarray) -> "PlattCalibrator":
        scores = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
        y = np.asarray(y_true, dtype=np.int32).ravel()
        if len(np.unique(y)) < 2:
            logger.warning("PlattCalibrator.fit: 1 unica classe; calibrador identidade.")
            self.a_ = 1.0
            self.b_ = 0.0
            self.fitted = True
            return self

        logits = _to_logit(scores).reshape(-1, 1)
        lr = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=200,
            fit_intercept=self.fit_intercept,
        )
        lr.fit(logits, y)
        self.a_ = float(lr.coef_[0, 0])
        self.b_ = float(lr.intercept_[0]) if self.fit_intercept else 0.0
        self.fitted = True
        logger.info(
            "PlattCalibrator fitado: a=%.4f b=%.4f (n=%d, p_pos=%.3f)",
            self.a_, self.b_, len(y), float(y.mean()),
        )
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Recebe scores (sigmoid-saidas) em [0,1], devolve P calibrada em [0,1]."""
        if not self.fitted:
            return np.asarray(scores, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float64)
        logits = _to_logit(scores)
        z = self.a_ * logits + self.b_
        return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)

    def transform_logits(self, logits: np.ndarray) -> np.ndarray:
        """Variante quando o caller ja tem logits (evita ida-volta sigmoid->logit)."""
        if not self.fitted:
            return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
        z = self.a_ * np.asarray(logits, dtype=np.float64) + self.b_
        return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)

    def to_dict(self) -> dict:
        return {
            "fit_intercept": self.fit_intercept,
            "a_": self.a_,
            "b_": self.b_,
            "fitted": self.fitted,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PlattCalibrator":
        obj = cls(fit_intercept=bool(d.get("fit_intercept", True)))
        obj.a_ = float(d.get("a_", 1.0))
        obj.b_ = float(d.get("b_", 0.0))
        obj.fitted = bool(d.get("fitted", False))
        return obj

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".json":
            path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        else:
            joblib.dump(self.to_dict(), path)
        logger.info("PlattCalibrator salvo em %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "PlattCalibrator":
        path = Path(path)
        if path.suffix.lower() == ".json":
            d = json.loads(path.read_text(encoding="utf-8"))
        else:
            d = joblib.load(path)
        return cls.from_dict(d)


__all__ = ["PlattCalibrator"]
