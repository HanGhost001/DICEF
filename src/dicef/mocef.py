"""Class-prototype-constrained logit correction. No additional classifier."""
from dataclasses import dataclass
import numpy as np

CLASS_NAMES = ("CN", "MCI", "AD")
EPS = 1e-8


def probabilities(value):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 3 or not len(value):
        raise ValueError("Expected a nonempty (N, 3) probability array in CN/MCI/AD order")
    if not np.isfinite(value).all() or (value < 0).any() or not np.allclose(value.sum(1), 1, atol=1e-5):
        raise ValueError("Probabilities must be finite, nonnegative and sum to one")
    return value / value.sum(1, keepdims=True)


def labels(value, n):
    value = np.asarray(value)
    if value.shape != (n,) or not np.isin(value, [0, 1, 2]).all():
        raise ValueError("Labels must be an N-vector with CN=0, MCI=1, AD=2")
    return value.astype(np.int64)


def softmax(scores):
    scores = np.asarray(scores, dtype=np.float64)
    scores = scores - scores.max(axis=1, keepdims=True)
    weights = np.exp(scores)
    return weights / weights.sum(axis=1, keepdims=True)


@dataclass
class MOCEF:
    beta: float = 1.0
    prototypes: np.ndarray | None = None

    def __post_init__(self):
        if not np.isfinite(self.beta) or self.beta < 0:
            raise ValueError("beta must be finite and nonnegative")
        if self.prototypes is not None:
            self.prototypes = self._validate_prototypes(self.prototypes)

    @staticmethod
    def _validate_prototypes(value):
        value = probabilities(value)
        if value.shape != (3, 3):
            raise ValueError("Expected one 3-component prototype per class")
        value = np.clip(value, EPS, None)
        return value / value.sum(1, keepdims=True)

    def fit(self, development_image_probabilities, development_labels):
        p = probabilities(development_image_probabilities)
        y = labels(development_labels, len(p))
        if (np.bincount(y, minlength=3) < 2).any():
            raise ValueError("Each development class needs at least two samples")
        self.prototypes = self._validate_prototypes(np.stack([p[y == c].mean(0) for c in range(3)]))
        return self

    def logit_correction(self, image_probabilities):
        if self.prototypes is None:
            raise RuntimeError("Fit development prototypes before inference")
        p = probabilities(image_probabilities)
        return self.beta * (p @ np.log(self.prototypes).T)

    def compatibility(self, image_probabilities):
        p = probabilities(image_probabilities)
        if self.prototypes is None:
            raise RuntimeError("Fit development prototypes before inference")
        entropy_term = (p * np.log(np.clip(p, EPS, None))).sum(1, keepdims=True)
        return p @ np.log(self.prototypes).T - entropy_term

    def predict_proba(self, clinical_probabilities, image_probabilities):
        pc, pi = probabilities(clinical_probabilities), probabilities(image_probabilities)
        if pc.shape != pi.shape:
            raise ValueError("Clinical and imaging predictions must have identical aligned shapes")
        # The sample-only entropy term cancels in softmax.
        return softmax(np.log(np.clip(pc, EPS, None)) + self.logit_correction(pi))

    def to_dict(self):
        if self.prototypes is None:
            raise RuntimeError("Cannot serialize an unfitted MOCEF")
        return {"classes":list(CLASS_NAMES), "beta":self.beta, "prototypes":self.prototypes.tolist()}

    @classmethod
    def from_dict(cls, state):
        if state.get("classes") != list(CLASS_NAMES):
            raise ValueError("Unexpected class order")
        return cls(beta=state["beta"], prototypes=np.asarray(state["prototypes"]))
