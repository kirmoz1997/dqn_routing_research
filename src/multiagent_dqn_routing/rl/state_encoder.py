from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from multiagent_dqn_routing.agents import N_AGENTS


class TfidfStateEncoder:
    """TF-IDF text encoder + state concatenation helper."""

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 2),
            min_df=2,
        )
        self._is_fitted = False

    @property
    def text_dim(self) -> int:
        if not self._is_fitted:
            raise RuntimeError("Encoder is not fitted")
        return int(len(self.vectorizer.get_feature_names_out()))

    @property
    def state_dim(self) -> int:
        return self.text_dim + N_AGENTS + 1

    def fit(self, texts: Iterable[str]) -> None:
        self.vectorizer.fit(list(texts))
        self._is_fitted = True

    def transform_text(self, text: str) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Encoder is not fitted")
        vec = self.vectorizer.transform([text]).astype(np.float32)
        return vec.toarray().ravel().astype(np.float32, copy=False)

    def encode(
        self,
        text_vec: np.ndarray | sparse.spmatrix,
        selected_mask: np.ndarray,
        step_idx: int,
        max_steps: int,
    ) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Encoder is not fitted")
        if max_steps <= 0:
            raise ValueError("max_steps must be > 0")

        if sparse.issparse(text_vec):
            tv = text_vec.toarray().ravel()
        else:
            tv = np.asarray(text_vec)
            if tv.ndim > 1:
                tv = tv.reshape(-1)
        tv = tv.astype(np.float32, copy=False)

        mask = np.asarray(selected_mask, dtype=np.float32).reshape(-1)
        if mask.size != N_AGENTS:
            raise ValueError(f"selected_mask must have size {N_AGENTS}")

        step_frac = np.float32(step_idx / max_steps)
        state = np.concatenate(
            [tv, mask, np.asarray([step_frac], dtype=np.float32)],
            axis=0,
        )
        return state.astype(np.float32, copy=False)
