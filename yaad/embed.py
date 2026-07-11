"""Optional dense retrieval over chunks via sentence-transformers.

Everything here is lazy: the core tool works FTS-only if the `dense`
extra isn't installed. At personal-chat scale (a few thousand chunks)
a brute-force cosine scan in numpy is instant, so there is no ANN
index to babysit.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from .db import connect

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _load_model(name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "dense retrieval needs the extra: pip install 'yaad[dense]'"
        ) from e

    # huggingface_hub logs an "unauthenticated requests" warning, and
    # transformers draws its own "Loading weights" tqdm bar - both are noise
    # here (a public model needs no token) and the bar fights with any
    # spinner/status display the caller is showing. transformers caches
    # whether progress bars are enabled in a module-level flag *at import
    # time*, so disabling it on huggingface_hub alone is too late once
    # transformers has already been imported - disable_progress_bar() below
    # resets that cached flag directly instead.
    import logging

    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    try:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except ImportError:  # pragma: no cover
        pass
    try:
        from transformers.utils.logging import disable_progress_bar

        disable_progress_bar()
    except ImportError:  # pragma: no cover
        pass

    return SentenceTransformer(name)


def build_dense_index(
    db_path: str | Path, model_name: str = DEFAULT_MODEL, batch_size: int = 64
) -> int:
    """Embed every chunk and store vectors in the db. Returns chunk count."""
    import numpy as np

    model = _load_model(model_name)
    con = connect(db_path)
    try:
        rows = con.execute("SELECT id, text FROM chunks ORDER BY id").fetchall()
        if not rows:
            return 0
        vecs = model.encode(
            [r["text"] for r in rows],
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        con.executemany(
            "INSERT OR REPLACE INTO chunk_vectors (chunk_id, dim, vec) VALUES (?, ?, ?)",
            [
                (r["id"], len(v), np.asarray(v, dtype=np.float32).tobytes())
                for r, v in zip(rows, vecs)
            ],
        )
        con.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('embedding_model', ?)",
            (model_name,),
        )
        con.commit()
        return len(rows)
    finally:
        con.close()


class DenseSearcher:
    """In-memory cosine search over stored chunk vectors."""

    def __init__(self, con: sqlite3.Connection, model_name: str | None = None):
        import numpy as np

        rows = con.execute("SELECT chunk_id, vec FROM chunk_vectors ORDER BY chunk_id").fetchall()
        if not rows:
            raise RuntimeError("no chunk vectors in db - run ingest without --no-dense")
        self._np = np
        self.ids = [r["chunk_id"] for r in rows]
        self.matrix = np.vstack(
            [np.frombuffer(r["vec"], dtype=np.float32) for r in rows]
        )
        if model_name is None:
            row = con.execute("SELECT value FROM meta WHERE key='embedding_model'").fetchone()
            model_name = row["value"] if row else DEFAULT_MODEL
        self.model = _load_model(model_name)

    def search(
        self, query: str, top_k: int = 15, allowed_ids: set[int] | None = None
    ) -> list[tuple[int, float]]:
        np = self._np
        q = self.model.encode([query], normalize_embeddings=True)[0].astype(np.float32)
        scores = self.matrix @ q  # vectors are normalized -> cosine
        order = np.argsort(-scores)
        out: list[tuple[int, float]] = []
        for i in order:
            cid = self.ids[int(i)]
            if allowed_ids is not None and cid not in allowed_ids:
                continue
            out.append((cid, float(scores[int(i)])))
            if len(out) >= top_k:
                break
        return out
