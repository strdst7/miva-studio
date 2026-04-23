"""
miva/retrieval/vector_store.py
Vector store client with two-stage retrieval: ANN search + MMR reranking.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from miva.config import RetrievalConfig, VectorStoreConfig

logger = logging.getLogger(__name__)


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class AnchorEmbedding:
    """A single stored identity anchor embedding."""
    id: str
    subject_id: str
    embedding: np.ndarray       # Shape: (embedding_dim,), L2-normalized
    quality_score: float        # [0, 1] — set at enrollment time
    pose_tag: Optional[str] = None   # frontal | profile | three_quarter
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Output of a successful retrieval operation."""
    subject_id: str
    anchors: list[AnchorEmbedding]
    estimated_recall: float     # Rough estimate based on anchor count and quality
    latency_ms: float

    def is_empty(self) -> bool:
        return len(self.anchors) == 0


@dataclass
class ColdStartResult:
    """Returned when a subject_id has no stored anchors."""
    subject_id: str
    reason: str = "IDENTITY_NOT_FOUND"

    def is_empty(self) -> bool:
        return True


# ── Cosine similarity ─────────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalized vectors."""
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


# ── Vector store client ───────────────────────────────────────────────────────

class VectorStore:
    """
    Thin client over an ANN index with two-stage retrieval:
      Stage 1: Approximate nearest neighbor search (ANN)
      Stage 2: Max-marginal relevance reranking for pose diversity

    The vector store is intentionally provider-agnostic. The _query_backend()
    method should be overridden or replaced for each backend (Qdrant, FAISS, etc.)
    """

    def __init__(
        self,
        vs_config: VectorStoreConfig,
        retrieval_config: RetrievalConfig,
    ):
        self.vs_config = vs_config
        self.ret_config = retrieval_config
        self._backend = self._init_backend()

    def _init_backend(self):
        """Initialise the configured vector store backend."""
        provider = self.vs_config.provider.lower()
        if provider == "qdrant":
            return self._init_qdrant()
        elif provider == "faiss":
            return self._init_faiss()
        else:
            raise ValueError(f"Unsupported vector store provider: {provider}")

    def _init_qdrant(self):
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(
                host=self.vs_config.host,
                port=self.vs_config.port,
            )
            logger.info(
                "Connected to Qdrant at %s:%d", self.vs_config.host, self.vs_config.port
            )
            return client
        except ImportError:
            raise RuntimeError("qdrant-client not installed. Run: pip install qdrant-client")

    def _init_faiss(self):
        try:
            import faiss  # noqa: F401
            logger.info("FAISS backend initialised")
            return "faiss"  # Actual FAISS index loaded per-collection
        except ImportError:
            raise RuntimeError("faiss-cpu not installed. Run: pip install faiss-cpu")

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        subject_id: str,
        top_k: Optional[int] = None,
    ) -> RetrievalResult | ColdStartResult:
        """
        Main retrieval entry point.
        Returns RetrievalResult with up to top_k diverse anchors,
        or ColdStartResult if subject_id is unknown.
        """
        import time
        start = time.perf_counter()
        k = top_k or self.ret_config.top_k

        # Stage 1: ANN search — fetch candidate_multiplier × k candidates
        candidate_count = k * self.ret_config.candidate_multiplier
        candidates = self._query_backend(subject_id, candidate_count)

        if not candidates:
            logger.warning("No anchors found for subject_id=%s", subject_id)
            return ColdStartResult(subject_id=subject_id)

        # Stage 2: MMR reranking
        selected = self._mmr_rerank(candidates, top_k=k)

        if len(selected) < self.ret_config.min_anchor_count:
            logger.warning(
                "Only %d anchors retrieved for subject_id=%s (min recommended: %d)",
                len(selected), subject_id, self.ret_config.min_anchor_count,
            )

        latency_ms = (time.perf_counter() - start) * 1000
        estimated_recall = self._estimate_recall(selected, candidates)

        logger.debug(
            "Retrieved %d anchors for %s in %.1fms (recall_est=%.3f)",
            len(selected), subject_id, latency_ms, estimated_recall,
        )

        return RetrievalResult(
            subject_id=subject_id,
            anchors=selected,
            estimated_recall=estimated_recall,
            latency_ms=latency_ms,
        )

    def upsert(self, anchor: AnchorEmbedding) -> None:
        """Store an anchor embedding. Called at enrollment time."""
        if self.vs_config.provider == "qdrant":
            self._qdrant_upsert(anchor)
        else:
            raise NotImplementedError(f"Upsert not implemented for {self.vs_config.provider}")

    def get_all(self, subject_id: str) -> list[AnchorEmbedding]:
        """Retrieve all stored anchors for a subject (used in evaluation)."""
        return self._query_backend(subject_id, top_k=1000)

    def get_reference_embedding(self, subject_id: str) -> Optional[np.ndarray]:
        """Return the enrollment reference embedding for ground truth computation."""
        all_anchors = self.get_all(subject_id)
        if not all_anchors:
            return None
        # Reference = highest-quality anchor
        return max(all_anchors, key=lambda a: a.quality_score).embedding

    # ── Internal: backend query ───────────────────────────────────────────────

    def _query_backend(self, subject_id: str, top_k: int) -> list[AnchorEmbedding]:
        """Query the configured backend. Override for each provider."""
        if self.vs_config.provider == "qdrant":
            return self._qdrant_query(subject_id, top_k)
        elif self.vs_config.provider == "faiss":
            return self._faiss_query(subject_id, top_k)
        return []

    def _qdrant_query(self, subject_id: str, top_k: int) -> list[AnchorEmbedding]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        results = self._backend.search(
            collection_name=self.vs_config.collection,
            query_filter=Filter(
                must=[FieldCondition(key="subject_id", match=MatchValue(value=subject_id))]
            ),
            query_vector=np.zeros(self.vs_config.embedding_dim).tolist(),  # Filter-only search
            limit=top_k,
            with_payload=True,
            with_vectors=True,
        )
        return [
            AnchorEmbedding(
                id=str(r.id),
                subject_id=subject_id,
                embedding=np.array(r.vector),
                quality_score=r.payload.get("quality_score", 0.8),
                pose_tag=r.payload.get("pose_tag"),
                metadata=r.payload,
            )
            for r in results
        ]

    def _qdrant_upsert(self, anchor: AnchorEmbedding) -> None:
        from qdrant_client.models import PointStruct
        self._backend.upsert(
            collection_name=self.vs_config.collection,
            points=[PointStruct(
                id=anchor.id,
                vector=anchor.embedding.tolist(),
                payload={
                    "subject_id": anchor.subject_id,
                    "quality_score": anchor.quality_score,
                    "pose_tag": anchor.pose_tag,
                    **anchor.metadata,
                },
            )],
        )

    def _faiss_query(self, subject_id: str, top_k: int) -> list[AnchorEmbedding]:
        # FAISS implementation placeholder
        # In production: load per-subject index shard, run search
        raise NotImplementedError("FAISS backend query not yet implemented")

    # ── Internal: MMR reranking ───────────────────────────────────────────────

    def _mmr_rerank(
        self,
        candidates: list[AnchorEmbedding],
        top_k: int,
    ) -> list[AnchorEmbedding]:
        """
        Max-Marginal Relevance reranking for pose diversity.
        Selects anchors that are high-quality AND diverse relative to each other.

        Rationale: 20 embeddings of the same frontal pose add no more information
        than 1. We want the top-k to span the identity's appearance space.

        Reference: Carbonell & Goldstein (1998), MMR for diversity-based reranking.
        """
        if not candidates:
            return []

        # Sort by quality score descending — quality is the primary criterion
        sorted_candidates = sorted(candidates, key=lambda a: a.quality_score, reverse=True)
        selected: list[AnchorEmbedding] = []

        for candidate in sorted_candidates:
            if not selected:
                selected.append(candidate)
                continue

            # Check similarity to all already-selected anchors
            max_sim_to_selected = max(
                cosine_similarity(candidate.embedding, s.embedding)
                for s in selected
            )

            # Only include if sufficiently different from selected set
            if max_sim_to_selected < self.ret_config.diversity_threshold:
                selected.append(candidate)

            if len(selected) == top_k:
                break

        return selected

    def _estimate_recall(
        self,
        selected: list[AnchorEmbedding],
        candidates: list[AnchorEmbedding],
    ) -> float:
        """
        Rough recall estimate: ratio of high-quality anchors selected
        vs. high-quality anchors in the candidate pool.
        """
        HIGH_QUALITY_THRESHOLD = 0.80
        high_q_candidates = [a for a in candidates if a.quality_score >= HIGH_QUALITY_THRESHOLD]
        high_q_selected = [a for a in selected if a.quality_score >= HIGH_QUALITY_THRESHOLD]

        if not high_q_candidates:
            return 1.0  # No high-quality anchors to miss
        return len(high_q_selected) / len(high_q_candidates)
