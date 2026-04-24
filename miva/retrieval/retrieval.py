"""
Retrieval module for MIVA Studio.

Abstracts vector store operations (Qdrant, FAISS, Pinecone).
Handles ANN search, reranking, and quality gating.
"""

import logging
from typing import List, Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AnchorEmbedding:
    """Retrieved anchor embedding with metadata."""
    id: str
    embedding: np.ndarray
    subject_id: str
    quality_score: float
    pose: Optional[str] = None
    metadata: Optional[dict] = None


class VectorStore(ABC):
    """Abstract base class for vector store implementations."""
    
    @abstractmethod
    def query(self, subject_id: str, top_k: int = 5) -> List[AnchorEmbedding]:
        """Query for top-k similar embeddings."""
        pass
    
    @abstractmethod
    def add(self, subject_id: str, embedding: np.ndarray, metadata: dict = None) -> str:
        """Add embedding to store."""
        pass
    
    @abstractmethod
    def delete(self, subject_id: str):
        """Delete all embeddings for subject."""
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """Verify store is accessible."""
        pass


class FAISSVectorStore(VectorStore):
    """FAISS-based local vector store."""
    
    def __init__(self, path: str = "./vector_store", embedding_dim: int = 512):
        """Initialize FAISS vector store."""
        self.path = path
        self.embedding_dim = embedding_dim
        self.logger = logging.getLogger(__name__)
        
        try:
            import faiss
            self.faiss = faiss
        except ImportError:
            raise ImportError("faiss-cpu or faiss-gpu required. Install with: pip install faiss-cpu")
        
        # Initialize index
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.id_map = {}  # Maps index position to (subject_id, embedding_id)
        self.embeddings_metadata = {}
    
    def query(self, subject_id: str, top_k: int = 5) -> List[AnchorEmbedding]:
        """Query for top-k similar embeddings for subject."""
        try:
            # Get reference embedding for subject
            reference_embedding = self._get_reference_embedding(subject_id)
            if reference_embedding is None:
                self.logger.warning(f"No reference embedding found for {subject_id}")
                return []
            
            # Query index
            distances, indices = self.index.search(reference_embedding.reshape(1, -1), top_k)
            
            # Convert to AnchorEmbedding objects
            results = []
            for idx in indices[0]:
                if idx in self.id_map:
                    subject, emb_id = self.id_map[idx]
                    metadata = self.embeddings_metadata.get(emb_id, {})
                    
                    results.append(AnchorEmbedding(
                        id=emb_id,
                        embedding=self.index.reconstruct(int(idx)),
                        subject_id=subject,
                        quality_score=metadata.get('quality_score', 0.9),
                        pose=metadata.get('pose'),
                        metadata=metadata
                    ))
            
            return results
        except Exception as e:
            self.logger.error(f"Query failed: {e}")
            return []
    
    def add(self, subject_id: str, embedding: np.ndarray, metadata: dict = None) -> str:
        """Add embedding to store."""
        if metadata is None:
            metadata = {}
        
        embedding_id = f"{subject_id}_{len(self.id_map)}"
        
        # Add to FAISS index
        self.index.add(embedding.reshape(1, -1).astype(np.float32))
        
        # Track mapping
        idx = self.index.ntotal - 1
        self.id_map[idx] = (subject_id, embedding_id)
        self.embeddings_metadata[embedding_id] = metadata
        
        return embedding_id
    
    def delete(self, subject_id: str):
        """Delete all embeddings for subject (not supported in FAISS)."""
        self.logger.warning(f"Delete not supported for FAISS. Rebuild index to remove {subject_id}")
    
    def health_check(self) -> bool:
        """Check store health."""
        try:
            return self.index is not None and self.index.ntotal >= 0
        except Exception:
            return False
    
    def _get_reference_embedding(self, subject_id: str) -> Optional[np.ndarray]:
        """Get reference embedding for subject (first embedding)."""
        for idx, (subj, emb_id) in self.id_map.items():
            if subj == subject_id:
                return self.index.reconstruct(int(idx))
        return None


class QdrantVectorStore(VectorStore):
    """Qdrant-based vector store (local or cloud)."""
    
    def __init__(self, path: str = "./vector_store", embedding_dim: int = 512):
        """Initialize Qdrant vector store."""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
            
            self.QdrantClient = QdrantClient
            self.Distance = Distance
            self.VectorParams = VectorParams
        except ImportError:
            raise ImportError("qdrant-client required. Install with: pip install qdrant-client")
        
        self.path = path
        self.embedding_dim = embedding_dim
        self.collection_name = "miva_identities"
        self.logger = logging.getLogger(__name__)
        
        # Initialize client
        self.client = self.QdrantClient(path=path)
        
        # Ensure collection exists
        self._ensure_collection_exists()
    
    def query(self, subject_id: str, top_k: int = 5) -> List[AnchorEmbedding]:
        """Query for top-k similar embeddings."""
        try:
            # Get reference embedding
            reference = self.client.get(
                collection_name=self.collection_name,
                ids=[subject_id],
            )
            
            if not reference:
                return []
            
            ref_embedding = reference[0].vector
            
            # Search
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=ref_embedding,
                query_filter={
                    "must": [
                        {"key": "subject_id", "match": {"value": subject_id}}
                    ]
                },
                limit=top_k
            )
            
            # Convert results
            anchors = []
            for result in results:
                anchors.append(AnchorEmbedding(
                    id=result.id,
                    embedding=np.array(result.vector),
                    subject_id=result.payload.get("subject_id"),
                    quality_score=result.score,
                    metadata=result.payload
                ))
            
            return anchors
        except Exception as e:
            self.logger.error(f"Query failed: {e}")
            return []
    
    def add(self, subject_id: str, embedding: np.ndarray, metadata: dict = None) -> str:
        """Add embedding to Qdrant."""
        # Not implemented — placeholder
        self.logger.warning("Qdrant add not fully implemented")
        return f"{subject_id}_new"
    
    def delete(self, subject_id: str):
        """Delete embeddings for subject."""
        # Not implemented — placeholder
        self.logger.warning("Qdrant delete not fully implemented")
    
    def health_check(self) -> bool:
        """Check Qdrant health."""
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False
    
    def _ensure_collection_exists(self):
        """Ensure collection exists."""
        try:
            self.client.get_collection(self.collection_name)
        except:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=self.VectorParams(
                    size=self.embedding_dim,
                    distance=self.Distance.COSINE
                )
            )


def get_vector_store(store_type: str, path: str, embedding_dim: int = 512) -> VectorStore:
    """Factory function to get vector store."""
    if store_type == "faiss":
        return FAISSVectorStore(path, embedding_dim)
    elif store_type == "qdrant":
        return QdrantVectorStore(path, embedding_dim)
    else:
        raise ValueError(f"Unknown vector store type: {store_type}")


def retrieve_and_rerank(
    vector_store: VectorStore,
    subject_id: str,
    top_k: int = 5,
    diversity_threshold: float = 0.95
) -> List[AnchorEmbedding]:
    """
    Retrieve and rerank anchors by pose diversity.
    
    Stage 1: ANN search
    Stage 2: Rerank by max-marginal relevance (pose diversity)
    """
    # Stage 1: ANN search
    candidates = vector_store.query(subject_id, top_k=top_k * 3)
    
    if not candidates:
        return []
    
    # Stage 2: Rerank by pose diversity
    selected = []
    for candidate in sorted(candidates, key=lambda c: c.quality_score, reverse=True):
        if not selected:
            selected.append(candidate)
        else:
            # Compute max similarity to already-selected anchors
            max_sim = max(
                float(np.dot(candidate.embedding, s.embedding) / 
                     (np.linalg.norm(candidate.embedding) * np.linalg.norm(s.embedding) + 1e-8))
                for s in selected
            )
            
            # Only add if diverse enough
            if max_sim < diversity_threshold:
                selected.append(candidate)
        
        if len(selected) >= top_k:
            break
    
    return selected[:top_k]
