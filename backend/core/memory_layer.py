"""
Memory Layer Module
Sub-millisecond similarity search using FAISS GPU index
"""

import faiss
import numpy as np
from typing import List, Tuple, Dict
import pickle
from pathlib import Path


class MemoryLayer:
    def __init__(self, dimension: int = 1408, use_gpu: bool = True):
        """
        Initialize FAISS index for vector similarity search
        
        Args:
            dimension: Embedding dimension (1408 for Vertex AI)
            use_gpu: Use GPU acceleration if available
        """
        self.dimension = dimension
        self.use_gpu = use_gpu
        
        # Inner Product = Cosine Similarity for normalized vectors
        self.index = faiss.IndexFlatIP(dimension)
        
        # Move to GPU if available
        if use_gpu and faiss.get_num_gpus() > 0:
            print(f"Initializing FAISS on GPU ({faiss.get_num_gpus()} GPUs available)")
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            self.on_gpu = True
        else:
            print("Initializing FAISS on CPU")
            self.on_gpu = False
        
        self.clip_metadata: List[Dict] = []
    
    def add_broll_library(self, embeddings: np.ndarray, metadata: List[Dict]):
        """
        Add B-Roll vectors to index
        
        Args:
            embeddings: Array of shape (N, dimension) with L2-normalized vectors
            metadata: List of N dictionaries with clip information
        """
        assert embeddings.shape[1] == self.dimension, \
            f"Expected dimension {self.dimension}, got {embeddings.shape[1]}"
        
        # Verify normalization (critical for cosine similarity via inner product)
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-5):
            print("Warning: Embeddings not properly normalized. Normalizing now...")
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        self.index.add(embeddings)
        self.clip_metadata.extend(metadata)
        
        print(f"Added {len(embeddings)} vectors to index (total: {self.index.ntotal})")
    
    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find top-k most similar clips
        
        Args:
            query_vector: Query embedding of shape (dimension,) or (1, dimension)
            k: Number of neighbors to return
        
        Returns:
            distances: Cosine similarities (higher = better) of shape (1, k)
            indices: Clip IDs in metadata array of shape (1, k)
        """
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # Normalize query
        query_vector = query_vector / np.linalg.norm(query_vector, axis=1, keepdims=True)
        
        # Ensure we don't request more neighbors than we have
        k = min(k, self.index.ntotal)
        
        distances, indices = self.index.search(query_vector.astype('float32'), k)
        return distances, indices
    
    def save(self, path: str):
        """
        Persist index and metadata to disk
        
        Args:
            path: Base path (will create .index and .meta files)
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        if self.on_gpu:
            print("Moving index to CPU for saving...")
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, f"{path}.index")
        else:
            faiss.write_index(self.index, f"{path}.index")
        
        # Save metadata
        with open(f"{path}.meta", 'wb') as f:
            pickle.dump(self.clip_metadata, f)
        
        print(f"Saved index to {path}.index and {path}.meta")
    
    def load(self, path: str):
        """
        Load persisted index and metadata
        
        Args:
            path: Base path (will load .index and .meta files)
        """
        # Load index
        cpu_index = faiss.read_index(f"{path}.index")
        
        if self.use_gpu and faiss.get_num_gpus() > 0:
            print("Moving loaded index to GPU...")
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            self.on_gpu = True
        else:
            self.index = cpu_index
            self.on_gpu = False
        
        # Load metadata
        with open(f"{path}.meta", 'rb') as f:
            self.clip_metadata = pickle.load(f)
        
        print(f"Loaded index with {self.index.ntotal} vectors from {path}")
    
    def get_stats(self) -> Dict:
        """Get index statistics"""
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "on_gpu": self.on_gpu,
            "num_clips": len(self.clip_metadata)
        }
