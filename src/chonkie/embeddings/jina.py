"""Module for Jina AI embeddings integration."""

import os
import requests
import warnings
from typing import List, Optional
import numpy as np

from .base import BaseEmbeddings

class JinaEmbeddings(BaseEmbeddings):
    """Jina embeddings implementation using their API."""
    def __init__(
            self,
            model: str = "jina-embeddings-v3",
            task: str = "text-matching",
            late_chunking: bool = True,
            embedding_type: str = "float",
            dimensions: int = 1024,
            api_key: Optional[str] = None,
            batch_size = 128
    ):
        """Initialize Jina embeddings.
        Args:
            model: Name of the Jina embedding model to use
            task: Task for the Jina model
            late_chunking: Whether to use late chunking
            embedding_type: Type of the embedding
            dimensions: Dimensions of the embedding
            api_key: Jina API key (if not provided, looks for JINA_API_KEY env var)
            batch_size: Maximum number of texts to embed in one API call
        """
        super().__init__()
        self.model = model
        self.task = task
        self.late_chunking = late_chunking
        self._dimension = dimensions
        self.embedding_type = embedding_type
        self._batch_size = batch_size
        api_key = api_key or os.getenv("JINA_API_KEY")
        self.api_key = api_key
        self.url = 'https://api.jina.ai/v1/embeddings'
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
    
    def embed(self, text: List[str]) -> np.ndarray:
        """Embed a list of texts using the Jina embeddings API."""
        self.data = {
            "model": self.model,
            "task": self.task,
            "late_chunking": self.late_chunking,
            "embedding_type": self.embedding_type,
            "dimensions": self._dimension,
            "input": text
        }

        response = requests.post(self.url, json=self.data, headers=self.headers)
        response.raise_for_status()
        vector = response.json()
        return np.array(vector['data'][0]['embedding'], dtype = np.float32)
    
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed multiple texts using the Jina embeddings API."""
        if not texts:
            return []
        all_embeddings = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i:i + self._batch_size]
            try:
                response = requests.post(self.url, json=self.data, headers=self.headers)
                response.raise_for_status()
                sorted_embeddings = sorted(response.data, key=lambda x: x['index'])
                all_embeddings.extend(sorted_embeddings)
            except requests.exceptions.HTTPError as e:
                if len(batch)>1:
                    warnings.warn(f"Embedding failed : {str(e)}. Trying one by one")
        return all_embeddings

    def similarity(self, u, v):
        """Compute cosine similarity of two embeddings."""
        return np.divide(
            np.dot(u, v), np.linalg.norm(u) * np.linalg.norm(v), dtype=np.float32
        )  
    def count_tokens(self, text: str, tokenizer = 'cl100k_base') -> int:
        """Count tokens in text using the Jina segmenter."""
        api_key = self.api_key or os.getenv("JINA_API_KEY")
        if not api_key:
            raise ValueError("API key is required for Jina segmenter token count.")
        url = 'https://api.jina.ai/v1/segment'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }

        data = {
            "content": text,
            "tokenizer": tokenizer
        }
        response = requests.post(url, headers=headers, json=data)   
        response.raise_for_status()
        token_count = response.json()['num_tokens']
        return token_count

    def count_tokens_batch(self, texts):
        """Count tokens in multiple texts."""
        token_counts = [self.count_tokens(text) for text in texts]
        return token_counts
    
    def similarity(self, u: np.ndarray, v: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        return np.divide(
            np.dot(u, v), np.linalg.norm(u) * np.linalg.norm(v), dtype=float
        )
    @property
    def dimension(self) -> int:
        """Return the dimensions of the embeddings."""
        return self._dimension
    def get_tokenizer_or_token_counter(self):
        """Get the tokenizer or token counter for the embeddings."""
        return self.count_tokens
    
    def __repr__(self):
        return f"JinaEmbeddings(model={self.model}"