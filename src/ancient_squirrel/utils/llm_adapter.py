from typing import List, Dict, Any, Optional, Union
import os
import json
import logging
import numpy as np
from functools import lru_cache

logger = logging.getLogger(__name__)

class LLMAdapter:
    """Adapter for LLM services with fallback options"""
    
    def __init__(self, provider: str = "openai", api_key: Optional[str] = None, 
                 model: Optional[str] = None, fallback_to_local: bool = True):
        """
        Initialize LLM adapter
        
        Args:
            provider: LLM provider (openai, anthropic, etc.)
            api_key: API key for the provider
            model: Model name to use
            fallback_to_local: Whether to fall back to local models
        """
        self.provider = provider
        self.api_key = api_key or os.getenv(f"{provider.upper()}_API_KEY")
        self.model = model or self._get_default_model()
        self.fallback_to_local = fallback_to_local
        
        self.client = self._initialize_client()
        self.local_model = None
        
        if self.client is None and fallback_to_local:
            self._initialize_local_model()
    
    def _get_default_model(self) -> str:
        """Get default model based on provider"""
        if self.provider == "openai":
            return "gpt-4.1-mini"
        elif self.provider == "anthropic":
            return "claude-3-haiku-20240307"
        return "gpt2"  # Default local model
    
    def _initialize_client(self) -> Any:
        """Initialize client based on provider"""
        if self.provider == "openai":
            if not self.api_key:
                logger.warning("No OpenAI API key provided")
                return None
            
            try:
                from openai import OpenAI
                return OpenAI(api_key=self.api_key)
            except ImportError:
                logger.error("OpenAI package not installed")
                return None
            
        elif self.provider == "anthropic":
            if not self.api_key:
                logger.warning("No Anthropic API key provided")
                return None
                
            try:
                import anthropic
                return anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                logger.error("Anthropic package not installed")
                return None
        
        return None
    
    def _initialize_local_model(self) -> None:
        """Initialize local model for fallback"""
        try:
            logger.info("Initializing local model for fallback")
            # Try loading sentence transformers for embeddings
            from sentence_transformers import SentenceTransformer
            self.local_model = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            logger.warning("sentence-transformers not installed, falling back to spaCy")
            try:
                import spacy
                self.local_model = spacy.load("en_core_web_md")
            except:
                logger.error("Neither sentence-transformers nor spaCy available")
                self.local_model = None
    
    def generate_completion(self, prompt: str, max_tokens: int = 1000, 
                           temperature: float = 0.5) -> Dict[str, Any]:
        """
        Generate completion using the configured LLM
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Dictionary with completion result
        """
        if not self.client:
            logger.warning(f"No {self.provider} client available")
            return {"error": f"No {self.provider} client available", "content": ""}
        
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return {"content": response.choices[0].message.content}
            
            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                return {"content": response.content[0].text}
            
        except Exception as e:
            logger.error(f"Error generating completion: {str(e)}")
            return {"error": str(e), "content": ""}
        
        return {"error": "Unsupported provider", "content": ""}
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """
        Generate embeddings using the configured LLM
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            Array of embeddings
        """
        if not texts:
            return np.array([])
        
        # Use API client if available
        if self.client and self.provider == "openai":
            return self._generate_openai_embeddings(texts, batch_size)
        
        # Fall back to local model if available
        if self.local_model:
            return self._generate_local_embeddings(texts, batch_size)
        
        # Last resort: return zeros
        logger.error("No embedding method available")
        return np.zeros((len(texts), 384))  # Typical embedding size
    
    def _generate_openai_embeddings(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """Generate embeddings using OpenAI API"""
        import numpy as np
        from tqdm import tqdm
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch = [text if isinstance(text, str) else "" for text in batch]
            
            # Filter out empty strings
            batch_with_indices = [(j, text) for j, text in enumerate(batch) if text.strip()]
            valid_indices = [j for j, _ in batch_with_indices]
            valid_texts = [text for _, text in batch_with_indices]
            
            # Initialize batch embeddings with zeros
            batch_embeddings = [np.zeros(1536) for _ in range(len(batch))]
            
            # If no valid texts, continue
            if not valid_texts:
                all_embeddings.extend(batch_embeddings)
                continue
            
            # Try to get embeddings
            try:
                response = self.client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=valid_texts
                )
                
                # Update batch embeddings with valid embeddings
                for idx, embedding_data in zip(valid_indices, response.data):
                    batch_embeddings[idx] = np.array(embedding_data.embedding)
                
                all_embeddings.extend(batch_embeddings)
                
            except Exception as e:
                logger.error(f"Error generating OpenAI embeddings: {str(e)}")
                all_embeddings.extend(batch_embeddings)
        
        return np.array(all_embeddings)
    
    def _generate_local_embeddings(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Generate embeddings using local model"""
        import numpy as np
        from tqdm import tqdm
        
        # Use sentence transformers if available
        if hasattr(self.local_model, 'encode'):
            embeddings = []
            
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch = [text if isinstance(text, str) else "" for text in batch]
                
                batch_embeddings = self.local_model.encode(batch)
                embeddings.extend(batch_embeddings)
            
            return np.array(embeddings)
        
        # Use spaCy if available
        elif hasattr(self.local_model, 'pipe'):
            embeddings = []
            
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                batch_embeddings = []
                for text in batch:
                    if not isinstance(text, str) or not text.strip():
                        # Empty embedding for empty text
                        batch_embeddings.append(np.zeros(self.local_model.vocab.vectors.shape[1]))
                        continue
                    
                    doc = self.local_model(text)
                    if doc.vector.any():  # Check if vector is non-zero
                        batch_embeddings.append(doc.vector)
                    else:
                        # Fallback to average of token vectors if doc vector is zero
                        token_vecs = [token.vector for token in doc if token.has_vector]
                        if token_vecs:
                            avg_vec = np.mean(token_vecs, axis=0)
                            batch_embeddings.append(avg_vec)
                        else:
                            batch_embeddings.append(np.zeros(self.local_model.vocab.vectors.shape[1]))
                
                embeddings.extend(batch_embeddings)
            
            return np.array(embeddings)
        
        # No known method
        logger.error("Local model doesn't support embeddings")
        return np.zeros((len(texts), 384))