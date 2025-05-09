import numpy as np
import json
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from ..core.base_processor import BaseProcessor
from ..utils.text_utils import extract_bigrams

class ClusterInsightExtractor(BaseProcessor):
    import json
    """
    Refactored version of the original ClusterInsightExtractor that analyzes a
    DataFrame of YouTube videos and returns rich insights for clusters.
    """
    
    def __init__(self, config: Dict[str, Any], num_workers: Optional[int] = None,
                logger: Optional[logging.Logger] = None):
        """
        Initialize the cluster insight extractor
        
        Args:
            config: Configuration dictionary
            num_workers: Number of worker processes
            logger: Logger instance
        """
        super().__init__(num_workers, logger)
        self.config = config
        self._llm = self._initialize_llm() if config.get("llm_enabled", False) else None
    
    def _initialize_llm(self):
        """Initialize LLM client for insights"""
        from ..utils.llm_adapter import LLMAdapter
        
        openai_key = self.config.get("openai_api_key")
        model = self.config.get("llm_model", "gpt-4.1-mini")
        
        if not openai_key:
            self.logger.warning("OpenAI API key not provided, LLM insights disabled")
            return None
        
        return LLMAdapter(provider="openai", api_key=openai_key, model=model)
    
    def process(self, df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Process data and return results
        
        Args:
            df: Input DataFrame
            **kwargs: Additional options
            
        Returns:
            Tuple of (enhanced DataFrame, analysis results dict)
        """
        # Validate that required columns exist
        self._validate_dataframe(df)
        
        # Get top clusters to analyze
        top_n = kwargs.get("top_n", self.config.get("top_n_clusters", 10))
        
        # Extract insights
        insights = self.analyze_top_clusters(df, top_n=top_n)
        
        # The DataFrame is not modified in this processor
        return df, {"cluster_insights": insights}
    
    def analyze_top_clusters(self, df: pd.DataFrame, top_n: int = 10) -> Dict[str, Dict[str, Any]]:
        """
        Return insights for the top-N clusters by size.
        
        Args:
            df: DataFrame with cluster assignments and other data
            top_n: Number of top clusters to analyze
            
        Returns:
            Dictionary with insights for each top cluster
        """
        # Get top clusters by size
        top_clusters = (
            df["cluster"].value_counts()
            .head(top_n)
            .index.tolist()
        )
        
        insights: Dict[str, Dict[str, Any]] = {}
        for cid in tqdm(top_clusters, desc="Extracting cluster insights"):
            subset = df[df["cluster"] == cid].copy()
            
            insight = {
                "size": len(subset),
                **self._thematic_profile(subset),
                **self._channel_profile(subset),
                "sample_videos": subset["title"].head(3).tolist(),
            }
            
            # Add LLM-generated viewer intent if enabled
            if self._llm:
                try:
                    intent = self._summarize_viewer_intent(subset["title"].tolist())
                    insight["viewer_intent"] = intent
                except Exception as e:
                    self.logger.warning(f"LLM summarization failed: {e}")
            
            insights[str(cid)] = insight
        
        return insights
    
    def _thematic_profile(self, subset: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract thematic profile for a cluster subset
        
        Args:
            subset: DataFrame containing a single cluster
            
        Returns:
            Dictionary with thematic insights
        """
        titles: List[str] = subset["clean_title"].tolist()
        
        # 1) Extract top TF-IDF terms
        tfidf = TfidfVectorizer(
            stop_words="english",
            max_features=self.config.get("tfidf_max_features", 1000),
        )
        mat = tfidf.fit_transform(titles)
        term_scores = mat.mean(axis=0).A1
        top_term_idx = term_scores.argsort()[-10:][::-1]
        top_terms = tfidf.get_feature_names_out()[top_term_idx].tolist()
        
        # 2) Extract common bigrams
        bigrams = []
        all_text = " ".join(titles)
        
        # Import only what's needed for bigram extraction
        import spacy
        nlp = spacy.blank("en")  # Light-weight tokenizer
        
        # Tokenize all titles
        tokens = [
            token.text
            for doc in nlp.pipe(titles, disable=["tagger", "parser"])
            for token in doc
            if token.is_alpha and len(token.text) > 2
        ]
        
        # Use NLTK for bigram extraction
        from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
        
        finder = BigramCollocationFinder.from_words(tokens)
        finder.apply_freq_filter(self.config.get("bigram_min_freq", 3))
        
        # Get top bigrams
        bigram_top_k = self.config.get("bigram_top_k", 5)
        bigrams = [
            " ".join(bigram)
            for bigram, _ in finder.nbest(
                BigramAssocMeasures.likelihood_ratio, bigram_top_k * 2
            )
        ][:bigram_top_k]
        
        # 3) Find most central videos (closest to centroid)
        centroid_titles = []
        if (
            "embedding" in subset.columns
            and isinstance(subset["embedding"].iloc[0], (list, np.ndarray))
        ):
            # Calculate cluster centroid
            centroid = np.vstack(subset["embedding"]).mean(axis=0, keepdims=True)
            
            # Calculate similarity to centroid
            sims = cosine_similarity(
                centroid, np.vstack(subset["embedding"])
            )[0]
            
            # Get indices of most central videos
            centroid_top_k = self.config.get("centroid_top_k", 5)
            top_idx = sims.argsort()[-centroid_top_k:][::-1]
            
            # Get titles of most central videos
            centroid_titles = [subset["clean_title"].iloc[i] for i in top_idx]
        
        return {
            "centroid_terms": top_terms,
            "top_bigrams": bigrams,
            "key_ideas": centroid_titles,
        }
    
    def _channel_profile(self, subset: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract channel profile for a cluster subset
        
        Args:
            subset: DataFrame containing a single cluster
            
        Returns:
            Dictionary with channel insights
        """
        return {
            "top_channels_by_views": self._aggregate_channels(
                subset, "views", agg="sum"
            ),
            "top_channels_by_influence": self._aggregate_channels(
                subset, "influence", agg="mean"
            ),
        }
    
    @staticmethod
    def _aggregate_channels(
        subset: pd.DataFrame,
        col: str,
        agg: str = "sum",
        top_k: int = 5,
    ) -> List[List[Any]]:
        """
        Aggregate metrics by channel
        
        Args:
            subset: DataFrame containing a single cluster
            col: Column to aggregate
            agg: Aggregation method (sum, mean, etc.)
            top_k: Number of top channels to return
            
        Returns:
            List of [channel, value] pairs
        """
        # Skip if column doesn't exist
        if col not in subset.columns:
            return []
        
        # Group by channel and aggregate
        stats = (
            subset.groupby("channel")[col]
            .agg(agg)
            .sort_values(ascending=False)
            .head(top_k)
            .items()
        )
        
        # Format numeric values
        return [[ch, float(f"{val:.3g}")] for ch, val in stats]
    
    def _summarize_viewer_intent(self, titles: List[str]) -> str:
        """
        Use LLM to summarize viewer intent from titles
        
        Args:
            titles: List of video titles
            
        Returns:
            Summarized viewer intent
        """
        if not self._llm:
            return ""
        
        # Create prompt
        prompt = f"""
        You are a top YouTube strategist, with a background in psychology of media and choice.
        Summarize in **one or two sentences** the key question,
        curiosity or desire a viewer has when watching videos
        with titles like:
        
        {json.dumps(titles[:20], indent=2)}
        """
        
        # Get completion
        result = self._llm.generate_completion(prompt)
        
        # Return content or empty string on error
        return result.get("content", "")
    
    @staticmethod
    def _validate_dataframe(df: pd.DataFrame) -> None:
        """
        Validate that the DataFrame contains required columns
        
        Args:
            df: DataFrame to validate
            
        Raises:
            ValueError: If required columns are missing
        """
        missing = set(["cluster", "title", "clean_title"]).difference(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")