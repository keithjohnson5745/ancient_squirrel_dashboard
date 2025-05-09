import pandas as pd
import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

from ..core.base_processor import BaseProcessor
from ..utils.llm_adapter import LLMAdapter
from ..utils.text_utils import preprocess_text, analyze_linguistic_patterns

class NLPAnalyzer(BaseProcessor):
    """Enhanced NLP capabilities for YouTube video title analysis"""
    
    def __init__(self, config: Dict[str, Any], num_workers: Optional[int] = None,
                logger: Optional[logging.Logger] = None):
        """
        Initialize the NLP analyzer
        
        Args:
            config: Configuration dictionary
            num_workers: Number of worker processes
            logger: Logger instance
        """
        super().__init__(num_workers, logger)
        self.config = config
        
        # Initialize NLP components
        self.use_openai = config.get("use_openai", False)
        self.use_llm = config.get("use_llm", False)
        
        # Initialize LLM adapter if enabled
        self.llm_adapter = None
        if self.use_openai:
            openai_key = config.get("openai_api_key")
            if openai_key:
                self.llm_adapter = LLMAdapter(
                    provider="openai",
                    api_key=openai_key,
                    fallback_to_local=True
                )
            else:
                self.logger.warning("OpenAI API key not provided, using fallback options")
                self.llm_adapter = LLMAdapter(fallback_to_local=True)
    
    def process(self, df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Process data and return results
        
        Args:
            df: Input DataFrame
            **kwargs: Additional options
            
        Returns:
            Tuple of (enhanced DataFrame, analysis results dict)
        """
        # Copy the input DataFrame to avoid modifying the original
        result_df = df.copy()
        
        # Get text column from config or kwargs
        text_col = kwargs.get("text_col", self.config.get("text_col", "title"))
        clean_text_col = kwargs.get("clean_text_col", self.config.get("clean_text_col", "clean_title"))
        
        # Use clean text if available, otherwise use original titles
        if clean_text_col in df.columns:
            text_column = clean_text_col
        else:
            text_column = text_col
            
            # If clean_text_col doesn't exist but is requested, create it
            if clean_text_col not in df.columns:
                self.logger.info(f"Creating clean title column: {clean_text_col}")
                result_df[clean_text_col] = df[text_col].apply(preprocess_text)
                text_column = clean_text_col
        
        # Get text data
        texts = result_df[text_column].fillna('').tolist()
        
        # Initialize results dictionary
        results = {
            'metadata': {
                'text_column': text_column,
                'title_count': len(texts),
                'unique_titles': len(set(texts)),
            },
            'embeddings': {},
            'topic_modeling': {},
            'linguistic_analysis': {},
            'llm_insights': {}
        }
        
        # 1. Generate embeddings if LLM adapter is available
        if self.llm_adapter:
            self.logger.info("Generating embeddings")
            try:
                embeddings = self.llm_adapter.generate_embeddings(texts)
                results['embeddings']['method'] = self.llm_adapter.provider
                result_df['embedding'] = list(embeddings)
            except Exception as e:
                self.logger.error(f"Error generating embeddings: {str(e)}")
                results['embeddings']['error'] = str(e)
        
        # 2. Extract topics with NMF
        num_topics = kwargs.get("num_topics", self.config.get("num_topics", 15))
        self.logger.info(f"Extracting {num_topics} topics with NMF")
        
        try:
            nmf_results = self._extract_topics_nmf(texts, num_topics=num_topics)
            
            if 'error' not in nmf_results:
                result_df['nmf_topic'] = nmf_results['dominant_topics']
                results['topic_modeling']['nmf'] = {
                    'topic_terms': nmf_results['topic_terms'],
                    'coherence': nmf_results['coherence']
                }
            else:
                self.logger.error(f"Error in NMF: {nmf_results['error']}")
                results['topic_modeling']['nmf_error'] = nmf_results['error']
        except Exception as e:
            self.logger.error(f"Error in topic extraction: {str(e)}")
            results['topic_modeling']['error'] = str(e)
        
        # 3. Analyze linguistic patterns
        self.logger.info("Analyzing linguistic patterns")
        try:
            linguistic_results = self._analyze_linguistic_patterns(texts)
            
            # Add key linguistic features to DataFrame
            question_indices = linguistic_results['question_indices']
            imperative_indices = linguistic_results['imperative_indices']
            
            result_df['is_question'] = result_df.index.isin(question_indices)
            result_df['is_imperative'] = result_df.index.isin(imperative_indices)
            
            # Store statistics in results
            results['linguistic_analysis'] = linguistic_results['stats']
            
        except Exception as e:
            self.logger.error(f"Error in linguistic analysis: {str(e)}")
            results['linguistic_analysis']['error'] = str(e)
        
        # 4. LLM-based insights (if enabled)
        if self.use_llm and self.llm_adapter:
            # 4.1. Title pattern analysis
            sample_size = kwargs.get("sample_size", self.config.get("sample_size", 200))
            self.logger.info(f"Analyzing title patterns with LLM (sample size: {sample_size})")
            
            try:
                pattern_results = self._analyze_titles_with_llm(
                    result_df, 
                    sample_size=sample_size, 
                    influence_col=kwargs.get("influence_col", "influence")
                )
                results['llm_insights']['title_patterns'] = pattern_results
            except Exception as e:
                self.logger.error(f"Error in LLM title pattern analysis: {str(e)}")
                results['llm_insights']['title_patterns_error'] = str(e)
            
            # 4.2. Channel content strategy
            channel_col = kwargs.get("channel_col", self.config.get("channel_col", "channel"))
            if channel_col in result_df.columns:
                self.logger.info("Analyzing channel content strategies")
                
                # Create dict of channel -> titles
                channel_titles = {}
                for channel, channel_df in result_df.groupby(channel_col):
                    channel_titles[channel] = channel_df[text_column].tolist()
                
                # Analyze top channels
                try:
                    strategy_results = self._analyze_channel_content_strategy(channel_titles)
                    results['llm_insights']['content_strategy'] = strategy_results
                except Exception as e:
                    self.logger.error(f"Error in channel strategy analysis: {str(e)}")
                    results['llm_insights']['content_strategy_error'] = str(e)
            
            # 4.3. Cluster validation
            cluster_col = kwargs.get("cluster_col", self.config.get("cluster_col", "cluster"))
            if cluster_col in result_df.columns:
                self.logger.info("Validating clusters with LLM")
                
                # Create dict of cluster -> titles
                cluster_titles = {}
                for cluster, cluster_df in result_df.groupby(cluster_col):
                    cluster_titles[str(cluster)] = cluster_df[text_column].tolist()
                
                # Validate clusters
                try:
                    cluster_validations = self._validate_clusters(cluster_titles)
                    results['llm_insights']['cluster_validations'] = cluster_validations
                except Exception as e:
                    self.logger.error(f"Error in cluster validation: {str(e)}")
                    results['llm_insights']['cluster_validations_error'] = str(e)
            
            # 4.4. Community analysis - enhanced with per-community topics and embeddings
            community_col = kwargs.get("community_col", self.config.get("community_col", "community"))
            influence_col = kwargs.get("influence_col", self.config.get("influence_col", "influence"))
            top_n_per_community = kwargs.get("top_videos_per_community", self.config.get("top_videos_per_community", 30))
            community_topics_enabled = kwargs.get("community_topics_enabled", self.config.get("community_topics_enabled", True))
            community_topics_count = kwargs.get("community_topics_count", self.config.get("community_topics_count", 8))

            if community_col in result_df.columns:
                self.logger.info(f"Beginning enhanced community analysis")
                
                community_results = {}
                
                # Step 1: Generate per-community embeddings if LLM adapter available
                if self.llm_adapter:
                    self.logger.info(f"Generating embeddings for each community")
                    community_embeddings = self._generate_community_embeddings(
                        result_df, 
                        community_col=community_col,
                        text_col=text_column
                    )
                    community_results['embeddings'] = community_embeddings
                
                # Step 2: Extract topics for each community
                if community_topics_enabled:
                    self.logger.info(f"Extracting {community_topics_count} topics for each community")
                    community_topics = self._extract_community_topics(
                        result_df,
                        community_col=community_col,
                        text_col=text_column,
                        num_topics=community_topics_count
                    )
                    community_results['topics'] = community_topics
                    
                    # Add community topic information to dataframe
                    try:
                        # Create a new column for community-specific topic assignment
                        result_df['community_topic'] = pd.NA
                        
                        # For each community, assign topics to the original dataframe
                        for comm_id, topic_data in community_topics.items():
                            index_to_topic = topic_data.get('index_to_topic', {})
                            
                            # Assign topics
                            for idx, topic in index_to_topic.items():
                                result_df.at[idx, 'community_topic'] = int(topic)
                                
                                # Also create a combined community-topic identifier
                                result_df.at[idx, 'community_topic_id'] = f"{comm_id}_{topic}"
                        
                        # Log success
                        self.logger.info(f"Successfully assigned community topics to dataframe")
                        
                    except Exception as e:
                        self.logger.error(f"Error assigning community topics to dataframe: {str(e)}")
                
                # Step 3: Perform LLM analysis of communities with topic information
                if self.use_llm and self.llm_adapter:
                    self.logger.info(f"Analyzing community content with LLM using topic information")
                    
                    # Get community topic data if available
                    topic_data = community_results.get('topics', None)
                    
                    # Perform LLM analysis with topic information
                    community_analyses = self._analyze_community_content(
                        result_df,
                        community_col=community_col,
                        text_col=text_column,
                        influence_col=influence_col,
                        top_videos_per_community=top_n_per_community,
                        community_topics=topic_data
                    )
                    community_results['llm_analysis'] = community_analyses
                
                # Add all community results to main results
                results['community_analysis'] = community_results
        
        return result_df, results
    
    def _extract_topics_nmf(self, texts: List[str], num_topics: int = 10,
                          max_features: int = 1000, min_df: int = 2,
                          max_df: float = 0.85) -> Dict[str, Any]:
        """
        Extract topics using Non-negative Matrix Factorization
        
        Args:
            texts: List of texts to analyze
            num_topics: Number of topics to extract
            max_features: Maximum number of features for TF-IDF
            min_df: Minimum document frequency for terms
            max_df: Maximum document frequency for terms
            
        Returns:
            Dictionary with NMF results
        """
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            stop_words='english'
        )
        
        # Create document-term matrix
        try:
            # Handle non-string values
            valid_texts = [text if isinstance(text, str) else "" for text in texts]
            tfidf = vectorizer.fit_transform(valid_texts)
            
            # Apply NMF
            nmf_model = NMF(
                n_components=num_topics,
                random_state=42,
                init='nndsvd'  # Better for sparse matrices
            )
            
            nmf_topics = nmf_model.fit_transform(tfidf)
            
            # Get feature names
            feature_names = vectorizer.get_feature_names_out()
            
            # Extract top terms for each topic
            topic_terms = []
            for topic_idx, topic in enumerate(nmf_model.components_):
                top_features_ind = topic.argsort()[:-11:-1]  # Get indices of top 10 terms
                top_terms = [(feature_names[i], float(topic[i])) for i in top_features_ind]
                topic_terms.append(top_terms)
            
            # Assign dominant topic to each document
            dominant_topics = np.argmax(nmf_topics, axis=1)
            
            # Calculate topic coherence (simplified)
            topic_coherence = {}
            for topic_idx, terms in enumerate(topic_terms):
                term_list = [term for term, _ in terms]
                # Calculate term co-occurrence (could be enhanced with proper coherence metrics)
                coherence = sum(1 for doc in valid_texts if all(term in doc.lower() for term in term_list[:2]))
                topic_coherence[str(topic_idx)] = coherence / max(1, len(valid_texts))
            
            return {
                'topic_terms': topic_terms,
                'document_topics': nmf_topics,
                'dominant_topics': dominant_topics,
                'coherence': topic_coherence
            }
            
        except Exception as e:
            self.logger.error(f"Error in NMF topic extraction: {str(e)}")
            return {
                'error': str(e)
            }
    
    def _analyze_linguistic_patterns(self, texts: List[str]) -> Dict[str, Any]:
        """
        Analyze linguistic patterns in texts
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Dictionary with linguistic pattern analysis
        """
        # Initialize result containers
        question_indices = []
        imperative_indices = []
        all_patterns = []
        
        # Process each text
        for i, text in enumerate(tqdm(texts, desc="Analyzing linguistic patterns")):
            # Skip empty texts
            if not text or not isinstance(text, str):
                continue
                
            # Analyze patterns
            patterns = analyze_linguistic_patterns(text)
            
            # Track question and imperative titles
            if patterns["is_question"]:
                question_indices.append(i)
            
            if patterns["is_imperative"]:
                imperative_indices.append(i)
            
            all_patterns.append(patterns["patterns"])
        
        # Calculate statistics
        total_texts = len(texts)
        question_pct = len(question_indices) / total_texts * 100 if total_texts > 0 else 0
        imperative_pct = len(imperative_indices) / total_texts * 100 if total_texts > 0 else 0
        
        # Count pattern occurrences
        pattern_counts = {}
        for patterns in all_patterns:
            for pattern in patterns:
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        # Calculate pattern percentages
        pattern_pct = {pattern: count / total_texts * 100 
                      for pattern, count in pattern_counts.items()}
        
        return {
            'question_indices': question_indices,
            'imperative_indices': imperative_indices,
            'stats': {
                'question_pct': question_pct,
                'imperative_pct': imperative_pct,
                'pattern_counts': pattern_counts,
                'pattern_pct': pattern_pct
            }
        }
    
    def _analyze_titles_with_llm(self, df: pd.DataFrame, sample_size: int = 200,
                               influence_col: str = 'influence') -> Dict[str, Any]:
        """
        Use LLM to analyze patterns in titles
        
        Args:
            df: DataFrame with video data
            sample_size: Number of titles to sample
            influence_col: Column containing influence scores
            
        Returns:
            Dictionary with title pattern analysis
        """
        if not self.llm_adapter:
            return {"error": "LLM adapter not available"}
        
        # Check if influence column exists
        if influence_col not in df.columns:
            self.logger.warning(f"Influence column '{influence_col}' not found, using random sampling")
            # Fall back to random sampling
            if len(df) > sample_size:
                sample_df = df.sample(sample_size)
            else:
                sample_df = df
        else:
            # Sort by influence and take top videos
            sample_df = df.sort_values(influence_col, ascending=False).head(sample_size)
        
        # Extract titles
        sample_texts = sample_df['title'].tolist() if 'title' in sample_df.columns else []
        
        # Create prompt for pattern analysis
        prompt = """Analyze this sample of YouTube video titles from the most influential videos and identify:
1. Common structural patterns (how titles are formatted)
2. Recurring themes or topics
3. Linguistic devices used (e.g., questions, commands, hyperbole)
4. Content strategies evident from titles
5. Title templates or formulas that appear multiple times

Provide specific examples for each pattern you identify.

Sample titles from top influential videos:
"""
        
        for text in sample_texts:
            if isinstance(text, str):
                prompt += f"- {text}\n"
        
        # Generate completion
        response = self.llm_adapter.generate_completion(
            prompt=prompt,
            max_tokens=2000,
            temperature=0.7
        )
        
        # Return the analysis
        return {
            "pattern_analysis": response.get("content", "Error generating analysis"),
            "sample_size": len(sample_texts),
            "selection_method": "top_influence" if influence_col in df.columns else "random"
        }
    
    def _analyze_channel_content_strategy(self, channel_titles_dict: Dict[str, List[str]],
                                       max_channels: int = 5, max_titles: int = 50) -> Dict[str, Any]:
        """
        Use LLM to analyze content strategy for channels
        
        Args:
            channel_titles_dict: Dictionary mapping channel names to their video titles
            max_channels: Maximum number of channels to analyze
            max_titles: Maximum number of titles per channel
            
        Returns:
            Dictionary with channel content strategy analysis
        """
        if not self.llm_adapter:
            return {"error": "LLM adapter not available"}
        
        # Limit to top channels by video count
        channels = sorted(channel_titles_dict.keys(), 
                         key=lambda c: len(channel_titles_dict[c]), 
                         reverse=True)[:max_channels]
        
        channel_insights = {}
        
        # Process each channel
        for channel in channels:
            titles = channel_titles_dict[channel]
            # Limit number of titles to analyze
            sample_titles = titles[:max_titles]
            
            # Create prompt for content strategy analysis
            prompt = f"""Analyze these video titles from the YouTube channel "{channel}" and provide insights on:
1. The channel's apparent content strategy and target audience
2. Key topics and themes they focus on
3. How they structure titles to attract viewers
4. Any SEO patterns visible in their titling approach
5. Content gaps or opportunities they might be missing
6. How their strategy compares to typical YouTube best practices

Channel: {channel}
Sample titles:
"""
            
            for title in sample_titles:
                if isinstance(title, str):
                    prompt += f"- {title}\n"
            
            # Generate completion
            response = self.llm_adapter.generate_completion(
                prompt=prompt,
                max_tokens=1500,
                temperature=0.7
            )
            
            # Store the analysis
            channel_insights[channel] = {
                "strategy_analysis": response.get("content", "Error generating analysis"),
                "sample_size": len(sample_titles),
                "total_videos": len(titles)
            }
        
        return channel_insights
    
    def _validate_clusters(self, cluster_titles_dict: Dict[str, List[str]],
                         max_clusters: int = 10, max_titles: int = 20) -> Dict[str, Any]:
        """
        Use LLM to validate and name clusters
        
        Args:
            cluster_titles_dict: Dictionary mapping cluster IDs to their video titles
            max_clusters: Maximum number of clusters to analyze
            max_titles: Maximum number of titles per cluster
            
        Returns:
            Dictionary with cluster validation analysis
        """
        if not self.llm_adapter:
            return {"error": "LLM adapter not available"}
        
        # Sort clusters by size and limit
        clusters = sorted(cluster_titles_dict.keys(), 
                         key=lambda c: len(cluster_titles_dict[c]), 
                         reverse=True)[:max_clusters]
        
        validated_clusters = {}
        
        # Process each cluster
        for cluster_id in clusters:
            titles = cluster_titles_dict[cluster_id]
            # Limit sample size
            sample_titles = titles[:max_titles]
            
            # Create prompt for cluster validation
            prompt = f"""Here are titles that were grouped together by an algorithm. For this cluster:
1. Suggest an appropriate name or label that describes what unites these titles
2. Identify the common themes, topics, or patterns
3. Rate the cluster coherence on a scale of 1-10
4. Identify any titles that seem to be outliers or don't fit well
5. Suggest any sub-clusters within this group

Cluster #{cluster_id} titles:
"""
            
            for title in sample_titles:
                if isinstance(title, str):
                    prompt += f"- {title}\n"
            
            # Generate completion
            response = self.llm_adapter.generate_completion(
                prompt=prompt,
                max_tokens=1000,
                temperature=0.5
            )
            
            # Store the validation results
            validated_clusters[cluster_id] = {
                "validation": response.get("content", "Error generating validation"),
                "sample_size": len(sample_titles),
                "total_titles": len(titles)
            }
        
        return validated_clusters
        
    def _extract_community_topics(self, df: pd.DataFrame, community_col: str, 
                                text_col: str, num_topics: int = 8) -> Dict[str, Any]:
        """
        Extract topics separately for each community using NMF
        
        Args:
            df: DataFrame with video data
            community_col: Column containing community IDs
            text_col: Column containing preprocessed text
            num_topics: Number of topics to extract per community
            
        Returns:
            Dictionary mapping community IDs to their topic analysis results
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import NMF
        import numpy as np
        
        # Initialize results dictionary
        community_topics = {}
        
        # Get unique communities
        communities = df[community_col].unique()
        
        # Process each community separately
        for community_id in tqdm(communities, desc="Extracting community topics"):
            # Get community subset
            community_df = df[df[community_col] == community_id]
            
            # Skip if too few videos
            if len(community_df) < max(5, num_topics):
                self.logger.warning(f"Community {community_id} has too few videos ({len(community_df)}) for topic modeling. Skipping.")
                continue
            
            # Get text data for this community
            texts = community_df[text_col].fillna('').tolist()
            valid_texts = [text if isinstance(text, str) else "" for text in texts]
            
            # Skip if not enough valid text
            if len([t for t in valid_texts if t.strip()]) < max(5, num_topics):
                self.logger.warning(f"Community {community_id} has too few valid text entries for topic modeling. Skipping.")
                continue
                
            try:
                # Create TF-IDF vectorizer
                vectorizer = TfidfVectorizer(
                    max_features=min(1000, len(valid_texts) * 10),  # Adapt features to corpus size
                    min_df=2,
                    max_df=0.85,
                    stop_words='english'
                )
                
                # Create document-term matrix
                tfidf = vectorizer.fit_transform(valid_texts)
                
                # Adjust number of topics if necessary
                actual_num_topics = min(num_topics, tfidf.shape[0] - 1, tfidf.shape[1] - 1)
                if actual_num_topics < num_topics:
                    self.logger.warning(f"Reducing topics for community {community_id} to {actual_num_topics} due to data constraints")
                
                # Skip if matrix is too small for meaningful topics
                if actual_num_topics < 3:
                    self.logger.warning(f"Community {community_id} matrix too small for topic modeling. Skipping.")
                    continue
                    
                # Apply NMF
                nmf_model = NMF(
                    n_components=actual_num_topics,
                    random_state=42,
                    init='nndsvd'  # Better for sparse matrices
                )
                
                nmf_topics = nmf_model.fit_transform(tfidf)
                
                # Get feature names
                feature_names = vectorizer.get_feature_names_out()
                
                # Extract top terms for each topic
                topic_terms = []
                for topic_idx, topic in enumerate(nmf_model.components_):
                    top_features_ind = topic.argsort()[:-11:-1]  # Get indices of top 10 terms
                    top_terms = [(feature_names[i], float(topic[i])) for i in top_features_ind]
                    topic_terms.append(top_terms)
                
                # Assign dominant topic to each document
                dominant_topics = np.argmax(nmf_topics, axis=1)
                
                # Create mapping of original dataframe indices to topic assignments
                original_indices = community_df.index.tolist()
                index_to_topic = {idx: topic for idx, topic in zip(original_indices, dominant_topics)}
                
                # Calculate topic coherence (simplified)
                topic_coherence = {}
                for topic_idx, terms in enumerate(topic_terms):
                    term_list = [term for term, _ in terms]
                    # Calculate term co-occurrence
                    coherence = sum(1 for doc in valid_texts if all(term in doc.lower() for term in term_list[:2]))
                    topic_coherence[str(topic_idx)] = coherence / max(1, len(valid_texts))
                
                # Sample videos for each topic
                topic_videos = {}
                for topic_idx in range(actual_num_topics):
                    # Get videos with this dominant topic
                    topic_mask = dominant_topics == topic_idx
                    if not any(topic_mask):
                        continue
                        
                    # Get sample of video titles
                    sample_indices = [original_indices[i] for i, mask in enumerate(topic_mask) if mask][:5]
                    sample_titles = df.loc[sample_indices]['title'].tolist() if 'title' in df.columns else []
                    
                    topic_videos[str(topic_idx)] = sample_titles
                
                # Store results for this community
                community_topics[str(community_id)] = {
                    'topic_terms': topic_terms,
                    'topic_coherence': topic_coherence,
                    'topic_videos': topic_videos,
                    'video_count': len(community_df),
                    'index_to_topic': index_to_topic  # Map of original indices to topic assignments
                }
                
            except Exception as e:
                self.logger.error(f"Error in topic extraction for community {community_id}: {str(e)}")
                continue
        
        return community_topics

    def _generate_community_embeddings(self, df: pd.DataFrame, community_col: str, 
                                     text_col: str) -> Dict[str, Any]:
        """
        Generate embeddings separately for each community
        
        Args:
            df: DataFrame with video data
            community_col: Column containing community IDs
            text_col: Column containing preprocessed text
            
        Returns:
            Dictionary mapping community IDs to embedding results
        """
        # Initialize results dictionary
        community_embeddings = {}
        
        # Skip if LLM adapter not available
        if not self.llm_adapter:
            self.logger.warning("LLM adapter not available, skipping community embeddings")
            return community_embeddings
        
        # Get unique communities
        communities = df[community_col].unique()
        
        # Process each community separately
        for community_id in tqdm(communities, desc="Generating community embeddings"):
            # Get community subset
            community_df = df[df[community_col] == community_id]
            
            # Skip if too few videos
            if len(community_df) < 5:
                self.logger.warning(f"Community {community_id} has too few videos for embeddings. Skipping.")
                continue
            
            # Get text data for this community
            texts = community_df[text_col].fillna('').tolist()
            
            try:
                # Generate embeddings
                embeddings = self.llm_adapter.generate_embeddings(texts)
                
                # Store results
                community_embeddings[str(community_id)] = {
                    'embedding_count': len(embeddings),
                    'embedding_provider': self.llm_adapter.provider,
                    'community_size': len(community_df)
                }
                
                # Add embeddings to dataframe (subset only)
                for idx, embedding in zip(community_df.index, embeddings):
                    df.at[idx, 'embedding'] = embedding
                
            except Exception as e:
                self.logger.error(f"Error generating embeddings for community {community_id}: {str(e)}")
        
        return community_embeddings

    def _analyze_community_content(self, df: pd.DataFrame, community_col: str, 
                                 text_col: str, influence_col: Optional[str] = None,
                                 top_videos_per_community: int = 30,
                                 community_topics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Use LLM to analyze content strategy for communities, incorporating topic information
        
        Args:
            df: DataFrame with video data
            community_col: Column containing community IDs
            text_col: Column containing preprocessed text
            influence_col: Column containing influence scores (optional)
            top_videos_per_community: Number of top influential videos to analyze per community
            community_topics: Dictionary of community topic analysis (optional)
            
        Returns:
            Dictionary with community content analysis
        """
        if not self.llm_adapter:
            return {"error": "LLM adapter not available"}
        
        # Get unique communities
        communities = df[community_col].unique()
        
        # Sort by size if we need to limit
        if len(communities) > 10:
            community_sizes = {comm: len(df[df[community_col] == comm]) for comm in communities}
            communities = sorted(communities, key=lambda c: community_sizes.get(c, 0), reverse=True)[:10]
        
        community_insights = {}
        
        # Process each community
        for community_id in tqdm(communities, desc="Analyzing community content"):
            community_df = df[df[community_col] == community_id]
            
            # Skip if too few videos
            if len(community_df) < 5:
                self.logger.warning(f"Community {community_id} has too few videos for analysis. Skipping.")
                continue
            
            try:
                # Sort by influence if column exists
                if influence_col and influence_col in community_df.columns:
                    try:
                        # Convert to numeric, handling non-numeric values
                        community_df['_influence'] = pd.to_numeric(community_df[influence_col], errors='coerce')
                        community_df = community_df.sort_values('_influence', ascending=False)
                    except Exception as e:
                        self.logger.warning(f"Error sorting by influence: {str(e)}. Using unsorted data.")
                
                # Select top videos
                top_df = community_df.head(top_videos_per_community)
                
                # Get titles for prompt
                titles = top_df['title'].tolist() if 'title' in top_df.columns else top_df[text_col].tolist()
                titles = [t for t in titles if isinstance(t, str) and t.strip()][:top_videos_per_community]
                
                # Build the prompt, incorporating topic information if available
                prompt = f"""Analyze this YouTube community (community #{community_id}) and provide insights on:
1. The community's apparent content focus and target audience
2. Key topics and themes that unify this community
3. How videos in this community structure titles to attract viewers
4. Any SEO patterns or keyword strategies visible in their approach
5. Content gaps or opportunities this community might be missing
6. How this community compares to typical YouTube content strategies

Community: #{community_id}
Sample titles from most influential videos:
"""
                
                # Add titles to prompt
                for title in titles:
                    prompt += f"- {title}\n"
                
                # Add topic information if available
                topic_info = community_topics.get(str(community_id)) if community_topics else None
                if topic_info and 'topic_terms' in topic_info:
                    prompt += "\nThis community contains several subtopics:\n"
                    
                    for i, topic_terms in enumerate(topic_info['topic_terms']):
                        # Format topic terms
                        terms = ", ".join([term for term, _ in topic_terms[:5]])
                        
                        # Add sample videos if available
                        sample_videos = topic_info.get('topic_videos', {}).get(str(i), [])
                        sample_text = ""
                        if sample_videos:
                            sample_text = "\n   Example video: " + sample_videos[0]
                        
                        prompt += f"Topic {i+1}: {terms}{sample_text}\n"
                
                # Generate completion
                response = self.llm_adapter.generate_completion(
                    prompt=prompt,
                    max_tokens=1500,
                    temperature=0.7
                )
                
                # Store the analysis
                community_insights[str(community_id)] = {
                    "content_analysis": response.get("content", "Error generating analysis"),
                    "sample_size": len(titles),
                    "total_videos": len(community_df),
                    "has_topic_info": bool(topic_info)
                }
                
            except Exception as e:
                self.logger.error(f"Error analyzing community {community_id}: {str(e)}")
                community_insights[str(community_id)] = {
                    "error": str(e),
                    "sample_size": 0,
                    "total_videos": len(community_df)
                }
        
        return community_insights