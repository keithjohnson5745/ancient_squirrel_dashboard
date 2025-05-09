import json

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
import os
from datetime import datetime
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from sklearn.cluster import KMeans, MiniBatchKMeans
from collections import defaultdict  # Add this import

from ..core.base_processor import BaseProcessor
from ..utils.text_utils import preprocess_text
from ..utils.data_utils import process_vector_column
from ..core.config import AnalysisConfig
from .thumbnail_processor import ThumbnailProcessor
from .image_analyzer import ImageAnalyzer
from .title_thumbnail_analyzer import TitleThumbnailAnalyzer

class YouTubeDataProcessor(BaseProcessor):
    import json
    """Scalable processor for large YouTube video collections with theme analysis"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, 
                num_workers: Optional[int] = None,
                logger: Optional[logging.Logger] = None):
        """
        Initialize the YouTube data processor
        
        Args:
            config: Configuration dictionary or AnalysisConfig
            num_workers: Number of worker processes
            logger: Logger instance
        """
        super().__init__(num_workers, logger)
        
        # Convert config to dictionary if it's an AnalysisConfig object
        if isinstance(config, AnalysisConfig):
            self.config = vars(config)
        else:
            self.config = config or {}
        
        # Set up stopwords
        try:
            import nltk
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
            
            from nltk.corpus import stopwords
            self.stop_words = set(stopwords.words('english'))
            self.custom_stops = {'video', 'youtube', 'watch', 'subscribe', 'channel', 'like',
                              'comment', 'share', 'official', 'ft', 'featuring', 'presents'}
            self.stop_words.update(self.custom_stops)
        except ImportError:
            self.logger.warning("NLTK not installed, using basic stopwords")
            self.stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 'when',
                            'at', 'from', 'by', 'for', 'with', 'about', 'to', 'in', 'on', 'video', 'youtube'}
    
    def _sample_data_for_analysis(self, df: pd.DataFrame, 
                             community_col: str,
                             influence_col: str,
                             top_communities_count: int = 15,
                             videos_per_community: int = 50,
                             min_community_size: int = 10,
                             stratified_sampling: bool = True) -> pd.DataFrame:
        """
        Sample data for thumbnail and joint analysis to focus on most insightful videos
        
        Args:
            df: Input DataFrame with video data
            community_col: Column containing community IDs
            influence_col: Column containing influence scores
            top_communities_count: Number of top communities to analyze
            videos_per_community: Maximum videos to analyze per community
            min_community_size: Minimum community size for analysis
            stratified_sampling: Whether to use stratified sampling across influence levels
            
        Returns:
            Sampled DataFrame
        """
        self.logger.info(f"Sampling data for analysis: top {top_communities_count} communities, "
                    f"{videos_per_community} videos per community")
        
        # Validate input columns
        if community_col not in df.columns:
            self.logger.warning(f"Community column '{community_col}' not found. Skipping sampling.")
            return df
        
        # If influence column is missing, use video count as proxy for community importance
        using_video_count = False
        if influence_col not in df.columns:
            self.logger.warning(f"Influence column '{influence_col}' not found. Using video count as proxy.")
            influence_col = "video_count"
            community_counts = df[community_col].value_counts().reset_index()
            community_counts.columns = [community_col, "video_count"]
            df = df.merge(community_counts, on=community_col)
            using_video_count = True
        
        # Calculate total community influence
        community_influence = df.groupby(community_col)[influence_col].agg(
            total_influence=lambda x: x.sum() if not using_video_count else len(x),
            avg_influence=lambda x: x.mean() if not using_video_count else None
        ).reset_index()
        
        # Sort communities by total influence
        community_influence = community_influence.sort_values('total_influence', ascending=False)
        
        # Get top communities
        top_communities = community_influence.head(top_communities_count)[community_col].tolist()
        
        self.logger.info(f"Selected {len(top_communities)} top communities for analysis")
        
        # Function for stratified sampling
        def get_stratified_sample(community_df, influence_col, total_videos=50):
            """Get a stratified sample of videos across influence quantiles"""
            if len(community_df) <= total_videos:
                return community_df
                
            # Create influence quantiles (5 by default)
            try:
                community_df['influence_quantile'] = pd.qcut(
                    community_df[influence_col], 
                    q=5,  # 5 quantiles
                    labels=False
                )
            except ValueError:
                # Handle case where there aren't enough unique values for qcut
                self.logger.warning(f"Not enough unique influence values for quantiles in community. Using equal weights.")
                return community_df.sample(min(total_videos, len(community_df)))
            
            # Calculate videos per quantile
            videos_per_quantile = total_videos // 5
            remainder = total_videos % 5
            
            # Sample from each quantile
            sample = pd.DataFrame()
            for quantile in range(5):
                quantile_df = community_df[community_df['influence_quantile'] == quantile]
                
                # Add one extra to early quantiles if there's remainder
                extra = 1 if quantile < remainder else 0
                
                # Handle case where quantile has fewer videos than requested
                quantile_sample_size = min(len(quantile_df), videos_per_quantile + extra)
                
                # Sample from this quantile
                quantile_sample = quantile_df.sample(quantile_sample_size) if quantile_sample_size < len(quantile_df) else quantile_df
                
                sample = pd.concat([sample, quantile_sample])
            
            return sample
        
        # Create sample DataFrame
        sample_df = pd.DataFrame()
        
        # Select videos from each community
        for community in top_communities:
            community_df = df[df[community_col] == community]
            
            # Skip if community is too small
            if len(community_df) < min_community_size:
                self.logger.info(f"Skipping community {community}: too few videos ({len(community_df)} < {min_community_size})")
                continue
            
            self.logger.info(f"Sampling from community {community} ({len(community_df)} videos)")
            
            # Use stratified sampling if enabled
            if stratified_sampling and not using_video_count:
                community_sample = get_stratified_sample(community_df, influence_col, videos_per_community)
            else:
                # Otherwise use top N by influence
                community_sample = community_df.sort_values(influence_col, ascending=False).head(videos_per_community)
            
            self.logger.info(f"Selected {len(community_sample)} videos from community {community}")
            sample_df = pd.concat([sample_df, community_sample])
        
        # If no communities meet criteria, use overall top videos
        if len(sample_df) == 0:
            self.logger.warning("No communities met sampling criteria. Using top videos overall.")
            sample_df = df.sort_values(influence_col, ascending=False).head(top_communities_count * videos_per_community)
        
        self.logger.info(f"Final sample size: {len(sample_df)} videos "
                    f"({len(sample_df) / len(df) * 100:.1f}% of original dataset)")
        
        return sample_df

    def process(self, data: pd.DataFrame = None, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Process data and return results
        
        Args:
            data: Input DataFrame (optional, can be loaded from file)
            **kwargs: Additional options
            
        Returns:
            Tuple of (enhanced DataFrame, analysis results dict)
        """
        start_time = datetime.now()
        
        # Get input file from kwargs or config
        input_file = kwargs.get("input_file", self.config.get("input_file"))
        
        # Load data if not provided
        if data is None and input_file:
            data = self.load_data(input_file)
        elif data is None:
            raise ValueError("Either data or input_file must be provided")
        
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Preprocess titles
        self.logger.info("Preprocessing video titles")
        if "clean_title" not in df.columns:
            df = self.preprocess_titles(df)
        
        # Generate word vectors if needed
        if self.config.get("build_vectors", True) and "doc_vector" not in df.columns:
            self.logger.info("Building word vectors")
            word2vec_model = self.train_word_vectors(df)
            df = self.generate_document_vectors(df, word2vec_model)
        
        # Cluster videos if needed
        cluster_col = self.config.get("cluster_col", "cluster")
        if cluster_col not in df.columns:
            self.logger.info("Clustering videos")
            num_clusters = kwargs.get("num_clusters", self.config.get("num_clusters", 30))
            df, clustering_model = self.cluster_videos(df, n_clusters=num_clusters)
        
        # Create output directory
        output_dir = kwargs.get("output_dir", self.config.get("output_dir", "output"))
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine if sampling should be used
        use_sampling = kwargs.get("use_sampling", self.config.get("use_sampling", True))
        if kwargs.get("no_sampling", False):
            use_sampling = False
        
        # Create sampled dataset for thumbnail analysis if enabled
        if use_sampling and (
            self.config.get("download_thumbnails", False) or 
            self.config.get("analyze_thumbnails", False) or 
            self.config.get("analyze_title_thumbnail", False)
        ):
            community_col = kwargs.get("community_col", self.config.get("community_col", "community"))
            influence_col = kwargs.get("influence_col", self.config.get("influence_col", "influence"))
            
            self.logger.info("Creating optimized sample for thumbnail analysis")
            
            # Apply sampling to create an efficient subset for thumbnail processing
            thumbnail_df = self._sample_data_for_analysis(
                df,
                community_col=community_col, 
                influence_col=influence_col,
                top_communities_count=kwargs.get("top_communities", self.config.get("top_communities_count", 15)),
                videos_per_community=kwargs.get("videos_per_community", self.config.get("videos_per_community", 50)),
                min_community_size=kwargs.get("min_community_size", self.config.get("min_community_size", 10)),
                stratified_sampling=kwargs.get("stratified_sampling", self.config.get("stratified_sampling", True))
            )
            
            # Save the sampled dataset
            sample_file = os.path.join(output_dir, "thumbnail_analysis_sample.json")
            self.save_data(thumbnail_df, sample_file)
            
            self.logger.info(f"Using sampled dataset with {len(thumbnail_df)} videos for thumbnail analysis")
        else:
            # Use the full dataset if sampling is disabled
            thumbnail_df = df.copy()
            self.logger.info(f"Using full dataset with {len(thumbnail_df)} videos for thumbnail analysis")
        
        # Download thumbnails if enabled
        if self.config.get("download_thumbnails", False):
            self.logger.info("Downloading video thumbnails")
            
            # Create thumbnail processor
            thumbnail_processor = ThumbnailProcessor(self.config, self.num_workers, self.logger)
            
            # Process the sampled dataset for thumbnails
            thumbnail_df, thumbnail_results = thumbnail_processor.process(
                thumbnail_df,  # Use the sampled dataset
                video_id_col=kwargs.get("video_id_col", self.config.get("video_id_col", "video_id")),
                thumbnail_col=kwargs.get("thumbnail_col", self.config.get("thumbnail_col", "thumbnail_path")),
                force_download=kwargs.get("force_download", self.config.get("force_download", False))
            )
            
            # Save interim data with thumbnail paths
            interim_file = os.path.join(output_dir, "data_with_thumbnails.json")
            self.save_data(thumbnail_df, interim_file)
            
            # Copy thumbnail paths back to main DataFrame if using sampling
            if use_sampling and "thumbnail_path" in thumbnail_df.columns:
                if "thumbnail_path" not in df.columns:
                    df["thumbnail_path"] = pd.NA
                    
                # Get video ID column name for joining
                video_id_col = kwargs.get("video_id_col", self.config.get("video_id_col", "video_id"))
                
                if video_id_col in df.columns and video_id_col in thumbnail_df.columns:
                    # Create a mapping of video_id to thumbnail_path
                    thumbnail_map = dict(zip(thumbnail_df[video_id_col], thumbnail_df["thumbnail_path"]))
                    
                    # Update main DataFrame with thumbnail paths
                    for idx, row in df.iterrows():
                        video_id = row.get(video_id_col)
                        if video_id in thumbnail_map:
                            df.at[idx, "thumbnail_path"] = thumbnail_map[video_id]
                else:
                    # Fall back to index-based updating if video_id column not available
                    for idx, row in thumbnail_df.iterrows():
                        if idx in df.index:
                            df.at[idx, "thumbnail_path"] = row["thumbnail_path"]
        
        # Analyze thumbnails if enabled
        if self.config.get("analyze_thumbnails", False) and "thumbnail_path" in thumbnail_df.columns:
            self.logger.info("Analyzing thumbnail images")
            
            # Create image analyzer
            image_analyzer = ImageAnalyzer(self.config, self.num_workers, self.logger)
            
            # Process the sampled dataset for image analysis
            thumbnail_df, image_analysis_results = image_analyzer.process(
                thumbnail_df,  # Use the sampled dataset
                thumbnail_col=kwargs.get("thumbnail_col", self.config.get("thumbnail_col", "thumbnail_path")),
                title_col=kwargs.get("title_col", self.config.get("title_col", "title")),
                use_llm=kwargs.get("use_llm", self.config.get("use_llm", False))
            )
            
            # Save interim data with image analysis
            interim_file = os.path.join(output_dir, "data_with_image_analysis.json")
            self.save_data(thumbnail_df, interim_file)
            
            # Copy image analysis results back to main DataFrame if using sampling
            if use_sampling and "thumbnail_analysis" in thumbnail_df.columns:
                if "thumbnail_analysis" not in df.columns:
                    df["thumbnail_analysis"] = pd.NA
                    
                # Get video ID column name for joining
                video_id_col = kwargs.get("video_id_col", self.config.get("video_id_col", "video_id"))
                
                if video_id_col in df.columns and video_id_col in thumbnail_df.columns:
                    # Create a mapping of video_id to analysis
                    analysis_map = dict(zip(thumbnail_df[video_id_col], thumbnail_df["thumbnail_analysis"]))
                    
                    # Update main DataFrame with analysis
                    for idx, row in df.iterrows():
                        video_id = row.get(video_id_col)
                        if video_id in analysis_map:
                            df.at[idx, "thumbnail_analysis"] = analysis_map[video_id]
                else:
                    # Fall back to index-based updating if video_id column not available
                    for idx, row in thumbnail_df.iterrows():
                        if idx in df.index and "thumbnail_analysis" in row:
                            df.at[idx, "thumbnail_analysis"] = row["thumbnail_analysis"]
        
        # Perform joint title-thumbnail analysis if enabled
        if self.config.get("analyze_title_thumbnail", False) and "thumbnail_path" in thumbnail_df.columns:
            self.logger.info("Analyzing title-thumbnail pairs")
            
            # Create joint analyzer
            joint_analyzer = TitleThumbnailAnalyzer(self.config, self.num_workers, self.logger)
            
            # Process the sampled dataset for joint analysis
            thumbnail_df, joint_analysis_results = joint_analyzer.process(
                thumbnail_df,  # Use the sampled dataset
                video_id_col=kwargs.get("video_id_col", self.config.get("video_id_col", "video_id")),
                title_col=kwargs.get("title_col", self.config.get("title_col", "title")),
                thumbnail_col=kwargs.get("thumbnail_col", self.config.get("thumbnail_col", "thumbnail_path")),
                thumbnail_analysis_col=kwargs.get("thumbnail_analysis_col", "thumbnail_analysis"),
                community_col=kwargs.get("community_col", self.config.get("community_col", "community")),
                influence_col=kwargs.get("influence_col", self.config.get("influence_col", "influence")),
                use_llm=kwargs.get("use_llm", self.config.get("use_llm", False)),
                n_subclusters=kwargs.get("n_subclusters", self.config.get("n_subclusters", 3))
            )
            
            # Copy subcluster assignments back to main DataFrame if using sampling
            if use_sampling:
                # Get subcluster column name based on community column
                community_col = kwargs.get("community_col", self.config.get("community_col", "community"))
                subcluster_col = f"{community_col}_subcluster"
                
                if subcluster_col in thumbnail_df.columns:
                    if subcluster_col not in df.columns:
                        df[subcluster_col] = pd.NA
                    
                    # Get video ID column name for joining
                    video_id_col = kwargs.get("video_id_col", self.config.get("video_id_col", "video_id"))
                    
                    if video_id_col in df.columns and video_id_col in thumbnail_df.columns:
                        # Create a mapping of video_id to subcluster
                        subcluster_map = dict(zip(thumbnail_df[video_id_col], thumbnail_df[subcluster_col]))
                        
                        # Update main DataFrame with subcluster assignments
                        for idx, row in df.iterrows():
                            video_id = row.get(video_id_col)
                            if video_id in subcluster_map:
                                df.at[idx, subcluster_col] = subcluster_map[video_id]
                    else:
                        # Fall back to index-based updating if video_id column not available
                        for idx, row in thumbnail_df.iterrows():
                            if idx in df.index:
                                df.at[idx, subcluster_col] = row[subcluster_col]
        
        # Save processed data
        output_file = os.path.join(output_dir, "processed_data.json")
        self.save_data(df, output_file)
        
        # Extract cluster themes
        self.logger.info("Extracting cluster themes")
        cluster_themes = self.extract_cluster_themes(df)
        
        # Extract cluster stats
        self.logger.info("Extracting cluster statistics")
        cluster_stats = self.extract_cluster_stats(df)
        
        # Run additional analyses based on config
        results = {
            'metadata': {
                'input_file': input_file,
                'output_dir': output_dir,
                'video_count': len(df),
                'channel_count': df['channel'].nunique() if 'channel' in df.columns else 0,
                'cluster_count': df[cluster_col].nunique() if cluster_col in df.columns else 0,
                'processing_time': str(datetime.now() - start_time)
            },
            'cluster_themes': cluster_themes,
            'cluster_stats': cluster_stats
        }
        
        # Add sampling information if used
        if use_sampling:
            results['metadata']['sampling_info'] = {
                'enabled': True,
                'sampled_videos_count': len(thumbnail_df),
                'sampling_percentage': round(len(thumbnail_df) / len(df) * 100, 2),
                'top_communities_analyzed': kwargs.get("top_communities", self.config.get("top_communities_count", 15)),
                'videos_per_community': kwargs.get("videos_per_community", self.config.get("videos_per_community", 50)),
                'stratified_sampling': kwargs.get("stratified_sampling", self.config.get("stratified_sampling", True))
            }
        
        # Add thumbnail results if available
        if self.config.get("download_thumbnails", False):
            results['thumbnail_download'] = thumbnail_results.get("thumbnail_download", {})
        
        # Add image analysis results if available
        if self.config.get("analyze_thumbnails", False):
            results['thumbnail_analysis'] = image_analysis_results.get("thumbnail_analysis", {})
        
        # Add joint analysis results if available
        if self.config.get("analyze_title_thumbnail", False):
            results['title_thumbnail_analysis'] = joint_analysis_results.get("title_thumbnail_analysis", {})
        
        # Add cluster insights if enabled
        if self.config.get("enable_cluster_insights", True):
            from .cluster_analyzer import ClusterInsightExtractor
            
            insight_config = {
                "top_n_clusters": self.config.get("cluster_insight_top_n", 10),
                "llm_enabled": self.config.get("cluster_insight_use_llm", False),
                "openai_api_key": self.config.get("openai_api_key"),
                "tfidf_max_features": 1000,
                "bigram_min_freq": 3,
                "bigram_top_k": 5,
                "centroid_top_k": 5
            }
            
            cie = ClusterInsightExtractor(insight_config, self.num_workers, self.logger)
            _, cluster_insights = cie.process(df)
            results['cluster_insights'] = cluster_insights.get("cluster_insights", {})
        
        # Add subcluster insights if joint analysis was performed
        if self.config.get("analyze_title_thumbnail", False) and f"{cluster_col}_subcluster" in df.columns:
            # Extract insights for subclusters
            subcluster_insights = self._extract_subcluster_insights(
                df, 
                f"{cluster_col}_subcluster",
                kwargs.get("title_col", self.config.get("title_col", "title")),
                "title_thumbnail_analysis"
            )
            results['subcluster_insights'] = subcluster_insights
    
        # Save full results
        results_file = os.path.join(output_dir, "analysis_results.json")
        self.save_analysis(results, results_file)
        
        return df, results

    def _extract_subcluster_insights(self, df: pd.DataFrame, subcluster_col: str,
                                title_col: str, analysis_col: str) -> Dict[str, Any]:
        """
        Extract insights for subclusters
        
        Args:
            df: DataFrame with subcluster assignments
            subcluster_col: Column containing subcluster IDs
            title_col: Column containing video titles
            analysis_col: Column containing title-thumbnail analysis
            
        Returns:
            Dictionary with subcluster insights
        """
        insights = {}
        
        try:
            # Get unique subclusters
            subclusters = df[subcluster_col].dropna().unique()
            
            for subcluster_id in subclusters:
                # Filter for this subcluster
                subcluster_df = df[df[subcluster_col] == subcluster_id]
                
                # Skip if too few videos
                if len(subcluster_df) < 5:
                    continue
                
                # Extract common patterns
                patterns = defaultdict(int)
                
                for analysis_json in subcluster_df[analysis_col].dropna():
                    try:
                        analysis = json.loads(analysis_json)
                        for pattern in analysis.get("patterns", []):
                            patterns[pattern] += 1
                    except:
                        continue
                
                # Calculate pattern percentages
                pattern_pct = {
                    pattern: (count / len(subcluster_df) * 100)
                    for pattern, count in patterns.items()
                }
                
                # Get top video titles as examples
                sample_titles = subcluster_df[title_col].head(5).tolist()
                
                # Calculate average scores
                avg_scores = {}
                
                scores_to_extract = ["clickbait_score", "text_visual_alignment"]
                for score_name in scores_to_extract:
                    values = []
                    
                    for analysis_json in subcluster_df[analysis_col].dropna():
                        try:
                            analysis = json.loads(analysis_json)
                            value = analysis.get(score_name, None)
                            if value is not None:
                                values.append(value)
                        except:
                            continue
                    
                    if values:
                        avg_scores[score_name] = sum(values) / len(values)
                
                # Store insights
                insights[subcluster_id] = {
                    "size": len(subcluster_df),
                    "top_patterns": {k: v for k, v in sorted(pattern_pct.items(), key=lambda x: x[1], reverse=True)[:5]},
                    "avg_scores": avg_scores,
                    "sample_titles": sample_titles
                }
        
        except Exception as e:
            self.logger.error(f"Error extracting subcluster insights: {e}")
        
        return insights
    
    def preprocess_titles(self, df: pd.DataFrame, title_col: str = 'title',
                         output_col: str = 'clean_title') -> pd.DataFrame:
        """
        Preprocess video titles in parallel
        
        Args:
            df: DataFrame with video data
            title_col: Column containing video titles
            output_col: Column to store preprocessed titles
            
        Returns:
            DataFrame with preprocessed titles
        """
        # Create a copy to avoid modifying the original
        processed_df = df.copy()
        
        # Process titles in parallel
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Create list of futures
            futures = {
                executor.submit(preprocess_text, title, self.stop_words): i 
                for i, title in enumerate(df[title_col])
            }
            
            # Process results as they complete
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing titles"):
                idx = futures[future]
                processed_df.loc[idx, output_col] = future.result()
        
        return processed_df
    
    def train_word_vectors(self, df: pd.DataFrame, text_col: str = 'clean_title',
                        vector_size: int = 100, window: int = 5, min_count: int = 5) -> Any:
        """
        Train word2vec model for semantic analysis
        
        Args:
            df: DataFrame with preprocessed titles
            text_col: Column containing preprocessed text
            vector_size: Dimensionality of word vectors
            window: Context window size
            min_count: Minimum word count
            
        Returns:
            Word2Vec model
        """
        try:
            from gensim.models import Word2Vec
            
            # Prepare sentences (tokenized texts)
            sentences = [text.split() for text in df[text_col] if isinstance(text, str)]
            
            # Train model
            model = Word2Vec(
                sentences=sentences,
                vector_size=vector_size,
                window=window,
                min_count=min_count,
                workers=self.num_workers
            )
            
            return model
        except ImportError:
            self.logger.warning("Gensim not installed, using fallback vector method")
            return None
    
    def generate_document_vectors(self, df: pd.DataFrame, word2vec_model: Any = None,
                                text_col: str = 'clean_title',
                                agg_method: str = 'mean') -> pd.DataFrame:
        """
        Generate document vectors by aggregating word vectors
        
        Args:
            df: DataFrame with preprocessed titles
            word2vec_model: Trained Word2Vec model (optional)
            text_col: Column containing preprocessed text
            agg_method: Method to aggregate word vectors
            
        Returns:
            DataFrame with document vectors
        """
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # If no Word2Vec model and gensim not available, fall back to TF-IDF
        if word2vec_model is None:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.decomposition import TruncatedSVD
                
                self.logger.info("Generating document vectors using TF-IDF and SVD")
                
                # Create TF-IDF matrix
                vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(df[text_col].fillna(''))
                
                # Reduce dimensionality with SVD
                svd = TruncatedSVD(n_components=100, random_state=42)
                document_vectors = svd.fit_transform(tfidf_matrix)
                
                # Store as list of vectors
                result_df['doc_vector'] = list(document_vectors)
                return result_df
            except ImportError:
                self.logger.error("Neither gensim nor scikit-learn available, skipping vectors")
                return result_df
        
        # Function to convert text to vector using Word2Vec
        def text_to_vector(text, model, method='mean'):
            if not isinstance(text, str):
                # Return zero vector for non-text
                return np.zeros(model.wv.vector_size)
                
            words = text.split()
            vectors = [model.wv[word] for word in words if word in model.wv]
            
            if not vectors:
                return np.zeros(model.wv.vector_size)
                
            if method == 'mean':
                return np.mean(vectors, axis=0)
            elif method == 'sum':
                return np.sum(vectors, axis=0)
            else:
                # Default to mean
                return np.mean(vectors, axis=0)
        
        # Process in parallel
        vector_size = word2vec_model.wv.vector_size
        document_vectors = np.zeros((len(df), vector_size))
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Create list of futures
            futures = {
                executor.submit(text_to_vector, text, word2vec_model, agg_method): i 
                for i, text in enumerate(df[text_col])
            }
            
            # Process results
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating vectors"):
                idx = futures[future]
                document_vectors[idx] = future.result()
        
        # Store document vectors
        result_df['doc_vector'] = list(document_vectors)
        
        return result_df
    
    def cluster_videos(self, df: pd.DataFrame, vectors: Optional[np.ndarray] = None,
                     vector_col: str = 'doc_vector', n_clusters: Optional[int] = None,
                     method: str = 'kmeans', random_state: int = 42) -> Tuple[pd.DataFrame, Any]:
        """
        Cluster videos based on their vectors
        
        Args:
            df: DataFrame with video data
            vectors: Optional numpy array of vectors
            vector_col: Column containing document vectors
            n_clusters: Number of clusters
            method: Clustering method (kmeans, minibatch)
            random_state: Random seed
            
        Returns:
            Tuple of (DataFrame with cluster assignments, clustering model)
        """
        # Get vectors
        if vectors is None:
            if vector_col not in df.columns:
                raise ValueError(f"Column '{vector_col}' not found in DataFrame")
                
            # Process vector column if needed
            df = process_vector_column(df, vector_col)
            
            # Convert list of vectors to numpy array
            vectors = np.array(df[vector_col].tolist())
        
        # Determine number of clusters if not specified
        if n_clusters is None:
            # Heuristic: sqrt of number of samples, capped at 100
            n_clusters = min(100, int(np.sqrt(len(df))))
            self.logger.info(f"Automatically determined {n_clusters} clusters")
        
        # Perform clustering
        if method == 'kmeans':
            clustering = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        elif method == 'minibatch':
            # Faster for large datasets
            clustering = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, 
                                       batch_size=1000, n_init=10)
        else:
            raise ValueError(f"Unsupported clustering method: {method}")
        
        # Fit and predict
        clusters = clustering.fit_predict(vectors)
        
        # Add cluster assignments to DataFrame
        result_df = df.copy()
        result_df['cluster'] = clusters
        
        return result_df, clustering
    
    def extract_cluster_themes(self, df: pd.DataFrame, n_terms: int = 20,
                             text_col: str = 'clean_title') -> Dict[str, List[Tuple[str, float]]]:
        """
        Extract themes for each cluster using TF-IDF
        
        Args:
            df: DataFrame with cluster assignments
            n_terms: Number of terms to extract per cluster
            text_col: Column containing preprocessed text
            
        Returns:
            Dictionary mapping cluster IDs to lists of (term, score) pairs
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Check if cluster column exists
        if 'cluster' not in df.columns:
            self.logger.error("Cluster column not found in DataFrame")
            return {}
        
        # We'll analyze each cluster independently
        cluster_terms = {}
        
        # Loop through each cluster
        for cluster_id in tqdm(df['cluster'].unique(), desc="Extracting cluster themes"):
            self.logger.info(f"Analyzing cluster {cluster_id}")
            
            # Get data for this cluster
            cluster_df = df[df['cluster'] == cluster_id]
            
            # Skip small clusters
            if len(cluster_df) < 5:
                self.logger.warning(f"Skipping small cluster {cluster_id} with only {len(cluster_df)} videos")
                continue
            
            # Create a fresh TF-IDF vectorizer for this cluster
            vectorizer = TfidfVectorizer(
                max_features=500,
                min_df=2,  # Term must appear in at least 2 documents
                stop_words='english',
                ngram_range=(1, 1)  # Single words only
            )
            
            # Fit and transform on just this cluster's data
            try:
                X = vectorizer.fit_transform(cluster_df[text_col].fillna(''))
                
                # Get feature names
                feature_names = vectorizer.get_feature_names_out()
                
                # Get TF-IDF scores (sum across all documents)
                tfidf_sums = X.sum(axis=0).A1
                
                # Create (term, score) pairs
                term_scores = [(feature_names[i], float(tfidf_sums[i])) for i in range(len(tfidf_sums))]
                
                # Sort by score
                term_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Take top terms 
                top_terms = term_scores[:n_terms]
                
                cluster_terms[str(cluster_id)] = top_terms
                
            except Exception as e:
                self.logger.error(f"Error processing cluster {cluster_id}: {str(e)}")
                continue
        
        return cluster_terms
    
    def extract_cluster_stats(self, df: pd.DataFrame, cluster_col: str = 'cluster') -> Dict[str, Dict[str, Any]]:
        """
        Extract statistics for each cluster
        
        Args:
            df: DataFrame with cluster assignments
            cluster_col: Column containing cluster assignments
            
        Returns:
            Dictionary with cluster statistics
        """
        # Get unique clusters
        clusters = df[cluster_col].unique()
        stats = {}
        
        for cluster_id in clusters:
            cluster_df = df[df[cluster_col] == cluster_id]
            
            # Basic stats
            stats[str(cluster_id)] = {
                'size': len(cluster_df),
                'channels': cluster_df['channel'].nunique() if 'channel' in cluster_df.columns else 0,
                'top_channels': cluster_df['channel'].value_counts().head(5).to_dict() if 'channel' in cluster_df.columns else {}
            }
            
            # Add influence stats if available
            if 'influence' in cluster_df.columns:
                stats[str(cluster_id)].update({
                    'avg_influence': float(cluster_df['influence'].mean()),
                    'max_influence': float(cluster_df['influence'].max()),
                    'min_influence': float(cluster_df['influence'].min())
                })
            
            # Add view stats if available
            if 'views' in cluster_df.columns:
                stats[str(cluster_id)].update({
                    'total_views': int(cluster_df['views'].sum()),
                    'avg_views': float(cluster_df['views'].mean())
                })
        
        return stats
    
    def analyze_temporal_trends(self, df: pd.DataFrame, date_col: str = 'publish_date',
                              text_col: str = 'clean_title', time_unit: str = 'month') -> Dict[str, Any]:
        """
        Analyze how themes change over time
        
        Args:
            df: DataFrame with video data
            date_col: Column containing publication dates
            text_col: Column containing preprocessed text
            time_unit: Time unit for grouping (day, month, year)
            
        Returns:
            Dictionary with temporal trend analysis
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        if date_col not in df.columns:
            self.logger.warning(f"Column '{date_col}' not found. Cannot analyze temporal trends.")
            return {}
            
        # Convert to datetime and group by time unit
        df['date'] = pd.to_datetime(df[date_col], errors='coerce')
        
        if time_unit == 'day':
            df['time_period'] = df['date'].dt.date
        elif time_unit == 'month':
            df['time_period'] = df['date'].dt.to_period('M').astype(str)
        elif time_unit == 'year':
            df['time_period'] = df['date'].dt.year
        else:
            self.logger.warning(f"Unsupported time unit: {time_unit}")
            return {}
        
        # Group by time period
        period_groups = df.groupby('time_period')
        periods = sorted(df['time_period'].unique())
        
        # Analyze each time period
        period_themes = {}
        
        for period in periods:
            period_df = period_groups.get_group(period)
            
            if len(period_df) < 10:
                continue
                
            # Extract themes using TF-IDF
            vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
            X = vectorizer.fit_transform(period_df[text_col].fillna(''))
            
            # Get top terms
            feature_names = vectorizer.get_feature_names_out()
            tfidf_sums = X.sum(axis=0).A1
            
            # Sort by importance
            sorted_indices = tfidf_sums.argsort()[::-1]
            top_terms = [(feature_names[i], float(tfidf_sums[i])) for i in sorted_indices[:20]]
            
            period_themes[str(period)] = {
                'count': len(period_df),
                'top_terms': top_terms
            }
        
        return period_themes
    
    def analyze_influence_factors(self, df: pd.DataFrame, text_col: str = 'clean_title',
                              influence_col: str = 'influence',
                              threshold_percentile: int = 75) -> Dict[str, Any]:
        """
        Analyze factors associated with high influence
        
        Args:
            df: DataFrame with video data
            text_col: Column containing preprocessed text
            influence_col: Column containing influence scores
            threshold_percentile: Percentile threshold for high influence
            
        Returns:
            Dictionary with influence analysis results
        """
        if influence_col not in df.columns:
            self.logger.warning(f"Column '{influence_col}' not found. Cannot analyze influence factors.")
            return {}
        
        # Ensure the column is numeric
        df[influence_col] = pd.to_numeric(df[influence_col], errors='coerce')
        
        # Filter out NaN values
        valid_df = df.dropna(subset=[influence_col])
        
        # Check if we have valid data
        if len(valid_df) == 0:
            self.logger.warning("No valid numeric data in influence column")
            return {
                'error': "No valid numeric data for influence analysis"
            }
        
        # Determine threshold for "high influence"
        threshold = np.percentile(valid_df[influence_col], threshold_percentile)
        
        # Split into high and normal influence groups
        high_df = valid_df[valid_df[influence_col] >= threshold]
        normal_df = valid_df[valid_df[influence_col] < threshold]
        
        # Basic result structure with counts
        result = {
            'threshold': float(threshold),
            'high_count': int(len(high_df)),
            'normal_count': int(len(normal_df))
        }
        
        # Skip detailed analysis if there's not enough data
        if len(high_df) < 5 or len(normal_df) < 5:
            self.logger.warning(f"Insufficient data for detailed influence analysis")
            result['error'] = "Insufficient data for detailed analysis"
            return result
        
        # Only proceed with TF-IDF analysis if we have enough data and text content
        if text_col in df.columns and len(high_df[text_col].dropna()) > 0 and len(normal_df[text_col].dropna()) > 0:
            try:
                # Compare term usage between groups using TF-IDF
                from sklearn.feature_extraction.text import TfidfVectorizer
                
                vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
                
                # Fit on all text to get consistent feature space
                vectorizer.fit(valid_df[text_col].fillna(''))
                
                # Transform both groups
                high_X = vectorizer.transform(high_df[text_col].fillna(''))
                normal_X = vectorizer.transform(normal_df[text_col].fillna(''))
                
                if high_X.shape[0] > 0 and normal_X.shape[0] > 0:
                    # Calculate average TF-IDF for each term in both groups
                    high_avg = high_X.mean(axis=0).A1
                    normal_avg = normal_X.mean(axis=0).A1
                    
                    # Calculate difference in term usage
                    term_diff = high_avg - normal_avg
                    
                    # Get feature names
                    feature_names = vectorizer.get_feature_names_out()
                    
                    # Find terms more common in high influence videos
                    high_indices = term_diff.argsort()[::-1][:50]
                    high_terms = [(feature_names[i], float(term_diff[i])) for i in high_indices]
                    
                    # Find terms more common in normal influence videos
                    normal_indices = term_diff.argsort()[:50]
                    normal_terms = [(feature_names[i], float(term_diff[i])) for i in normal_indices]
                    
                    # Add to results
                    result['high_terms'] = high_terms
                    result['normal_terms'] = normal_terms
            except Exception as e:
                self.logger.error(f"Error during TF-IDF analysis: {str(e)}")
                result['tfidf_error'] = str(e)
        
        # Analyze channel influence if channel column exists
        if 'channel' in df.columns:
            try:
                high_channels = high_df['channel'].value_counts().head(20).to_dict()
                channel_influence = {}
                
                for channel, count in high_channels.items():
                    channel_df = df[df['channel'] == channel]
                    avg_influence = float(channel_df[influence_col].mean())
                    channel_influence[channel] = {
                        'count': int(count),
                        'avg_influence': avg_influence
                    }
                
                result['channel_influence'] = channel_influence
            except Exception as e:
                self.logger.error(f"Error in channel analysis: {str(e)}")
        
        return result