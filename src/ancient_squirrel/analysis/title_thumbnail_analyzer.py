# src/ancient_squirrel/analysis/title_thumbnail_analyzer.py

import os
import pandas as pd
import re
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import json
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import defaultdict

from ..core.base_processor import BaseProcessor

class TitleThumbnailAnalyzer(BaseProcessor):
    """
    Analyze title-thumbnail pairs to identify patterns and clusters within communities
    """
    
    def __init__(self, config: Dict[str, Any], num_workers: Optional[int] = None,
                logger: Optional[logging.Logger] = None):
        """
        Initialize the title-thumbnail analyzer
        
        Args:
            config: Configuration dictionary
            num_workers: Number of worker processes
            logger: Logger instance
        """
        super().__init__(num_workers, logger)
        self.config = config
        
        # Initialize LLM adapter for joint analysis
        self.use_llm = config.get("use_llm", False)
        self.llm_adapter = None
        
        if self.use_llm:
            try:
                from ..utils.llm_adapter import LLMAdapter
                
                openai_key = config.get("openai_api_key")
                model = config.get("llm_model", "gpt-4.1-vision")
                
                if openai_key:
                    self.llm_adapter = LLMAdapter(
                        provider="openai",
                        api_key=openai_key,
                        model=model
                    )
                else:
                    self.logger.warning("OpenAI API key not provided, LLM joint analysis disabled")
            except ImportError:
                self.logger.warning("LLMAdapter not available, disabling LLM joint analysis")
        
        # Set up cache directory for joint analysis results
        self.cache_dir = config.get("joint_analysis_cache_dir", "joint_analysis_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
    
    # Add the same conversion function to TitleThumbnailAnalyzer class in src/ancient_squirrel/analysis/title_thumbnail_analyzer.py

    def _convert_to_json_serializable(self, obj):
        """
        Convert numpy types to Python native types to ensure JSON serializability
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON serializable object
        """
        import numpy as np
        
        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, np.bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return self._convert_to_json_serializable(obj.tolist())
        else:
            return obj
    
    def _get_cache_path(self, video_id: str) -> str:
        """
        Get cache path for joint analysis results
        
        Args:
            video_id: Video ID
            
        Returns:
            Path to the cache file
        """
        import hashlib
        # Create a hash of the video ID to use as the cache filename
        hash_obj = hashlib.md5(video_id.encode())
        return os.path.join(self.cache_dir, f"{hash_obj.hexdigest()}.json")
    
    def _load_cached_analysis(self, video_id: str) -> Optional[Dict[str, Any]]:
        """
        Load cached joint analysis results if available
        
        Args:
            video_id: Video ID
            
        Returns:
            Cached analysis results or None if not available
        """
        cache_path = self._get_cache_path(video_id)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Error loading cached joint analysis: {e}")
        
        return None
    
    def _save_analysis_to_cache(self, video_id: str, analysis: Dict[str, Any]) -> None:
        """
        Save joint analysis results to cache
        
        Args:
            video_id: Video ID
            analysis: Analysis results to cache
        """
        cache_path = self._get_cache_path(video_id)
        try:
            with open(cache_path, 'w') as f:
                json.dump(analysis, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Error saving joint analysis to cache: {e}")
    
    def analyze_title_thumbnail_pair(self, title: str, thumbnail_path: str, 
                                    basic_analysis: Optional[Dict[str, Any]] = None,
                                    video_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze a title-thumbnail pair to identify patterns
        
        Args:
            title: Video title
            thumbnail_path: Path to the thumbnail image
            basic_analysis: Optional pre-computed basic image analysis
            video_id: Optional video ID for caching
            
        Returns:
            Dictionary with joint analysis results
        """
        # Check cache first if video_id is provided
        if video_id:
            cached_analysis = self._load_cached_analysis(video_id)
            if cached_analysis:
                return cached_analysis
        
        # Initialize result dictionary
        result = {
            "title": title,
            "thumbnail_path": thumbnail_path,
            "title_length": len(title) if title else 0,
            "patterns": [],
            "text_visual_alignment": 0.0,
            "clickbait_score": 0.0
        }
        
        # Pattern detection
        patterns = []
        
        # Check if title starts with common patterns
        title_lower = title.lower() if title else ""
        
        if title_lower.startswith(("how to", "how i", "how we")):
            patterns.append("how_to")
        
        if title_lower.startswith(("why ", "what if", "what is", "when ")):
            patterns.append("question")
        
        if any(marker in title_lower for marker in ["vs", "versus", "compared to"]):
            patterns.append("comparison")
        
        if any(marker in title_lower for marker in ["best", "top", "ultimate", "perfect"]):
            patterns.append("superlative")
        
        # Check for numbered lists
        if re.search(r"^\d+\s+", title_lower) or re.search(r"\b\d+\s+(?:things|ways|tips|tricks|hacks|reasons|steps)\b", title_lower):
            patterns.append("numbered_list")
        
        # Look for emotional triggers
        if any(word in title_lower for word in ["shocking", "amazing", "incredible", "unbelievable", "surprising"]):
            patterns.append("emotional_trigger")
        
        # Check for timestamps/chapters
        if re.search(r"\(\d+:\d+\)|\[\d+:\d+\]", title):
            patterns.append("timestamps")
        
        # Check for thumbnail patterns using basic_analysis if available
        if basic_analysis:
            # Text in thumbnail
            if basic_analysis.get("composition", {}).get("has_text", False):
                patterns.append("thumbnail_text")
            
            # Faces in thumbnail
            if basic_analysis.get("composition", {}).get("has_faces", False):
                patterns.append("thumbnail_faces")
            
            # Bright colors
            colors = basic_analysis.get("colors", {}).get("dominant", [])
            bright_colors = ["yellow", "orange", "red", "pink", "cyan"]
            if any(color.get("name") in bright_colors for color in colors):
                patterns.append("bright_colors")
            
            # High contrast
            if basic_analysis.get("colors", {}).get("contrast", 0) > 0.5:
                patterns.append("high_contrast")
        
        # Add patterns to result
        result["patterns"] = patterns
        
        # Calculate approximate text-visual alignment
        title_keywords = set(re.findall(r'\b\w{3,}\b', title_lower))
        
        # Use title keywords to estimate alignment without LLM
        if basic_analysis:
            # Simple heuristic scoring
            alignment_score = 0.0
            
            # Faces in thumbnail with personal pronouns in title
            if basic_analysis.get("composition", {}).get("has_faces", False):
                if any(word in title_lower for word in ["i", "me", "my", "we", "our", "you", "your"]):
                    alignment_score += 0.3
            
            # Text in thumbnail likely reinforces title
            if basic_analysis.get("composition", {}).get("has_text", False):
                alignment_score += 0.3
            
            # Bright colors and emotional language
            colors = basic_analysis.get("colors", {}).get("dominant", [])
            bright_colors = ["yellow", "orange", "red", "pink", "cyan"]
            if any(color.get("name") in bright_colors for color in colors):
                if any(word in title_lower for word in ["shocking", "amazing", "incredible", "unbelievable", "surprising"]):
                    alignment_score += 0.2
            
            # Adjust for brightness
            brightness = basic_analysis.get("colors", {}).get("brightness", 0.5)
            if brightness > 0.6:
                alignment_score += 0.1
            
            # Normalize to 0-1
            alignment_score = min(1.0, alignment_score)
            result["text_visual_alignment"] = alignment_score
        
        # Calculate clickbait score based on title and thumbnail patterns
        clickbait_patterns = {
            "emotional_trigger": 0.3,
            "superlative": 0.2,
            "thumbnail_faces": 0.15,
            "thumbnail_text": 0.15,
            "bright_colors": 0.1,
            "high_contrast": 0.1
        }
        
        # Calculate score based on detected patterns
        clickbait_score = sum(clickbait_patterns.get(pattern, 0) for pattern in patterns)
        
        # Check for specific title phrases that indicate clickbait
        clickbait_phrases = ["you won't believe", "mind blowing", "will shock you", "gone wrong", 
                           "never seen before", "secret", "leaked", "exposed", "don't tell"]
        
        if any(phrase in title_lower for phrase in clickbait_phrases):
            clickbait_score += 0.3
        
        # Normalize to 0-1
        clickbait_score = min(1.0, clickbait_score)
        result["clickbait_score"] = clickbait_score
        
        # Save to cache if video_id is provided
        if video_id:
            self._save_analysis_to_cache(video_id, result)
        
        return result
    
    def analyze_pair_with_llm(self, title: str, thumbnail_path: str, video_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Use LLM to analyze title-thumbnail pair
        
        Args:
            title: Video title
            thumbnail_path: Path to the thumbnail image
            video_id: Optional video ID for caching
            
        Returns:
            Dictionary with LLM analysis results
        """
        if not self.llm_adapter:
            return {"error": "LLM adapter not available"}
        
        # Check cache first if video_id is provided
        if video_id:
            cached_analysis = self._load_cached_analysis(video_id)
            if cached_analysis:
                return cached_analysis
        
        try:
            # Convert image to base64
            from ..analysis.image_analyzer import ImageAnalyzer
            image_analyzer = ImageAnalyzer(self.config)
            image_base64 = image_analyzer._image_to_base64(thumbnail_path)
            
            if not image_base64:
                return {"error": "Failed to encode image"}
            
            # Create prompt for joint analysis
            prompt = f"""
            Analyze this YouTube video thumbnail image alongside its title:
            
            TITLE: "{title}"
            
            Please provide a detailed analysis of:
            
            1. How the thumbnail and title work together as a unified content unit
            2. What information or emotional content exists in each
            3. The psychological triggers employed across both elements
            4. The audience targeting signals present
            5. How they distribute information between visual and textual elements
            
            Classify this pair into a content strategy category and explain why.
            
            Format your response as a structured JSON object with these categories:
            - content_strategy: the overall strategy type
            - promise_type: what the content promises to deliver
            - engagement_hooks: psychological triggers used
            - target_audience: who this content targets
            - visual_reinforcement: how the visual elements support the title
            - conversion_strategy: how it aims to get viewers to click
            - effectiveness_score: 1-10 rating of how effective this pair is
            
            Be specific and objective in your analysis, focusing on observable patterns.
            """
            
            # Create message for vision model
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert in analyzing YouTube content strategy. Provide detailed, accurate observations in a structured JSON format."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ]
            
            # Generate completion
            response = self.llm_adapter.client.chat.completions.create(
                model=self.llm_adapter.model,
                messages=messages,
                response_format={"type": "json_object"},
                max_tokens=1500
            )
            
            # Parse response content
            content = response.choices[0].message.content
            analysis = json.loads(content)
            
            # Save to cache if video_id is provided
            if video_id:
                self._save_analysis_to_cache(video_id, analysis)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing with LLM: {e}")
            return {"error": str(e)}
    
    def identify_subclusters(self, df: pd.DataFrame, community_col: str, 
                           joint_analysis_col: str, n_subclusters: int = 3) -> pd.DataFrame:
        """
        Identify subclusters within communities based on title-thumbnail patterns
        
        Args:
            df: DataFrame with joint analysis results
            community_col: Column containing community IDs
            joint_analysis_col: Column containing joint analysis results
            n_subclusters: Number of subclusters to create per community
            
        Returns:
            DataFrame with subcluster assignments
        """
        try:
            # Create a copy of the input DataFrame
            result_df = df.copy()
            
            # Add subcluster column
            subcluster_col = f"{community_col}_subcluster"
            result_df[subcluster_col] = pd.NA
            
            # Get unique communities
            communities = df[community_col].unique()
            
            for community_id in tqdm(communities, desc="Identifying subclusters"):
                # Filter for this community
                community_df = result_df[result_df[community_col] == community_id]
                
                # Skip if too few videos
                if len(community_df) < n_subclusters * 2:
                    continue
                
                # Extract features from joint analysis
                features = []
                indices = []
                
                for idx, row in community_df.iterrows():
                    if pd.isna(row[joint_analysis_col]):
                        continue
                    
                    try:
                        analysis = json.loads(row[joint_analysis_col])
                        
                        # Construct feature vector
                        feature_vector = [
                            analysis.get("title_length", 0) / 100,  # Normalize title length
                            analysis.get("text_visual_alignment", 0),
                            analysis.get("clickbait_score", 0)
                        ]
                        
                        # Add pattern features (one-hot encoding)
                        all_patterns = ["how_to", "question", "comparison", "superlative", 
                                      "numbered_list", "emotional_trigger", "timestamps", 
                                      "thumbnail_text", "thumbnail_faces", "bright_colors", 
                                      "high_contrast"]
                        
                        patterns = analysis.get("patterns", [])
                        for pattern in all_patterns:
                            feature_vector.append(1.0 if pattern in patterns else 0.0)
                        
                        features.append(feature_vector)
                        indices.append(idx)
                    
                    except Exception as e:
                        self.logger.warning(f"Error extracting features for row {idx}: {e}")
                
                if not features:
                    continue
                
                # Convert to numpy array
                features_array = np.array(features)
                
                # Perform K-means clustering
                n_actual_clusters = min(n_subclusters, len(features))
                kmeans = KMeans(n_clusters=n_actual_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(features_array)
                
                # Update DataFrame with subcluster assignments
                for i, idx in enumerate(indices):
                    result_df.at[idx, subcluster_col] = f"{community_id}_{cluster_labels[i]}"
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error identifying subclusters: {e}")
            return df
    
    def process(self, df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Process data and analyze title-thumbnail pairs
        
        Args:
            df: Input DataFrame with video data, titles, and thumbnail paths
            **kwargs: Additional options
            
        Returns:
            Tuple of (enhanced DataFrame, analysis results dict)
        """
        # Get options from kwargs or config
        video_id_col = kwargs.get("video_id_col", self.config.get("video_id_col", "video_id"))
        title_col = kwargs.get("title_col", self.config.get("title_col", "title"))
        thumbnail_col = kwargs.get("thumbnail_col", self.config.get("thumbnail_col", "thumbnail_path"))
        thumbnail_analysis_col = kwargs.get("thumbnail_analysis_col", "thumbnail_analysis")
        community_col = kwargs.get("community_col", self.config.get("community_col", "community"))
        influence_col = kwargs.get("influence_col", self.config.get("influence_col", "influence"))
        use_llm = kwargs.get("use_llm", self.config.get("use_llm", False))
        batch_size = kwargs.get("batch_size", self.config.get("batch_size", 50))
        n_subclusters = kwargs.get("n_subclusters", self.config.get("n_subclusters", 3))
        
        # Validate input DataFrame
        required_cols = [title_col, thumbnail_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"DataFrame missing required columns: {missing_cols}")
        
        # Create a copy of the input DataFrame
        result_df = df.copy()
        
        # Create output column names
        joint_analysis_col = "title_thumbnail_analysis"
        llm_joint_analysis_col = "title_thumbnail_llm_analysis"
        
        # Initialize analysis columns
        result_df[joint_analysis_col] = pd.NA
        if use_llm:
            result_df[llm_joint_analysis_col] = pd.NA
        
        # Filter out records without titles or thumbnail paths
        valid_df = result_df[
            ~result_df[title_col].isna() & 
            ~result_df[thumbnail_col].isna()
        ]
        
        # Sort by influence if column exists, to prioritize high-influence videos
        if influence_col in valid_df.columns:
            valid_df = valid_df.sort_values(influence_col, ascending=False)
        
        self.logger.info(f"Analyzing {len(valid_df)} title-thumbnail pairs")
        
        # Process pairs in batches
        analysis_results = {
            "total": len(valid_df),
            "successful": 0,
            "failed": 0,
            "joint_analysis_col": joint_analysis_col
        }
        
        if use_llm:
            analysis_results["llm_joint_analysis_col"] = llm_joint_analysis_col
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            for i in range(0, len(valid_df), batch_size):
                batch_df = valid_df.iloc[i:i + batch_size]
                self.logger.info(f"Processing batch {i//batch_size + 1}/{(len(valid_df)-1)//batch_size + 1} ({len(batch_df)} pairs)")
                
                # Create list of futures for basic joint analysis
                basic_futures = {}
                for idx, row in batch_df.iterrows():
                    title = row[title_col]
                    thumbnail_path = row[thumbnail_col]
                    
                    # Skip if invalid
                    if pd.isna(title) or pd.isna(thumbnail_path) or not os.path.exists(thumbnail_path):
                        continue
                    
                    # Get video ID if available
                    video_id = row.get(video_id_col, None)
                    
                    # Get basic thumbnail analysis if available
                    basic_analysis = None
                    if thumbnail_analysis_col in row and not pd.isna(row[thumbnail_analysis_col]):
                        try:
                            basic_analysis = json.loads(row[thumbnail_analysis_col])
                        except Exception as e:
                            self.logger.debug(f"Error parsing thumbnail analysis: {str(e)}")
                    
                    # Submit analysis task
                    future = executor.submit(
                        self.analyze_title_thumbnail_pair, 
                        title, 
                        thumbnail_path, 
                        basic_analysis, 
                        video_id
                    )
                    basic_futures[future] = idx
                
                # Process basic joint analysis results
                for future in tqdm(as_completed(basic_futures), total=len(basic_futures), desc="Joint analysis"):
                    idx = basic_futures[future]
                    
                    try:
                        result = future.result()
                        
                        if "error" not in result:
                            # Convert NumPy types to Python native types
                            serializable_result = self._numpy_to_python(result)
                            
                            # Update DataFrame with analysis
                            result_df.at[idx, joint_analysis_col] = json.dumps(serializable_result)
                            analysis_results["successful"] += 1
                        else:
                            analysis_results["failed"] += 1
                    except Exception as e:
                        self.logger.error(f"Error processing joint analysis result: {str(e)}")
                        import traceback
                        self.logger.debug(f"Traceback: {traceback.format_exc()}")
                        analysis_results["failed"] += 1
                
                # Process LLM joint analysis if enabled
                if use_llm and self.llm_adapter:
                    llm_futures = {}
                    for idx, row in batch_df.iterrows():
                        title = row[title_col]
                        thumbnail_path = row[thumbnail_col]
                        
                        # Skip if invalid
                        if pd.isna(title) or pd.isna(thumbnail_path) or not os.path.exists(thumbnail_path):
                            continue
                        
                        # Get video ID if available
                        video_id = row.get(video_id_col, None)
                        
                        # Submit analysis task
                        future = executor.submit(
                            self.analyze_pair_with_llm, 
                            title, 
                            thumbnail_path,
                            video_id
                        )
                        llm_futures[future] = idx
                    
                    # Process LLM joint analysis results
                    for future in tqdm(as_completed(llm_futures), total=len(llm_futures), desc="LLM joint analysis"):
                        idx = llm_futures[future]
                        
                        try:
                            result = future.result()
                            
                            # Convert NumPy types to Python native types
                            serializable_result = self._numpy_to_python(result)
                            
                            # Update DataFrame with LLM analysis
                            result_df.at[idx, llm_joint_analysis_col] = json.dumps(serializable_result)
                            
                        except Exception as e:
                            self.logger.error(f"Error processing LLM joint analysis result: {str(e)}")
                            import traceback
                            self.logger.debug(f"Traceback: {traceback.format_exc()}")
        
        self.logger.info(f"Title-thumbnail analysis complete: {analysis_results['successful']} successful, "
                    f"{analysis_results['failed']} failed")
        
        # Identify subclusters if community column exists
        if community_col in result_df.columns:
            try:
                self.logger.info(f"Identifying subclusters within {community_col} communities")
                
                result_df = self.identify_subclusters(
                    result_df, 
                    community_col, 
                    joint_analysis_col, 
                    n_subclusters
                )
                
                # Count subclusters
                subcluster_col = f"{community_col}_subcluster"
                if subcluster_col in result_df.columns:
                    analysis_results["subclusters"] = {
                        "count": result_df[subcluster_col].nunique(),
                        "column": subcluster_col
                    }
            except Exception as e:
                self.logger.error(f"Error identifying subclusters: {str(e)}")
                import traceback
                self.logger.debug(f"Traceback: {traceback.format_exc()}")
        
        # Extract pattern statistics
        try:
            pattern_stats = self._extract_pattern_statistics(result_df, joint_analysis_col)
            analysis_results["pattern_statistics"] = pattern_stats
        except Exception as e:
            self.logger.error(f"Error extracting pattern statistics: {str(e)}")
            analysis_results["pattern_statistics"] = {"error": str(e)}
        
        # Extract high-influence patterns if influence column exists
        if influence_col in result_df.columns:
            try:
                influence_patterns = self._extract_influence_patterns(
                    result_df, 
                    joint_analysis_col, 
                    influence_col
                )
                analysis_results["influence_patterns"] = influence_patterns
            except Exception as e:
                self.logger.error(f"Error extracting influence patterns: {str(e)}")
                analysis_results["influence_patterns"] = {"error": str(e)}
        
        return result_df, {
            "title_thumbnail_analysis": analysis_results
        }

    def _numpy_to_python(self, obj):
        """
        Convert NumPy types to Python native types for JSON serialization
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON serializable object
        """
        import numpy as np
        
        if isinstance(obj, dict):
            return {k: self._numpy_to_python(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._numpy_to_python(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._numpy_to_python(item) for item in obj)
        elif isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):  # Only use np.bool_ instead of np.bool
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return self._numpy_to_python(obj.tolist())
        elif obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        else:
            # For any other types, convert to string (last resort)
            try:
                return str(obj)
            except:
                return None
    
    def _extract_pattern_statistics(self, df: pd.DataFrame, analysis_col: str) -> Dict[str, Any]:
        """
        Extract statistics about title-thumbnail patterns
        
        Args:
            df: DataFrame with joint analysis results
            analysis_col: Column containing joint analysis
            
        Returns:
            Dictionary with pattern statistics
        """
        stats = {
            "patterns": {},
            "clickbait_score": {
                "mean": 0.0,
                "median": 0.0
            },
            "text_visual_alignment": {
                "mean": 0.0,
                "median": 0.0
            }
        }
        
        # Continuing the _extract_pattern_statistics method in title_thumbnail_analyzer.py

        try:
            # Filter to valid rows
            valid_df = df[~df[analysis_col].isna()]
            
            if len(valid_df) == 0:
                return stats
            
            # Parse JSON strings
            parsed_analyses = []
            for analysis_json in valid_df[analysis_col]:
                try:
                    analysis = json.loads(analysis_json)
                    parsed_analyses.append(analysis)
                except:
                    continue
            
            if not parsed_analyses:
                return stats
            
            # Count pattern occurrences
            pattern_counts = defaultdict(int)
            clickbait_scores = []
            alignment_scores = []
            
            for analysis in parsed_analyses:
                # Add patterns
                for pattern in analysis.get("patterns", []):
                    pattern_counts[pattern] += 1
                
                # Add scores
                clickbait_scores.append(analysis.get("clickbait_score", 0.0))
                alignment_scores.append(analysis.get("text_visual_alignment", 0.0))
            
            # Calculate pattern percentages
            total = len(parsed_analyses)
            pattern_stats = {
                pattern: {
                    "count": count,
                    "percentage": (count / total) * 100
                }
                for pattern, count in pattern_counts.items()
            }
            stats["patterns"] = pattern_stats
            
            # Calculate score statistics
            if clickbait_scores:
                stats["clickbait_score"] = {
                    "mean": np.mean(clickbait_scores),
                    "median": np.median(clickbait_scores),
                    "high_percentage": sum(1 for score in clickbait_scores if score > 0.6) / total * 100
                }
            
            if alignment_scores:
                stats["text_visual_alignment"] = {
                    "mean": np.mean(alignment_scores),
                    "median": np.median(alignment_scores),
                    "high_percentage": sum(1 for score in alignment_scores if score > 0.7) / total * 100
                }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error extracting pattern statistics: {e}")
            return stats
    
    def _extract_influence_patterns(self, df: pd.DataFrame, analysis_col: str, 
                                  influence_col: str) -> Dict[str, Any]:
        """
        Extract patterns associated with high influence videos
        
        Args:
            df: DataFrame with joint analysis results
            analysis_col: Column containing joint analysis
            influence_col: Column containing influence scores
            
        Returns:
            Dictionary with influence pattern analysis
        """
        result = {
            "high_influence_patterns": {},
            "low_influence_patterns": {},
            "correlation": {}
        }
        
        try:
            # Ensure influence column is numeric
            df[influence_col] = pd.to_numeric(df[influence_col], errors='coerce')
            
            # Filter to valid rows
            valid_df = df[(~df[analysis_col].isna()) & (~df[influence_col].isna())]
            
            if len(valid_df) == 0:
                return result
            
            # Calculate influence threshold (75th percentile)
            threshold = np.percentile(valid_df[influence_col], 75)
            
            # Split into high and low influence groups
            high_df = valid_df[valid_df[influence_col] >= threshold]
            low_df = valid_df[valid_df[influence_col] < threshold]
            
            # Parse JSON strings
            def parse_analyses(dataframe):
                analyses = []
                for analysis_json in dataframe[analysis_col]:
                    try:
                        analyses.append(json.loads(analysis_json))
                    except:
                        pass
                return analyses
            
            high_analyses = parse_analyses(high_df)
            low_analyses = parse_analyses(low_df)
            
            if not high_analyses or not low_analyses:
                return result
            
            # Extract patterns
            def count_patterns(analyses):
                counts = defaultdict(int)
                for analysis in analyses:
                    for pattern in analysis.get("patterns", []):
                        counts[pattern] += 1
                return counts
            
            high_pattern_counts = count_patterns(high_analyses)
            low_pattern_counts = count_patterns(low_analyses)
            
            # Calculate percentages
            high_total = len(high_analyses)
            low_total = len(low_analyses)
            
            high_patterns = {
                pattern: {
                    "count": count,
                    "percentage": (count / high_total) * 100
                }
                for pattern, count in high_pattern_counts.items()
            }
            
            low_patterns = {
                pattern: {
                    "count": count,
                    "percentage": (count / low_total) * 100
                }
                for pattern, count in low_pattern_counts.items()
            }
            
            # Find patterns more common in high-influence videos
            pattern_diffs = {}
            all_patterns = set(high_pattern_counts.keys()) | set(low_pattern_counts.keys())
            
            for pattern in all_patterns:
                high_pct = high_patterns.get(pattern, {}).get("percentage", 0)
                low_pct = low_patterns.get(pattern, {}).get("percentage", 0)
                diff = high_pct - low_pct
                pattern_diffs[pattern] = diff
            
            # Sort by difference (descending)
            sorted_diffs = sorted(pattern_diffs.items(), key=lambda x: x[1], reverse=True)
            
            # Get top patterns for each group
            result["high_influence_patterns"] = {
                pattern: diff for pattern, diff in sorted_diffs[:5] if diff > 0
            }
            
            result["low_influence_patterns"] = {
                pattern: abs(diff) for pattern, diff in sorted_diffs[-5:] if diff < 0
            }
            
            # Analyze correlation between influence and metrics
            influence_values = valid_df[influence_col].tolist()
            clickbait_scores = []
            alignment_scores = []
            
            for analysis_json in valid_df[analysis_col]:
                try:
                    analysis = json.loads(analysis_json)
                    clickbait_scores.append(analysis.get("clickbait_score", 0.0))
                    alignment_scores.append(analysis.get("text_visual_alignment", 0.0))
                except:
                    clickbait_scores.append(0.0)
                    alignment_scores.append(0.0)
            
            # Calculate correlation
            if len(influence_values) > 10:
                try:
                    clickbait_corr = np.corrcoef(influence_values, clickbait_scores)[0, 1]
                    alignment_corr = np.corrcoef(influence_values, alignment_scores)[0, 1]
                    
                    result["correlation"] = {
                        "clickbait_score": clickbait_corr,
                        "text_visual_alignment": alignment_corr
                    }
                except:
                    pass
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error extracting influence patterns: {e}")
            return result