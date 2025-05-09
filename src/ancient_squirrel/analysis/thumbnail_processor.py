# src/ancient_squirrel/analysis/thumbnail_processor.py

import os
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path
import time
import hashlib

from ..core.base_processor import BaseProcessor


class ThumbnailProcessor(BaseProcessor):
    """Processor for fetching and managing YouTube video thumbnails"""
    
    THUMBNAIL_QUALITY_LEVELS = {
        "max": "maxresdefault.jpg",
        "high": "hqdefault.jpg",
        "medium": "mqdefault.jpg",
        "standard": "sddefault.jpg",
        "default": "default.jpg"
    }
    
    def __init__(self, config: Dict[str, Any], num_workers: Optional[int] = None,
                logger: Optional[logging.Logger] = None):
        """
        Initialize the thumbnail processor
        
        Args:
            config: Configuration dictionary
            num_workers: Number of worker processes
            logger: Logger instance
        """
        super().__init__(num_workers, logger)
        self.config = config
        
        # Set default values if not in config
        self.thumbnail_dir = config.get("thumbnail_dir", "thumbnails")
        self.quality = config.get("thumbnail_quality", "max")
        self.max_retries = config.get("max_retries", 3)
        self.request_timeout = config.get("request_timeout", 10)
        self.rate_limit = config.get("rate_limit", 0.2)  # Default: 5 requests per second
        
        # Create thumbnails directory
        os.makedirs(self.thumbnail_dir, exist_ok=True)
        
        # Initialize cache
        self.cache_file = os.path.join(self.thumbnail_dir, "cache.json")
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict[str, Any]:
        """Load the download cache if it exists"""
        if os.path.exists(self.cache_file):
            try:
                import json
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Error loading cache: {e}. Creating new cache.")
        
        return {
            "downloaded": {},
            "failed": {},
            "last_updated": time.time()
        }
    
    def _save_cache(self) -> None:
        """Save the download cache"""
        try:
            import json
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f)
        except Exception as e:
            self.logger.error(f"Error saving cache: {e}")
    
    def _get_thumbnail_url(self, video_id: str, quality: Optional[str] = None) -> str:
        """
        Get the URL for a YouTube thumbnail
        
        Args:
            video_id: YouTube video ID
            quality: Thumbnail quality level (max, high, medium, standard, default)
            
        Returns:
            Thumbnail URL
        """
        quality = quality or self.quality
        if quality not in self.THUMBNAIL_QUALITY_LEVELS:
            self.logger.warning(f"Invalid quality '{quality}'. Using 'max' instead.")
            quality = "max"
            
        return f"http://img.youtube.com/vi/{video_id}/{self.THUMBNAIL_QUALITY_LEVELS[quality]}"
    
    def _get_thumbnail_path(self, video_id: str, quality: Optional[str] = None) -> str:
        """
        Get the local path for storing a thumbnail
        
        Args:
            video_id: YouTube video ID
            quality: Thumbnail quality level
            
        Returns:
            Local path for the thumbnail
        """
        quality = quality or self.quality
        return os.path.join(self.thumbnail_dir, f"{video_id}_{quality}.jpg")
    
    def download_thumbnail(self, video_id: str, quality: Optional[str] = None, 
                          force: bool = False) -> Tuple[bool, str]:
        """
        Download a single thumbnail
        
        Args:
            video_id: YouTube video ID
            quality: Thumbnail quality level
            force: Force download even if already cached
            
        Returns:
            Tuple of (success, path_or_error_message)
        """
        if not video_id or not isinstance(video_id, str):
            return False, "Invalid video ID"
            
        quality = quality or self.quality
        thumbnail_path = self._get_thumbnail_path(video_id, quality)
        
        # Check if already downloaded and not forcing re-download
        if not force and video_id in self.cache["downloaded"]:
            cached_path = self.cache["downloaded"][video_id].get("path")
            if cached_path and os.path.exists(cached_path):
                return True, cached_path
        
        # Check if previously failed and not forcing retry
        if not force and video_id in self.cache["failed"]:
            fail_count = self.cache["failed"][video_id].get("count", 0)
            if fail_count >= self.max_retries:
                return False, f"Maximum retry count ({self.max_retries}) reached"
        
        # Download the thumbnail
        url = self._get_thumbnail_url(video_id, quality)
        retries = 0
        
        while retries < self.max_retries:
            try:
                response = requests.get(url, timeout=self.request_timeout)
                
                if response.status_code == 200:
                    # Save the thumbnail
                    with open(thumbnail_path, 'wb') as f:
                        f.write(response.content)
                    
                    # Update cache
                    self.cache["downloaded"][video_id] = {
                        "path": thumbnail_path,
                        "quality": quality,
                        "timestamp": time.time()
                    }
                    
                    # Remove from failed if it was there
                    if video_id in self.cache["failed"]:
                        del self.cache["failed"][video_id]
                    
                    return True, thumbnail_path
                else:
                    # Try a lower quality if available
                    qualities = list(self.THUMBNAIL_QUALITY_LEVELS.keys())
                    current_idx = qualities.index(quality)
                    
                    if current_idx < len(qualities) - 1:
                        next_quality = qualities[current_idx + 1]
                        self.logger.info(f"Thumbnail not found at {quality} quality. Trying {next_quality}.")
                        return self.download_thumbnail(video_id, next_quality, force)
                    else:
                        error_msg = f"HTTP error: {response.status_code}"
                        
                        # Update failed cache
                        if video_id not in self.cache["failed"]:
                            self.cache["failed"][video_id] = {"count": 1, "error": error_msg}
                        else:
                            self.cache["failed"][video_id]["count"] += 1
                            self.cache["failed"][video_id]["error"] = error_msg
                        
                        return False, error_msg
            
            except requests.exceptions.RequestException as e:
                retries += 1
                if retries < self.max_retries:
                    self.logger.warning(f"Error downloading thumbnail for {video_id}: {e}. Retrying ({retries}/{self.max_retries}).")
                    time.sleep(1)  # Wait before retry
                else:
                    # Update failed cache
                    error_msg = str(e)
                    if video_id not in self.cache["failed"]:
                        self.cache["failed"][video_id] = {"count": 1, "error": error_msg}
                    else:
                        self.cache["failed"][video_id]["count"] += 1
                        self.cache["failed"][video_id]["error"] = error_msg
                    
                    return False, error_msg
    
    def process(self, df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Process data and download thumbnails
        
        Args:
            df: Input DataFrame with video data
            **kwargs: Additional options
            
        Returns:
            Tuple of (enhanced DataFrame, download results dict)
        """
        # Validate input DataFrame
        if 'video_id' not in df.columns:
            raise ValueError("DataFrame must contain a 'video_id' column")
        
        # Get options from kwargs or config
        video_id_col = kwargs.get("video_id_col", self.config.get("video_id_col", "video_id"))
        quality = kwargs.get("quality", self.config.get("thumbnail_quality", "max"))
        force_download = kwargs.get("force_download", self.config.get("force_download", False))
        batch_size = kwargs.get("batch_size", self.config.get("batch_size", 100))
        
        # Create thumbnail column name
        thumbnail_col = kwargs.get("thumbnail_col", "thumbnail_path")
        
        # Create a copy of the input DataFrame
        result_df = df.copy()
        
        # Initialize empty thumbnail path column
        result_df[thumbnail_col] = pd.NA
        
        # Get list of video IDs to process
        video_ids = df[video_id_col].dropna().unique().tolist()
        
        self.logger.info(f"Downloading thumbnails for {len(video_ids)} videos")
        
        # Download thumbnails in batches
        download_results = {
            "total": len(video_ids),
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "failures": {}
        }
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            for i in range(0, len(video_ids), batch_size):
                batch = video_ids[i:i + batch_size]
                self.logger.info(f"Processing batch {i//batch_size + 1}/{(len(video_ids)-1)//batch_size + 1} ({len(batch)} videos)")
                
                # Create list of futures
                futures = {}
                for video_id in batch:
                    # Skip if already in cache and not forcing re-download
                    if not force_download and video_id in self.cache["downloaded"]:
                        cached_path = self.cache["downloaded"][video_id].get("path")
                        if cached_path and os.path.exists(cached_path):
                            # Update DataFrame
                            result_df.loc[result_df[video_id_col] == video_id, thumbnail_col] = cached_path
                            download_results["skipped"] += 1
                            continue
                    
                    # Submit download task
                    future = executor.submit(self.download_thumbnail, video_id, quality, force_download)
                    futures[future] = video_id
                    
                    # Rate limiting
                    time.sleep(self.rate_limit)
                
                # Process results as they complete
                for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading thumbnails"):
                    video_id = futures[future]
                    success, result = future.result()
                    
                    if success:
                        # Update DataFrame with thumbnail path
                        result_df.loc[result_df[video_id_col] == video_id, thumbnail_col] = result
                        download_results["successful"] += 1
                    else:
                        download_results["failed"] += 1
                        download_results["failures"][video_id] = result
                
                # Save cache periodically
                self._save_cache()
        
        # Final cache update
        self._save_cache()
        
        self.logger.info(f"Thumbnail download complete: {download_results['successful']} successful, "
                       f"{download_results['failed']} failed, {download_results['skipped']} skipped")
        
        return result_df, {
            "thumbnail_download": download_results,
            "thumbnail_column": thumbnail_col,
            "thumbnail_dir": self.thumbnail_dir
        }