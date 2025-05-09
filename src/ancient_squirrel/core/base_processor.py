from abc import ABC, abstractmethod
import logging
import pandas as pd
import numpy as np
import os
import json
import multiprocessing
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import pickle
from datetime import datetime

from ..utils.data_utils import load_data, save_data

class BaseProcessor(ABC):
    """Base class for all data processors"""
    
    def __init__(self, num_workers: Optional[int] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize the base processor
        
        Args:
            num_workers: Number of worker processes to use (default: CPU count - 1)
            logger: Logger instance to use (default: class name)
        """
        self.num_workers = num_workers or max(1, multiprocessing.cpu_count() - 1)
        self.logger = logger or logging.getLogger(self.__class__.__name__)
    
    def load_data(self, file_path: str, format: str = "json") -> pd.DataFrame:
        """
        Load data from file
        
        Args:
            file_path: Path to the data file
            format: File format (auto, csv, parquet, json)
            
        Returns:
            Loaded DataFrame
        """
        self.logger.info(f"Loading data from {file_path}")
        return load_data(file_path, format)
    
    def save_data(self, df: pd.DataFrame, file_path: str, format: str = "json") -> None:
        """
        Save data to file
        
        Args:
            df: DataFrame to save
            file_path: Path to save the data
            format: File format (auto, csv, parquet, json)
        """
        self.logger.info(f"Saving data to {file_path}")
        save_data(df, file_path, format)
    
    def save_analysis(self, results: Dict[str, Any], output_file: str) -> None:
        """
        Save analysis results to file
        
        Args:
            results: Analysis results dictionary
            output_file: Path to output file
        """
        self.logger.info(f"Saving analysis results to {output_file}")
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

        # Convert any numpy scalar types to Python native types
        def _convert_types(obj):
            if isinstance(obj, dict):
                return {_convert_types(k): _convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_convert_types(v) for v in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        safe_results = _convert_types(results)

        # Save based on file extension
        ext = os.path.splitext(output_file)[1].lower()
        if ext == '.json':
            with open(output_file, 'w') as f:
                json.dump(safe_results, f, indent=2)
        elif ext in ['.pkl', '.pickle']:
            with open(output_file, 'wb') as f:
                pickle.dump(safe_results, f)
        else:
            self.logger.warning(f"Unsupported file format: {ext}. Defaulting to JSON.")
            with open(f"{output_file}.json", 'w') as f:
                json.dump(safe_results, f, indent=2)
    
    @abstractmethod
    def process(self, data: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Process data and return results
        
        Args:
            data: Input DataFrame
            **kwargs: Additional options
            
        Returns:
            Tuple of (enhanced DataFrame, analysis results dict)
        """
        pass