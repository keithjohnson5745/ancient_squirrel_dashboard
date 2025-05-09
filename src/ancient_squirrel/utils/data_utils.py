import pandas as pd
import numpy as np
import os
import json
from typing import Dict, List, Optional, Any, Union
import pickle
import logging

logger = logging.getLogger(__name__)

def load_data(file_path: str, format: str = "auto") -> pd.DataFrame:
    """
    Load data from file
    
    Args:
        file_path: Path to the data file
        format: File format (auto, csv, parquet, json)
        
    Returns:
        Loaded DataFrame
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Auto-detect format from extension if not specified
    if format == "auto":
        format = os.path.splitext(file_path)[1].lower().lstrip('.')
    
    # Load based on format
    if format == "csv":
        return pd.read_csv(file_path)
    elif format == "parquet":
        return pd.read_parquet(file_path)
    elif format == "json":
        # Try loading as JSON lines first, fall back to regular JSON
        try:
            return pd.read_json(file_path, lines=True)
        except:
            return pd.read_json(file_path)
    elif format == "pickle" or format == "pkl":
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported format: {format}")

def save_data(df: pd.DataFrame, file_path: str, format: str = "auto") -> None:
    """
    Save data to file
    
    Args:
        df: DataFrame to save
        file_path: Path to save the data
        format: File format (auto, csv, parquet, json)
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    # Auto-detect format from extension if not specified
    if format == "auto":
        format = os.path.splitext(file_path)[1].lower().lstrip('.')
    
    # Save based on format
    if format == "csv":
        df.to_csv(file_path, index=False)
    elif format == "parquet":
        df.to_parquet(file_path, index=False)
    elif format == "json":
        df.to_json(file_path, orient="records", lines=True)
    elif format == "pickle" or format == "pkl":
        with open(file_path, 'wb') as f:
            pickle.dump(df, f)
    else:
        logger.warning(f"Unsupported format: {format}. Defaulting to CSV.")
        df.to_csv(f"{file_path}.csv", index=False)

# Improved process_vector_column function for data_utils.py

def process_vector_column(df: pd.DataFrame, vector_col: str = 'doc_vector') -> pd.DataFrame:
    """
    Convert string representations of vectors to numpy arrays
    
    Args:
        df: Input DataFrame
        vector_col: Column containing vectors
        
    Returns:
        DataFrame with processed vector column
    """
    import numpy as np
    import logging
    logger = logging.getLogger(__name__)
    
    if vector_col not in df.columns:
        return df
    
    # Check if we should process the column at all
    sample_val = df[vector_col].iloc[0] if len(df) > 0 else None
    
    # If already a list/array or None, no processing needed
    if sample_val is None or isinstance(sample_val, (list, np.ndarray)):
        return df
    
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Define a safe conversion function
    def safe_convert_to_array(x):
        if not isinstance(x, str):
            return np.zeros(100)  # Default size
            
        try:
            # Try different parsing approaches
            import ast
            import re
            
            # First try: direct eval
            try:
                return np.array(ast.literal_eval(x))
            except (SyntaxError, ValueError) as e:
                # Second try: clean up and try again
                try:
                    # Remove non-numeric chars except brackets, commas, etc.
                    clean_x = re.sub(r'[^\d\s\.\-e\+\[\],]', '', x)
                    return np.array(ast.literal_eval(clean_x))
                except Exception:
                    # Third try: extract numbers and make array
                    nums = re.findall(r'[-+]?\d*\.\d+|\d+', x)
                    if nums:
                        return np.array([float(n) for n in nums])
                    return np.zeros(100)  # Default fallback
                
        except Exception as e:
            logger.debug(f"Could not convert string to vector: {str(e)}")
            return np.zeros(100)  # Default fallback
    
    # Apply conversion with progress reporting
    try:
        # Process the column
        result_df[vector_col] = df[vector_col].apply(safe_convert_to_array)
        return result_df
    except Exception as e:
        logger.warning(f"Error processing vector column: {str(e)}")
        return df  # Return original if processing fails