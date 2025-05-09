import streamlit as st
import pandas as pd
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

from ancient_squirrel.utils.data_utils import load_data

def render_sidebar():
    """Render the data loading section in the sidebar"""
    
    st.header("Data Loading")
    
    # File upload or path entry
    data_source = st.radio("Data Source", ["Upload File", "Enter Path"])
    
    if data_source == "Upload File":
        uploaded_file = st.file_uploader("Upload Video Data", type=["csv", "parquet"])
        if uploaded_file is not None:
            # Save uploaded file to temp location
            temp_dir = Path("temp")
            temp_dir.mkdir(exist_ok=True)
            temp_path = temp_dir / "temp_data.csv"
            
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            try:
                st.session_state.data = load_data(str(temp_path))
                st.session_state.data_loaded = True
                st.success(f"Data loaded successfully: {len(st.session_state.data)} rows")
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
    else:
        data_path = st.text_input("Path to Video Data File", "data/youtube_videos.csv")
        if st.button("Load Data") and data_path:
            if os.path.exists(data_path):
                try:
                    st.session_state.data = load_data(data_path)
                    st.session_state.data_loaded = True
                    st.success(f"Data loaded successfully: {len(st.session_state.data)} rows")
                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")
            else:
                st.error(f"File not found: {data_path}")
    
    # Analysis results
    if st.session_state.data_loaded:
        st.header("Analysis Results")
        
        analysis_source = st.radio("Analysis Results", ["Upload File", "Enter Path", "None"])
        
        if analysis_source == "Upload File":
            uploaded_analysis = st.file_uploader("Upload Analysis Results", type=["json", "pkl"])
            if uploaded_analysis is not None:
                # Save uploaded file to temp location
                temp_dir = Path("temp")
                temp_dir.mkdir(exist_ok=True)
                
                if uploaded_analysis.name.endswith(".json"):
                    temp_path = temp_dir / "temp_analysis.json"
                else:
                    temp_path = temp_dir / "temp_analysis.pkl"
                
                with open(temp_path, "wb") as f:
                    f.write(uploaded_analysis.getvalue())
                
                try:
                    st.session_state.analysis_results = load_analysis_results(str(temp_path))
                    st.session_state.analysis_loaded = True
                    st.success("Analysis results loaded successfully")
                except Exception as e:
                    st.error(f"Error loading analysis results: {str(e)}")
        
        elif analysis_source == "Enter Path":
            analysis_path = st.text_input("Path to Analysis Results", "output/analysis_results.json")
            if st.button("Load Analysis") and analysis_path:
                if os.path.exists(analysis_path):
                    try:
                        st.session_state.analysis_results = load_analysis_results(analysis_path)
                        st.session_state.analysis_loaded = True
                        st.success("Analysis results loaded successfully")
                    except Exception as e:
                        st.error(f"Error loading analysis results: {str(e)}")
                else:
                    st.error(f"File not found: {analysis_path}")
        
        # Add NLP Data Loading Section
        st.header("NLP Data (Optional)")
        
        nlp_data_source = st.radio("NLP Enhanced Data", ["Upload File", "Enter Path", "None"], index=2)
        
        if nlp_data_source == "Upload File":
            uploaded_nlp_data = st.file_uploader("Upload NLP Enhanced Data", type=["csv"])
            if uploaded_nlp_data is not None:
                # Save uploaded file to temp location
                temp_dir = Path("temp")
                temp_dir.mkdir(exist_ok=True)
                temp_path = temp_dir / "temp_nlp_data.csv"
                
                with open(temp_path, "wb") as f:
                    f.write(uploaded_nlp_data.getvalue())
                
                try:
                    st.session_state.nlp_data = load_data(str(temp_path))
                    st.session_state.nlp_data_loaded = True
                    
                    # Merge NLP data with main data if possible
                    if merge_nlp_data():
                        st.success("NLP data loaded and merged successfully")
                    else:
                        st.success("NLP data loaded successfully")
                except Exception as e:
                    st.error(f"Error loading NLP data: {str(e)}")
        
        elif nlp_data_source == "Enter Path":
            nlp_data_path = st.text_input("Path to NLP Enhanced Data", "output/nlp_enhanced_data.csv")
            if st.button("Load NLP Data") and nlp_data_path:
                if os.path.exists(nlp_data_path):
                    try:
                        st.session_state.nlp_data = load_data(nlp_data_path)
                        st.session_state.nlp_data_loaded = True
                        
                        # Merge NLP data with main data if possible
                        if merge_nlp_data():
                            st.success("NLP data loaded and merged successfully")
                        else:
                            st.success("NLP data loaded successfully")
                    except Exception as e:
                        st.error(f"Error loading NLP data: {str(e)}")
                else:
                    st.error(f"File not found: {nlp_data_path}")
        
        # Add NLP Analysis Results Loading Section
        st.header("NLP Analysis Results (Optional)")
        
        nlp_results_source = st.radio("NLP Analysis Results", ["Upload File", "Enter Path", "None"], index=2)
        
        if nlp_results_source == "Upload File":
            uploaded_nlp_results = st.file_uploader("Upload NLP Analysis Results", type=["json", "pkl"])
            if uploaded_nlp_results is not None:
                # Save uploaded file to temp location
                temp_dir = Path("temp")
                temp_dir.mkdir(exist_ok=True)
                
                if uploaded_nlp_results.name.endswith(".json"):
                    temp_path = temp_dir / "temp_nlp_results.json"
                else:
                    temp_path = temp_dir / "temp_nlp_results.pkl"
                
                with open(temp_path, "wb") as f:
                    f.write(uploaded_nlp_results.getvalue())
                
                try:
                    st.session_state.nlp_results = load_analysis_results(str(temp_path))
                    st.session_state.nlp_results_loaded = True
                    st.success("NLP analysis results loaded successfully")
                except Exception as e:
                    st.error(f"Error loading NLP analysis results: {str(e)}")
        
        elif nlp_results_source == "Enter Path":
            nlp_results_path = st.text_input("Path to NLP Analysis Results", "output/nlp_analysis_results.json")
            if st.button("Load NLP Results") and nlp_results_path:
                if os.path.exists(nlp_results_path):
                    try:
                        st.session_state.nlp_results = load_analysis_results(nlp_results_path)
                        st.session_state.nlp_results_loaded = True
                        st.success("NLP analysis results loaded successfully")
                    except Exception as e:
                        st.error(f"Error loading NLP analysis results: {str(e)}")
                else:
                    st.error(f"File not found: {nlp_results_path}")

def load_analysis_results(results_path: str) -> Dict[str, Any]:
    """
    Load pre-computed analysis results
    
    Args:
        results_path: Path to analysis results file
        
    Returns:
        Dictionary with analysis results
    """
    import pickle
    
    if results_path.endswith(".json"):
        with open(results_path, "r") as f:
            results = json.load(f)
    elif results_path.endswith((".pkl", ".pickle")):
        with open(results_path, "rb") as f:
            results = pickle.load(f)
    else:
        raise ValueError(f"Unsupported file format: {results_path}")
    
    return results

def merge_nlp_data() -> bool:
    """
    Merge NLP data with main data if possible
    
    Returns:
        True if merge successful, False otherwise
    """
    try:
        if not st.session_state.data_loaded or not st.session_state.nlp_data_loaded:
            return False
        
        df = st.session_state.data
        nlp_df = st.session_state.nlp_data
        
        # Check for NLP-specific columns like nmf_topic
        nlp_columns = [col for col in nlp_df.columns if col not in df.columns]
        
        if not nlp_columns:
            st.warning("No new columns found in NLP data")
            return False
        
        # Try to merge datasets on common columns
        common_cols = ['video_id'] if 'video_id' in df.columns and 'video_id' in nlp_df.columns else ['title']
        
        # Merge datasets
        merged_df = pd.merge(df, nlp_df[common_cols + nlp_columns], on=common_cols, how='left')
        
        # Update main dataframe
        st.session_state.data = merged_df
        
        st.info(f"Added {len(nlp_columns)} NLP columns: {', '.join(nlp_columns)}")
        return True
        
    except Exception as e:
        st.error(f"Error merging NLP data: {str(e)}")
        return False