import streamlit as st
import pandas as pd
import os
import json
import subprocess
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from ancient_squirrel.utils.data_utils import load_data


def render_sidebar():
    """Legacy sidebar renderer - maintained for backward compatibility"""
    st.warning("Using legacy sidebar data loader. Please use the main page data loader for enhanced features.")
    render_data_loader(in_sidebar=True)


def render_data_loader(in_sidebar: bool = False):
    """
    Render the enhanced data loading interface (either in sidebar or main page)
    
    Args:
        in_sidebar: Whether to render in sidebar (legacy mode)
    """
    container = st.sidebar if in_sidebar else st
    
    container.header("Ancient Squirrel Dashboard")
    container.write("YouTube Network and Thumbnail Analysis Explorer")
    
    # Create tabs for the two workflows
    tabs = container.tabs(["Run New Analysis", "Load Existing Results"])
    
    # Tab 1: Run New Analysis
    with tabs[0]:
        render_new_analysis_tab()
    
    # Tab 2: Load Existing Results
    with tabs[1]:
        render_load_existing_tab()


def render_new_analysis_tab():
    """Render the 'Run New Analysis' tab"""
    st.subheader("YouTube Data Input")
    
    # Input file selection
    data_source = st.radio("Data Source", ["Upload File", "Enter Path"])
    
    input_file_path = None
    
    if data_source == "Upload File":
        uploaded_file = st.file_uploader("Upload Video Data", type=["csv", "parquet", "json"])
        if uploaded_file is not None:
            # Save uploaded file to temp location
            temp_dir = Path("temp")
            temp_dir.mkdir(exist_ok=True)
            temp_path = temp_dir / "temp_data.csv"
            
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            input_file_path = str(temp_path)
            st.success(f"File uploaded successfully")
    else:
        input_file_path = st.text_input("Path to Video Data File", "data/youtube_videos.csv")
    
    # Output directory
    output_dir = st.text_input("Output Directory", "output/analysis_results")
    
    # Analysis Parameters (collapsible sections)
    with st.expander("Clustering Parameters", expanded=False):
        num_clusters = st.slider("Number of Clusters", 5, 100, 30)
        clustering_method = st.selectbox("Clustering Method", ["kmeans", "minibatch"])
    
    with st.expander("Thumbnail Analysis Parameters", expanded=True):
        download_thumbnails = st.checkbox("Download Thumbnails", value=True)
        analyze_thumbnails = st.checkbox("Analyze Thumbnails", value=True)
        
        # Only show these options if downloading thumbnails
        if download_thumbnails:
            thumbnail_quality = st.selectbox(
                "Thumbnail Quality", 
                ["max", "high", "medium", "standard", "default"],
                index=0
            )
            force_download = st.checkbox("Force Re-download", value=False)
        
        # Only show these options if analyzing thumbnails
        if analyze_thumbnails:
            use_llm_visual = st.checkbox("Use LLM for Visual Analysis", value=False)
            analyze_title_thumbnail = st.checkbox("Analyze Title-Thumbnail Pairs", value=True)
            
            # Only show if analyzing title-thumbnail pairs
            if analyze_title_thumbnail:
                num_subclusters = st.slider("Number of Visual Subclusters", 2, 10, 3)
    
    with st.expander("NLP Parameters", expanded=False):
        enable_nlp = st.checkbox("Enable NLP Analysis", value=True)
        
        # Only show these options if NLP is enabled
        if enable_nlp:
            use_llm = st.checkbox("Use LLM for Text Insights", value=False)
            num_topics = st.slider("Number of Topics", 5, 50, 15)
            community_topics = st.checkbox("Extract Topics per Community", value=True)
            
            # Only show if community topics enabled
            if community_topics:
                community_topics_count = st.slider("Topics per Community", 3, 15, 8)
    
    with st.expander("Sampling Parameters", expanded=False):
        use_sampling = st.checkbox("Use Sampling for Thumbnail Analysis", value=True, 
                              help="Analyze thumbnails for a sample of videos instead of the entire dataset")
        
        # Only show these options if sampling is enabled
        if use_sampling:
            top_communities = st.slider("Top Communities to Analyze", 5, 30, 15)
            videos_per_community = st.slider("Max Videos per Community", 10, 200, 50)
            min_community_size = st.slider("Minimum Community Size", 5, 50, 10)
            stratified_sampling = st.checkbox("Use Stratified Sampling", value=True,
                                        help="Sample videos across different influence levels")
    
    with st.expander("Performance Parameters", expanded=False):
        num_workers = st.slider("Number of Worker Processes", 1, 16, 4)
        quiet_mode = st.checkbox("Quiet Mode (Less Logging)", value=False)
    
    with st.expander("API Keys", expanded=False):
        use_openai = st.checkbox("Use OpenAI API", value=False)
        
        # Only show API key input if OpenAI is enabled
        if use_openai:
            openai_api_key = st.text_input("OpenAI API Key", type="password")
            if not openai_api_key and 'OPENAI_API_KEY' in os.environ:
                st.info("Using OpenAI API key from environment variable")
                openai_api_key = os.environ['OPENAI_API_KEY']
    
    # Run Analysis button
    if st.button("Run Analysis", type="primary", disabled=not input_file_path):
        if not os.path.exists(input_file_path):
            st.error(f"Input file not found: {input_file_path}")
        else:
            # Create command
            cmd = build_analysis_command(
                input_file=input_file_path,
                output_dir=output_dir,
                num_clusters=num_clusters,
                clustering_method=clustering_method,
                download_thumbnails=download_thumbnails,
                thumbnail_quality=thumbnail_quality if download_thumbnails else None,
                force_download=force_download if download_thumbnails else False,
                analyze_thumbnails=analyze_thumbnails,
                use_llm_visual=use_llm_visual if analyze_thumbnails else False,
                analyze_title_thumbnail=analyze_title_thumbnail if analyze_thumbnails else False,
                num_subclusters=num_subclusters if analyze_thumbnails and analyze_title_thumbnail else 3,
                enable_nlp=enable_nlp,
                use_llm=use_llm if enable_nlp else False,
                num_topics=num_topics if enable_nlp else 15,
                community_topics=community_topics if enable_nlp else False,
                community_topics_count=community_topics_count if enable_nlp and community_topics else 8,
                use_sampling=use_sampling,
                top_communities=top_communities if use_sampling else 15,
                videos_per_community=videos_per_community if use_sampling else 50,
                min_community_size=min_community_size if use_sampling else 10,
                stratified_sampling=stratified_sampling if use_sampling else False,
                num_workers=num_workers,
                quiet_mode=quiet_mode,
                openai_api_key=openai_api_key if use_openai else None
            )
            
            # Show command
            with st.expander("Generated Command", expanded=False):
                st.code(" ".join(cmd))
            
            # Run analysis in background thread and show progress
            run_analysis_with_progress(cmd)


def render_load_existing_tab():
    """Render the 'Load Existing Results' tab"""
    st.subheader("Load Pre-computed Results")
    
    # File selection method
    source_method = st.radio("File Selection Method", ["Enter Paths", "Upload Files"])
    
    # Initialize variables to track loaded data
    data_loaded = False
    analysis_loaded = False
    nlp_loaded = False
    
    if source_method == "Enter Paths":
        # Main data file path
        data_path = st.text_input("Path to Processed Data File", "output/processed_data.json")
        
        # Analysis results path
        analysis_path = st.text_input("Path to Analysis Results File", "output/analysis_results.json")
        
        # NLP results path (optional)
        nlp_path = st.text_input("Path to NLP Results File (Optional)", "")
        
        # Load button
        if st.button("Load Results", type="primary"):
            # Check and load data file
            if data_path and os.path.exists(data_path):
                try:
                    st.session_state.data = load_data(data_path)
                    st.session_state.data_loaded = True
                    data_loaded = True
                    st.success(f"Data loaded successfully: {len(st.session_state.data)} rows")
                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")
            else:
                st.error(f"Data file not found: {data_path}")
            
            # Check and load analysis results
            if analysis_path and os.path.exists(analysis_path):
                try:
                    st.session_state.analysis_results = load_analysis_results(analysis_path)
                    st.session_state.analysis_loaded = True
                    analysis_loaded = True
                    st.success(f"Analysis results loaded successfully")
                except Exception as e:
                    st.error(f"Error loading analysis results: {str(e)}")
            else:
                st.error(f"Analysis file not found: {analysis_path}")
            
            # Check and load NLP results (optional)
            if nlp_path and os.path.exists(nlp_path):
                try:
                    st.session_state.nlp_results = load_analysis_results(nlp_path)
                    st.session_state.nlp_loaded = True
                    nlp_loaded = True
                    st.success(f"NLP results loaded successfully")
                except Exception as e:
                    st.error(f"Error loading NLP results: {str(e)}")
    else:  # Upload Files
        # Data file uploader
        uploaded_data = st.file_uploader("Upload Processed Data", type=["csv", "json", "parquet"])
        if uploaded_data is not None:
            try:
                # Save to temp file and load
                temp_dir = Path("temp")
                temp_dir.mkdir(exist_ok=True)
                temp_path = temp_dir / uploaded_data.name
                
                with open(temp_path, "wb") as f:
                    f.write(uploaded_data.getvalue())
                
                st.session_state.data = load_data(str(temp_path))
                st.session_state.data_loaded = True
                data_loaded = True
                st.success(f"Data loaded successfully: {len(st.session_state.data)} rows")
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
        
        # Analysis results uploader
        uploaded_analysis = st.file_uploader("Upload Analysis Results", type=["json"])
        if uploaded_analysis is not None:
            try:
                # Save to temp file and load
                temp_dir = Path("temp")
                temp_dir.mkdir(exist_ok=True)
                temp_path = temp_dir / uploaded_analysis.name
                
                with open(temp_path, "wb") as f:
                    f.write(uploaded_analysis.getvalue())
                
                st.session_state.analysis_results = load_analysis_results(str(temp_path))
                st.session_state.analysis_loaded = True
                analysis_loaded = True
                st.success(f"Analysis results loaded successfully")
            except Exception as e:
                st.error(f"Error loading analysis results: {str(e)}")
        
        # NLP results uploader (optional)
        uploaded_nlp = st.file_uploader("Upload NLP Results (Optional)", type=["json"])
        if uploaded_nlp is not None:
            try:
                # Save to temp file and load
                temp_dir = Path("temp")
                temp_dir.mkdir(exist_ok=True)
                temp_path = temp_dir / uploaded_nlp.name
                
                with open(temp_path, "wb") as f:
                    f.write(uploaded_nlp.getvalue())
                
                st.session_state.nlp_results = load_analysis_results(str(temp_path))
                st.session_state.nlp_loaded = True
                nlp_loaded = True
                st.success(f"NLP results loaded successfully")
            except Exception as e:
                st.error(f"Error loading NLP results: {str(e)}")
    
    # Data summary if loaded
    if data_loaded or 'data_loaded' in st.session_state and st.session_state.data_loaded:
        df = st.session_state.data
        st.write("### Data Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Videos", f"{len(df):,}")
        
        with col2:
            if 'channel' in df.columns:
                st.metric("Channels", f"{df['channel'].nunique():,}")
            else:
                st.metric("Channels", "N/A")
        
        with col3:
            # Check for community or cluster column
            community_col = None
            if 'community' in df.columns:
                community_col = 'community'
                st.metric("Communities", f"{df['community'].nunique():,}")
            elif 'cluster' in df.columns:
                community_col = 'cluster'
                st.metric("Clusters", f"{df['cluster'].nunique():,}")
            else:
                st.metric("Communities/Clusters", "N/A")
        
        # Data columns preview
        st.write("#### Available Columns:")
        columns_df = pd.DataFrame({
            'Column': df.columns,
            'Non-Null Count': df.count().values,
            'Type': df.dtypes.values
        })
        st.dataframe(columns_df, use_container_width=True)
    
    # Analysis summary if loaded
    if analysis_loaded or 'analysis_loaded' in st.session_state and st.session_state.analysis_loaded:
        analysis = st.session_state.analysis_results
        
        st.write("### Analysis Results Summary")
        
        if 'metadata' in analysis:
            metadata = analysis['metadata']
            
            # Create simplified metadata display
            meta_items = []
            for k, v in metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    meta_items.append({'Property': k, 'Value': v})
            
            meta_df = pd.DataFrame(meta_items)
            st.dataframe(meta_df, use_container_width=True)
        
        # List available analysis components
        components = []
        for key in analysis.keys():
            if key != 'metadata':
                if isinstance(analysis[key], dict):
                    size = len(analysis[key])
                elif isinstance(analysis[key], list):
                    size = len(analysis[key])
                else:
                    size = "N/A"
                
                components.append({'Component': key, 'Size': size})
        
        st.write("#### Available Analysis Components:")
        components_df = pd.DataFrame(components)
        st.dataframe(components_df, use_container_width=True)
    
    # Display thumbnail analysis info if available
    if (analysis_loaded or 'analysis_loaded' in st.session_state and st.session_state.analysis_loaded) and \
       'thumbnail_analysis' in st.session_state.analysis_results:
        
        st.write("### Thumbnail Analysis Available")
        tn_analysis = st.session_state.analysis_results['thumbnail_analysis']
        
        # Create summary of thumbnail analysis
        if 'summary' in tn_analysis:
            st.write("#### Thumbnail Summary:")
            summary = tn_analysis['summary']
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'has_text_percentage' in summary:
                    st.metric("Thumbnails with Text", f"{summary['has_text_percentage']:.1f}%")
                
                if 'color_distribution' in summary:
                    top_color = max(summary['color_distribution'].items(), key=lambda x: x[1])[0] if summary['color_distribution'] else "N/A"
                    st.metric("Most Common Color", top_color)
            
            with col2:
                if 'has_faces_percentage' in summary:
                    st.metric("Thumbnails with Faces", f"{summary['has_faces_percentage']:.1f}%")
                
                if 'composition_types' in summary:
                    top_comp = max(summary['composition_types'].items(), key=lambda x: x[1])[0] if summary['composition_types'] else "N/A"
                    st.metric("Most Common Composition", top_comp)
    
    # Navigation options if data is loaded
    if data_loaded or 'data_loaded' in st.session_state and st.session_state.data_loaded:
        st.write("### Quick Navigation")
        st.write("Jump directly to:")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.button("Overview", on_click=lambda: st.session_state.update({"active_tab": 0}))
            st.button("Themes", on_click=lambda: st.session_state.update({"active_tab": 1}))
        
        with col2:
            st.button("Influence", on_click=lambda: st.session_state.update({"active_tab": 2}))
            st.button("Network", on_click=lambda: st.session_state.update({"active_tab": 3}))
        
        with col3:
            st.button("Temporal", on_click=lambda: st.session_state.update({"active_tab": 4}))
            st.button("Channels", on_click=lambda: st.session_state.update({"active_tab": 5}))
        
        with col4:
            st.button("NLP", on_click=lambda: st.session_state.update({"active_tab": 6}))
            st.button("Thumbnails", on_click=lambda: st.session_state.update({"active_tab": 7}))


def build_analysis_command(
    input_file: str,
    output_dir: str,
    num_clusters: int = 30,
    clustering_method: str = "kmeans",
    download_thumbnails: bool = True,
    thumbnail_quality: Optional[str] = "max",
    force_download: bool = False,
    analyze_thumbnails: bool = True,
    use_llm_visual: bool = False,
    analyze_title_thumbnail: bool = True,
    num_subclusters: int = 3,
    enable_nlp: bool = True,
    use_llm: bool = False,
    num_topics: int = 15,
    community_topics: bool = True,
    community_topics_count: int = 8,
    use_sampling: bool = True,
    top_communities: int = 15,
    videos_per_community: int = 50,
    min_community_size: int = 10,
    stratified_sampling: bool = True,
    num_workers: int = 4,
    quiet_mode: bool = False,
    openai_api_key: Optional[str] = None
) -> List[str]:
    """
    Build the command to run the analysis
    
    Args:
        input_file: Path to input file
        output_dir: Path to output directory
        num_clusters: Number of clusters to extract
        clustering_method: Clustering method
        download_thumbnails: Whether to download thumbnails
        thumbnail_quality: Thumbnail quality level
        force_download: Whether to force re-download of thumbnails
        analyze_thumbnails: Whether to analyze thumbnails
        use_llm_visual: Whether to use LLM for visual analysis
        analyze_title_thumbnail: Whether to analyze title-thumbnail pairs
        num_subclusters: Number of subclusters to identify
        enable_nlp: Whether to enable NLP analysis
        use_llm: Whether to use LLM for text insights
        num_topics: Number of topics to extract
        community_topics: Whether to extract topics per community
        community_topics_count: Number of topics per community
        use_sampling: Whether to use sampling for thumbnail analysis
        top_communities: Number of top communities to analyze
        videos_per_community: Maximum videos per community
        min_community_size: Minimum community size
        stratified_sampling: Whether to use stratified sampling
        num_workers: Number of worker processes
        quiet_mode: Whether to use quiet mode
        openai_api_key: OpenAI API key
        
    Returns:
        List of command arguments
    """
    cmd = ["python", "-m", "ancient_squirrel.cli.analyze"]
    
    # Basic arguments
    cmd.extend(["--input", input_file])
    cmd.extend(["--output", output_dir])
    cmd.extend(["--clusters", str(num_clusters)])
    
    # Thumbnail options
    if download_thumbnails:
        cmd.append("--thumbnails")
        cmd.extend(["--thumbnail-quality", thumbnail_quality])
        
        if force_download:
            cmd.append("--force-thumbnails")
    
    if analyze_thumbnails:
        cmd.append("--analyze-thumbnails")
        
        if analyze_title_thumbnail:
            cmd.append("--joint-analysis")
            cmd.extend(["--subclusters", str(num_subclusters)])
        
        if use_llm_visual:
            cmd.append("--llm")
    
    # NLP options
    if enable_nlp:
        cmd.append("--nlp")
        cmd.extend(["--topics", str(num_topics)])
        
        if use_llm:
            cmd.append("--llm")
        
        if community_topics:
            cmd.append("--community-topics")
            cmd.extend(["--community-topics-count", str(community_topics_count)])
        else:
            cmd.append("--no-community-topics")
    
    # Sampling options
    if use_sampling:
        cmd.append("--use-sampling")
        cmd.extend(["--top-communities", str(top_communities)])
        cmd.extend(["--videos-per-community", str(videos_per_community)])
        cmd.extend(["--min-community-size", str(min_community_size)])
        
        if stratified_sampling:
            cmd.append("--stratified-sampling")
    else:
        cmd.append("--no-sampling")
    
    # Misc options
    if quiet_mode:
        cmd.append("--quiet")
    
    if use_llm_visual or use_llm:
        cmd.append("--openai")
    
    return cmd


def run_analysis_with_progress(cmd: List[str]):
    """
    Run the analysis command with progress reporting
    
    Args:
        cmd: Command to run
    """
    # Create progress container
    progress_container = st.empty()
    log_container = st.empty()
    progress_placeholder = progress_container.progress(0)
    logs = []
    
    def run_process():
        # Create environment with API key if provided
        env = os.environ.copy()
        if 'openai_api_key' in st.session_state and st.session_state.openai_api_key:
            env['OPENAI_API_KEY'] = st.session_state.openai_api_key
        
        # Start process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            env=env
        )
        
        # Monitor output
        total_steps = 8  # Approximate number of steps
        current_step = 0
        
        # Map of keywords to steps
        step_keywords = {
            "Preprocessing video titles": 1,
            "Building word vectors": 2,
            "Clustering videos": 3,
            "Downloading thumbnails": 4,
            "Analyzing thumbnail images": 5,
            "Analyzing title-thumbnail pairs": 6,
            "Extracting cluster themes": 7,
            "Extracting cluster statistics": 7,
            "Analysis complete": 8
        }
        
        # Process output
        for line in process.stdout:
            # Update logs
            logs.append(line.strip())
            if len(logs) > 20:
                logs.pop(0)
            log_container.code("\n".join(logs))
            
            # Update progress based on keywords
            for keyword, step in step_keywords.items():
                if keyword in line:
                    current_step = step
                    progress_placeholder.progress(min(current_step / total_steps, 1.0))
                    break
        
        # Ensure process is complete
        process.wait()
        
        # Final update
        if process.returncode == 0:
            progress_placeholder.progress(1.0)
            progress_container.success("Analysis complete!")
            
            # Update paths for loading
            output_dir = next((cmd[i+1] for i, arg in enumerate(cmd) if arg == "--output"), None)
            if output_dir:
                data_path = os.path.join(output_dir, "processed_data.json")
                analysis_path = os.path.join(output_dir, "analysis_results.json")
                
                # Try to load results automatically
                try:
                    if os.path.exists(data_path):
                        st.session_state.data = load_data(data_path)
                        st.session_state.data_loaded = True
                    
                    if os.path.exists(analysis_path):
                        st.session_state.analysis_results = load_analysis_results(analysis_path)
                        st.session_state.analysis_loaded = True
                    
                    log_container.success(f"Results automatically loaded from {output_dir}")
                except Exception as e:
                    log_container.error(f"Error auto-loading results: {str(e)}")
        else:
            progress_container.error("Analysis failed with errors")
    
    # Run in background thread
    thread = threading.Thread(target=run_process)
    thread.daemon = True
    thread.start()


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