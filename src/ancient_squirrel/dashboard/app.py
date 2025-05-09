import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
import logging


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Dashboard")

# Import dashboard components
from ancient_squirrel.dashboard.components import (
    data_loader,
    overview_tab,
    themes_tab,
    influence_tab,
    network_tab,
    temporal_tab,
    channel_tab,
    nlp_tab,
    strategy_tab,
)

# Import new thumbnail analysis tab
from ancient_squirrel.dashboard.components import thumbnail_analysis_tab

# Import utilities
from ancient_squirrel.utils.data_utils import (
    load_data,
    process_vector_column,
)

def main():
    """Main dashboard application"""
    
    # Setup page config
    st.set_page_config(
        page_title="Ancient Squirrel - YouTube Network Explorer",
        page_icon="🐿️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'analysis_loaded' not in st.session_state:
        st.session_state.analysis_loaded = False
    
    # Check for active tab in session state
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 0
    
    # Determine whether to use sidebar or main page data loader
    if not st.session_state.data_loaded:
        # Use enhanced data loader on main page
        data_loader.render_data_loader(in_sidebar=False)
    else:
        # If data is loaded, show sidebar controls
        with st.sidebar:
            st.title("🐿️ Ancient Squirrel")
            st.write("YouTube Network Theme Explorer")
            
            # Load data section for changing datasets
            data_loader.render_sidebar()
            
            # Display options
            st.header("Display Options")
            
            # Theme visualization options
            st.subheader("Theme Visualization")
            st.session_state.wordcloud_max_words = st.slider("Max Words in Wordcloud", 50, 200, 100)
            st.session_state.theme_color_map = st.selectbox(
                "Theme Color Scheme", 
                ["viridis", "plasma", "inferno", "magma", "cividis"]
            )
            
            # Influence visualization options
            st.subheader("Influence Analysis")
            st.session_state.influence_threshold = st.slider("Influence Percentile Threshold", 50, 95, 75)
            
            # Network visualization options
            st.subheader("Network Visualization")
            st.session_state.min_community_size = st.slider("Minimum Community Size", 3, 30, 5)
    
        # Main content area - render when data is loaded
        if st.session_state.data_loaded:
            # Load the data
            df = st.session_state.data
            
            # Process vector column if exists
            if 'doc_vector' in df.columns:
                df = process_vector_column(df)
                st.session_state.data = df
            
            # Load analysis results if available
            analysis_results = st.session_state.get("analysis_results")
            nlp_results = st.session_state.get("nlp_results")
            
            # Create tabs
            tabs = st.tabs([
                "Overview", 
                "Community Themes", 
                "Influence Analysis", 
                "Network Visualization",
                "Temporal Trends",
                "Channel Analysis",
                "NLP Analysis",
                "Content Strategy",
                "Thumbnail Analysis"  # New tab added
            ])

            # Helper function for safer tab rendering
            def render_tab_safely(tab_function, df, *args):
                try:
                    tab_function(df, *args)
                except Exception as e:
                    st.error(f"Error rendering tab: {str(e)}")
                    st.info("This tab may require additional data. Check the console for more details.")
                    st.expander("Error details").write(str(e))

            # Set active tab from session state
            active_tab = st.session_state.active_tab
            
            # Render each tab safely, but only the active one
            with tabs[0]:
                if active_tab == 0:
                    render_tab_safely(overview_tab.render, df, analysis_results)

            with tabs[1]:
                if active_tab == 1:
                    render_tab_safely(themes_tab.render, df, analysis_results)

            with tabs[2]:
                if active_tab == 2:
                    render_tab_safely(influence_tab.render, df, analysis_results)

            with tabs[3]:
                if active_tab == 3:
                    render_tab_safely(network_tab.render, df, analysis_results)

            with tabs[4]:
                if active_tab == 4:
                    render_tab_safely(temporal_tab.render, df, analysis_results)

            with tabs[5]:
                if active_tab == 5:
                    render_tab_safely(channel_tab.render, df, analysis_results)

            with tabs[6]:
                if active_tab == 6:
                    render_tab_safely(nlp_tab.render, df, analysis_results, nlp_results)

            with tabs[7]:
                if active_tab == 7:
                    render_tab_safely(strategy_tab.render, df, analysis_results, nlp_results)
                    
            with tabs[8]:
                if active_tab == 8:
                    render_tab_safely(thumbnail_analysis_tab.render, df, analysis_results, nlp_results)
            
            # Update active tab based on which tab is clicked
            tabs_clicked = int(st.experimental_get_query_params().get("tab_clicked", [active_tab])[0])
            if tabs_clicked != active_tab:
                st.session_state.active_tab = tabs_clicked

if __name__ == "__main__":
    main()