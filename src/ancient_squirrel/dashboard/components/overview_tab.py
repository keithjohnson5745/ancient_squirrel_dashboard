import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from typing import Dict, Any, Optional

def render(df: pd.DataFrame, analysis_results: Optional[Dict[str, Any]] = None):
    """
    Render the overview tab
    
    Args:
        df: DataFrame with video data
        analysis_results: Optional analysis results dictionary
    """
    st.header("YouTube Network Overview")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Videos", f"{len(df):,}")
    
    with col2:
        if 'channel' in df.columns:
            st.metric("Unique Channels", f"{df['channel'].nunique():,}")
        else:
            st.metric("Unique Channels", "N/A")
    
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
    
    with col4:
        if 'views' in df.columns:
            total_views = df['views'].sum()
            st.metric("Total Views", f"{total_views:,}")
        elif 'influence' in df.columns:
            avg_influence = df['influence'].mean()
            st.metric("Avg Influence", f"{avg_influence:.2f}")
        else:
            st.metric("Views/Influence", "N/A")
    
    # Show metadata from analysis if available
    if analysis_results and 'metadata' in analysis_results:
        st.subheader("Analysis Metadata")
        
        metadata = analysis_results['metadata']
        meta_cols = st.columns(3)
        
        with meta_cols[0]:
            if 'processing_time' in metadata:
                st.info(f"Processing Time: {metadata['processing_time']}")
        
        with meta_cols[1]:
            if 'cluster_count' in metadata:
                st.info(f"Clusters Analyzed: {metadata['cluster_count']}")
        
        with meta_cols[2]:
            if 'channel_count' in metadata:
                st.info(f"Channels Analyzed: {metadata['channel_count']}")
    
    # Community size distribution
    if community_col:
        st.subheader(f"{community_col.title()} Size Distribution")
        
        community_sizes = df[community_col].value_counts().reset_index()
        community_sizes.columns = ['Community', 'Size']
        
        fig = px.bar(
            community_sizes,
            x='Community',
            y='Size',
            labels={"Community": f"{community_col.title()} ID", "Size": "Number of Videos"},
            title=f"Videos per {community_col.title()}",
            color='Size',
            color_continuous_scale=st.session_state.get('theme_color_map', 'viridis')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Channel distribution
    if 'channel' in df.columns:
        st.subheader("Top Channels")
        
        top_channels = df['channel'].value_counts().head(20).reset_index()
        top_channels.columns = ['Channel', 'Videos']
        
        fig = px.bar(
            top_channels,
            y='Channel',
            x='Videos',
            orientation='h',
            title="Top 20 Channels by Video Count",
            labels={"Channel": "Channel Name", "Videos": "Number of Videos"},
            color='Videos',
            color_continuous_scale=st.session_state.get('theme_color_map', 'viridis')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Data preview
    with st.expander("Data Preview", expanded=False):
        st.subheader("Data Preview")
        st.dataframe(df.head(10), use_container_width=True)