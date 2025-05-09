import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
from wordcloud import WordCloud

def render(df: pd.DataFrame, analysis_results: Optional[Dict[str, Any]] = None):
    """
    Render the community themes tab
    
    Args:
        df: DataFrame with video data
        analysis_results: Optional analysis results dictionary
    """
    st.header("Community Theme Analysis")
    
    # Show cluster insights if available
    if analysis_results and "cluster_insights" in analysis_results:
        st.subheader("Cluster Insights")
        
        for cid, insight in analysis_results["cluster_insights"].items():
            with st.expander(f"Cluster {cid} — {insight['size']} videos"):
                # Create two columns
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Dominant terms:** {', '.join(insight['centroid_terms'])}")
                    st.markdown(f"**Top bigrams:** {', '.join(insight['top_bigrams'])}")
                    
                    if "viewer_intent" in insight:
                        st.markdown(f"**Viewer intent:** {insight['viewer_intent']}")
                
                with col2:
                    st.markdown("**Top channels (influence):**")
                    # Convert the list of lists to a DataFrame for better display
                    if "top_channels_by_influence" in insight:
                        channels_df = pd.DataFrame(
                            insight["top_channels_by_influence"], 
                            columns=["Channel", "Influence"]
                        )
                        st.dataframe(channels_df, hide_index=True)
                
                # Sample videos
                if "sample_videos" in insight:
                    st.markdown("**Sample videos:**")
                    for video in insight["sample_videos"]:
                        st.markdown(f"- {video}")
    
    # Community selection for detailed analysis
    st.markdown("---")
    st.subheader("Detailed Community Analysis")
    
    # Check for community or cluster column
    community_col = None
    if 'community' in df.columns:
        community_col = 'community'
    elif 'cluster' in df.columns:
        community_col = 'cluster'
    
    if community_col is not None:
        # Community selection
        st.subheader(f"Select {community_col.title()}")
        
        # Get community sizes
        community_sizes = df[community_col].value_counts().to_dict()

        # Create a list of (community_id, size) tuples
        community_options = [(comm_id, community_sizes.get(comm_id, 0)) 
                            for comm_id in df[community_col].unique()]

        # Sort by size (descending)
        sorted_communities = sorted(community_options, key=lambda x: x[1], reverse=True)

        # Create a dropdown with communities sorted by size
        community_id = st.selectbox(
            f"Choose a {community_col} to analyze",
            options=[comm_id for comm_id, _ in sorted_communities],
            format_func=lambda x: f"{community_col.title()} {x} ({community_sizes[x]} videos)"
        )
        
        if community_id is not None:
            # Filter for selected community
            community_df = df[df[community_col] == community_id]
            
            # Community metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Videos", f"{len(community_df):,}")
            
            with col2:
                if 'channel' in community_df.columns:
                    st.metric("Channels", f"{community_df['channel'].nunique():,}")
                else:
                    st.metric("Channels", "N/A")
            
            with col3:
                if 'influence' in community_df.columns:
                    avg_inf = community_df['influence'].mean()
                    st.metric("Avg Influence", f"{avg_inf:.2f}")
                elif 'views' in community_df.columns:
                    total_views = community_df['views'].sum()
                    st.metric("Total Views", f"{total_views:,}")
                else:
                    st.metric("Influence/Views", "N/A")
            
            # Theme visualization
            st.subheader("Theme Visualization")
            
            # WordCloud for theme visualization
            if 'clean_title' in community_df.columns:
                text_col = 'clean_title'
            else:
                text_col = 'title'
            
            text_data = ' '.join(community_df[text_col].fillna('').astype(str))
            
            if text_data:
                try:
                    # Get WordCloud settings from session state or use defaults
                    max_words = st.session_state.get('wordcloud_max_words', 100)
                    colormap = st.session_state.get('theme_color_map', 'viridis')
                    
                    wordcloud = WordCloud(
                        width=800, 
                        height=400,
                        max_words=max_words,
                        background_color='white',
                        colormap=colormap
                    ).generate(text_data)
                    
                    # Convert matplotlib figure to image
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    plt.tight_layout()
                    
                    # Display WordCloud
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error generating wordcloud: {str(e)}")
            
            # Display pre-computed themes if available
            if analysis_results and 'cluster_themes' in analysis_results:
                st.subheader("Extracted Themes")
                
                # Convert community_id to string for lookup
                community_str = str(community_id)
                
                if community_str in analysis_results['cluster_themes']:
                    themes = analysis_results['cluster_themes'][community_str]
                    
                    # Ensure themes are sorted by importance score (descending)
                    sorted_themes = sorted(themes, key=lambda x: x[1], reverse=True)
                    
                    # Display themes as table with additional formatting
                    theme_data = []
                    for term, score in sorted_themes:
                        theme_data.append({"Term": term, "Importance": round(score, 4)})
                    
                    theme_df = pd.DataFrame(theme_data)
                    
                    # Use st.dataframe with formatting options
                    st.dataframe(
                        theme_df,
                        column_config={
                            "Term": st.column_config.TextColumn("Term"),
                            "Importance": st.column_config.NumberColumn(
                                "Importance Score",
                                format="%.4f",
                            )
                        },
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Visualize top terms with improved styling
                    top_n = min(15, len(sorted_themes))
                    top_terms = sorted_themes[:top_n]
                    
                    fig = px.bar(
                        x=[term for term, _ in top_terms],
                        y=[score for _, score in top_terms],
                        labels={"x": "Term", "y": "Importance Score"},
                        title=f"Top {top_n} Terms in {community_col.title()} {community_id}",
                        color=[score for _, score in top_terms],
                        color_continuous_scale=st.session_state.get('theme_color_map', 'viridis')
                    )
                    
                    # Improve layout
                    fig.update_layout(
                        xaxis_title="",
                        yaxis_title="Importance Score",
                        height=400,
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    
                    # Display the chart
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"No theme data available for {community_col} {community_id}")
            
            # Top videos in community
            st.subheader("Most Influential Videos")
            
            if 'influence' in community_df.columns:
                top_videos = community_df.sort_values('influence', ascending=False).head(10)
                display_cols = ['title', 'channel', 'influence']
                
                if 'views' in top_videos.columns:
                    display_cols.append('views')
                
                st.dataframe(
                    top_videos[display_cols],
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("Influence scores not available in dataset")
            
            # Top channels in community
            if 'channel' in community_df.columns:
                st.subheader("Top Channels")
                
                channel_counts = community_df['channel'].value_counts().reset_index()
                channel_counts.columns = ['Channel', 'Videos']
                
                top_n = min(15, len(channel_counts))
                fig = px.bar(
                    channel_counts.head(top_n),
                    y='Channel',
                    x='Videos',
                    orientation='h',
                    title=f"Top {top_n} Channels in {community_col.title()} {community_id}",
                    labels={"Channel": "Channel Name", "Videos": "Number of Videos"},
                    color='Videos',
                    color_continuous_scale=st.session_state.get('theme_color_map', 'viridis')
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No community or cluster column found in the data.")