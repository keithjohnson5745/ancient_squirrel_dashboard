import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer

def render(df: pd.DataFrame, analysis_results: Optional[Dict[str, Any]] = None):
    """
    Render the influence analysis tab
    
    Args:
        df: DataFrame with video data
        analysis_results: Optional analysis results dictionary
    """
    st.header("Influence Analysis")
    
    if 'influence' not in df.columns:
        st.error("Influence score column not found in dataset")
        return
    
    # Get influence threshold from session state or use default
    influence_threshold = st.session_state.get('influence_threshold', 75)
    
    # Calculate influence threshold
    threshold = np.percentile(df['influence'], influence_threshold)
    
    # Split videos by influence
    high_influence = df[df['influence'] >= threshold]
    normal_influence = df[df['influence'] < threshold]
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Influence Threshold", f"{threshold:.2f}")
    
    with col2:
        st.metric("High Influence Videos", f"{len(high_influence):,}")
    
    with col3:
        high_pct = (len(high_influence) / len(df)) * 100
        st.metric("% of Total", f"{high_pct:.1f}%")
    
    # Influence distribution
    st.subheader("Influence Score Distribution")
    
    fig = px.histogram(
        df,
        x='influence',
        nbins=50,
        title="Distribution of Influence Scores",
        labels={"influence": "Influence Score", "count": "Number of Videos"},
        color_discrete_sequence=[px.colors.qualitative.Safe[0]]
    )
    
    fig.add_vline(x=threshold, line_dash="dash", line_color="red",
                annotation_text=f"Threshold ({threshold:.2f})")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Influence by community
    community_col = None
    if 'community' in df.columns:
        community_col = 'community'
    elif 'cluster' in df.columns:
        community_col = 'cluster'
    
    if community_col is not None:
        st.subheader(f"Influence by {community_col.title()}")
        
        community_influence = df.groupby(community_col)['influence'].agg(
            ['mean', 'median', 'max', 'count']
        ).reset_index()
        
        community_influence.columns = ['Community', 'Mean', 'Median', 'Max', 'Videos']
        community_influence = community_influence.sort_values('Mean', ascending=False)
        
        fig = px.bar(
            community_influence,
            x='Community',
            y='Mean',
            color='Videos',
            color_continuous_scale=st.session_state.get('theme_color_map', 'viridis'),
            title=f"Average Influence Score by {community_col.title()}",
            labels={
                "Community": f"{community_col.title()} ID", 
                "Mean": "Mean Influence Score",
                "Videos": "Number of Videos"
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Theme comparison between high and normal influence
    st.subheader("Theme Comparison: High vs. Normal Influence")
    
    clean_text_col = 'clean_title' if 'clean_title' in df.columns else 'title'
    
    if clean_text_col in df.columns:
        try:
            # Use TF-IDF to extract terms
            vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
            
            # Fit on all titles to get consistent feature space
            vectorizer.fit(df[clean_text_col].fillna(''))
            
            # Transform both groups
            high_X = vectorizer.transform(high_influence[clean_text_col].fillna(''))
            normal_X = vectorizer.transform(normal_influence[clean_text_col].fillna(''))
            
            # Calculate average TF-IDF for each term in both groups
            high_avg = high_X.mean(axis=0).A1
            normal_avg = normal_X.mean(axis=0).A1
            
            # Calculate difference in term usage
            term_diff = high_avg - normal_avg
            
            # Get feature names
            feature_names = vectorizer.get_feature_names_out()
            
            # Find terms more common in high influence videos
            high_indices = term_diff.argsort()[::-1][:15]
            high_terms = [(feature_names[i], float(term_diff[i])) for i in high_indices]
            
            # Find terms more common in normal influence videos
            normal_indices = term_diff.argsort()[:15]
            normal_terms = [(feature_names[i], float(term_diff[i])) for i in normal_indices]
            
            # Create a two-panel figure
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Terms Associated with High Influence", 
                              "Terms Associated with Lower Influence"),
                column_widths=[0.5, 0.5]
            )
            
            # Add bars for high influence terms
            fig.add_trace(
                go.Bar(
                    x=[score for _, score in high_terms],
                    y=[term for term, _ in high_terms],
                    orientation='h',
                    marker_color='darkgreen',
                    name="High Influence"
                ),
                row=1, col=1
            )
            
            # Add bars for normal influence terms
            fig.add_trace(
                go.Bar(
                    x=[abs(score) for _, score in normal_terms],
                    y=[term for term, _ in normal_terms],
                    orientation='h',
                    marker_color='darkred',
                    name="Lower Influence"
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                height=500,
                title_text="Terms Associated with Different Influence Levels",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error in influence theme comparison: {str(e)}")
    
    # Display pre-computed influence analysis if available
    if analysis_results and 'influence_analysis' in analysis_results:
        st.subheader("Pre-computed Influence Factors")
        
        influence_data = analysis_results['influence_analysis']
        
        # Display channel influence
        if 'channel_influence' in influence_data:
            st.write("### Channel Influence")
            
            channel_inf = influence_data['channel_influence']
            channel_df = pd.DataFrame([
                {
                    "Channel": channel,
                    "Videos": data['count'],
                    "Avg Influence": data['avg_influence']
                }
                for channel, data in channel_inf.items()
            ])
            
            channel_df = channel_df.sort_values('Avg Influence', ascending=False)
            st.dataframe(channel_df, hide_index=True)
    
    # Top high influence videos
    st.subheader("Top Influential Videos")
    
    top_videos = high_influence.sort_values('influence', ascending=False).head(20)
    
    cols_to_show = ['title', 'channel', 'influence']
    if 'views' in top_videos.columns:
        cols_to_show.append('views')
    if 'publish_date' in top_videos.columns:
        cols_to_show.append('publish_date')
    
    st.dataframe(top_videos[cols_to_show], use_container_width=True, hide_index=True)