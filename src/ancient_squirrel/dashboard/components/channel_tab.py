import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from typing import Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer

def render(df: pd.DataFrame, analysis_results: Optional[Dict[str, Any]] = None):
    """
    Render the channel analysis tab
    
    Args:
        df: DataFrame with video data
        analysis_results: Optional analysis results dictionary
    """
    st.header("Channel Analysis")
    
    if 'channel' not in df.columns:
        st.error("Channel column not found in dataset")
        return
    
    try:
        # Calculate channel metrics
        channel_counts = df['channel'].value_counts()
        channels = channel_counts.index.tolist()
        
        # Calculate channel influence if available
        channel_influence = None
        if 'influence' in df.columns:
            channel_influence = df.groupby('channel')['influence'].agg(
                ['mean', 'sum', 'max']
            ).reset_index()
            channel_influence.columns = ['channel', 'avg_influence', 'total_influence', 'max_influence']
            
            # Sort by total influence
            channel_influence = channel_influence.sort_values('total_influence', ascending=False)
        
        # Channel selection
        st.subheader("Select Channel")
        
        if channel_influence is not None:
            # Show top channels by influence with preview
            st.write("Top 10 Channels by Total Influence")
            st.dataframe(
                channel_influence.head(10)[['channel', 'total_influence', 'avg_influence']],
                use_container_width=True,
                hide_index=True
            )
            
            # Allow selection from all channels
            selected_channel = st.selectbox(
                "Choose a channel to analyze",
                options=channel_influence['channel'].tolist(),
                index=0
            )
        else:
            # Allow selection from all channels sorted by video count
            selected_channel = st.selectbox(
                "Choose a channel to analyze",
                options=channels,
                index=0,
                format_func=lambda c: f"{c} ({channel_counts[c]} videos)"
            )
        
        if selected_channel:
            # Filter for selected channel
            channel_df = df[df['channel'] == selected_channel]
            
            # Channel metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Videos", f"{len(channel_df):,}")
            
            with col2:
                community_col = None
                if 'community' in df.columns:
                    community_col = 'community'
                elif 'cluster' in df.columns:
                    community_col = 'cluster'
                    
                if community_col is not None:
                    community_count = channel_df[community_col].nunique()
                    st.metric(f"{community_col.title()}s", community_count)
                else:
                    st.metric("Communities", "N/A")
            
            with col3:
                if 'influence' in channel_df.columns:
                    avg_inf = channel_df['influence'].mean()
                    st.metric("Avg Influence", f"{avg_inf:.2f}")
                elif 'views' in channel_df.columns:
                    total_views = channel_df['views'].sum()
                    st.metric("Total Views", f"{total_views:,}")
                else:
                    st.metric("Influence/Views", "N/A")
            
            # Community distribution
            community_col = None
            if 'community' in df.columns:
                community_col = 'community'
            elif 'cluster' in df.columns:
                community_col = 'cluster'
                
            if community_col is not None:
                st.subheader(f"{community_col.title()} Distribution")
                
                community_counts = channel_df[community_col].value_counts().reset_index()
                community_counts.columns = ['Community', 'Videos']
                
                fig = px.pie(
                    community_counts,
                    values='Videos',
                    names='Community',
                    title=f"Videos by {community_col.title()} for {selected_channel}",
                    color_discrete_sequence=px.colors.sequential.Viridis
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Theme analysis
            st.subheader("Channel Themes")
            
            text_col = 'clean_title' if 'clean_title' in channel_df.columns else 'title'
            
            if text_col in channel_df.columns:
                # Generate WordCloud for channel
                text_data = ' '.join(channel_df[text_col].fillna('').astype(str))
                
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
                        
                        # Display WordCloud
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis('off')
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Extract terms using TF-IDF
                        vectorizer = TfidfVectorizer(max_features=30, stop_words='english')
                        X = vectorizer.fit_transform(channel_df[text_col].fillna(''))
                        
                        # Get top terms
                        feature_names = vectorizer.get_feature_names_out()
                        tfidf_sums = X.sum(axis=0).A1
                        
                        # Sort by importance
                        sorted_indices = tfidf_sums.argsort()[::-1]
                        top_terms = [(feature_names[i], float(tfidf_sums[i])) 
                                    for i in sorted_indices[:20]]
                        
                        # Display top terms
                        st.subheader("Top Terms")
                        
                        fig = px.bar(
                            x=[term for term, _ in top_terms],
                            y=[score for _, score in top_terms],
                            labels={"x": "Term", "y": "Importance Score"},
                            title=f"Top Terms for {selected_channel}",
                            color=[score for _, score in top_terms],
                            color_continuous_scale=colormap
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error in channel theme analysis: {str(e)}")
                else:
                    st.info("No text data available for WordCloud generation")
            else:
                st.warning(f"No {text_col} column available for theme analysis")
            
            # Temporal analysis if available
            if 'publish_date' in channel_df.columns:
                st.subheader("Temporal Analysis")
                
                try:
                    # Convert to datetime
                    channel_df['publish_date'] = pd.to_datetime(channel_df['publish_date'], errors='coerce')
                    
                    # Create year column
                    channel_df['year'] = channel_df['publish_date'].dt.year
                    
                    # Group by year
                    year_counts = channel_df.groupby('year').size().reset_index()
                    year_counts.columns = ['Year', 'Videos']
                    
                    # Plot video count by year
                    fig = px.line(
                        year_counts,
                        x='Year',
                        y='Videos',
                        markers=True,
                        title=f"Videos by Year for {selected_channel}",
                        labels={"Year": "Year", "Videos": "Number of Videos"}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error in channel temporal analysis: {str(e)}")
            
            # Top videos
            st.subheader("Top Videos")
            
            if 'influence' in channel_df.columns:
                top_videos = channel_df.sort_values('influence', ascending=False).head(10)
                
                community_col = None
                if 'community' in df.columns:
                    community_col = 'community'
                elif 'cluster' in df.columns:
                    community_col = 'cluster'
                    
                cols_to_show = ['title', 'influence']
                
                if community_col is not None:
                    cols_to_show.append(community_col)
                    
                if 'views' in top_videos.columns:
                    cols_to_show.append('views')
                if 'publish_date' in top_videos.columns:
                    cols_to_show.append('publish_date')
                
                st.dataframe(top_videos[cols_to_show], use_container_width=True, hide_index=True)
            else:
                # Just show videos sorted by community
                if community_col is not None:
                    st.dataframe(
                        channel_df[['title', community_col]].sort_values(community_col),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.dataframe(channel_df[['title']], use_container_width=True, hide_index=True)
    
    except Exception as e:
        st.error(f"Error in channel analysis: {str(e)}")