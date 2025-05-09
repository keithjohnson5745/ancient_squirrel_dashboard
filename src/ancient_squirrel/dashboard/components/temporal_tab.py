import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, Any, Optional

def render(df: pd.DataFrame, analysis_results: Optional[Dict[str, Any]] = None):
    """
    Render the temporal trends tab
    
    Args:
        df: DataFrame with video data
        analysis_results: Optional analysis results dictionary
    """
    st.header("Temporal Trends")
    
    if 'publish_date' not in df.columns:
        st.error("Publication date column not found in dataset")
        return
    
    try:
        # Convert to datetime
        df['publish_date'] = pd.to_datetime(df['publish_date'], errors='coerce')
        
        # Create time period columns
        df['year'] = df['publish_date'].dt.year
        df['month'] = df['publish_date'].dt.month
        df['quarter'] = df['publish_date'].dt.quarter
        
        # Time period selection
        time_unit = st.radio("Time Period", ["Year", "Quarter", "Month"])
        
        if time_unit == "Year":
            group_col = 'year'
            format_func = lambda x: f"{x}"
        elif time_unit == "Quarter":
            df['quarter_label'] = df['year'].astype(str) + "-Q" + df['quarter'].astype(str)
            group_col = 'quarter_label'
            format_func = lambda x: x
        else:  # Month
            df['month_label'] = df['year'].astype(str) + "-" + df['month'].astype(str).str.zfill(2)
            group_col = 'month_label'
            format_func = lambda x: x
        
        # Group by time period
        time_groups = df.groupby(group_col)
        periods = sorted(df[group_col].dropna().unique())
        
        # Video count by time period
        st.subheader("Video Count by Time Period")
        
        period_counts = time_groups.size().reset_index()
        period_counts.columns = ['Period', 'Videos']
        
        fig = px.line(
            period_counts,
            x='Period',
            y='Videos',
            markers=True,
            title=f"Videos by {time_unit}",
            labels={"Period": time_unit, "Videos": "Number of Videos"}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Influence trends if available
        if 'influence' in df.columns:
            st.subheader("Influence Trends")
            
            period_influence = time_groups['influence'].mean().reset_index()
            period_influence.columns = ['Period', 'Avg Influence']
            
            fig = px.line(
                period_influence,
                x='Period',
                y='Avg Influence',
                markers=True,
                title=f"Average Influence by {time_unit}",
                labels={"Period": time_unit, "Avg Influence": "Average Influence Score"}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Theme evolution
        st.subheader("Theme Evolution")
        
        text_col = 'clean_title' if 'clean_title' in df.columns else 'title'
        
        if text_col in df.columns:
            # Select time periods to compare
            periods_to_compare = st.multiselect(
                f"Select {time_unit}s to Compare",
                options=periods,
                default=periods[:min(3, len(periods))],
                format_func=format_func
            )
            
            if periods_to_compare:
                # Display top terms for each period
                st.subheader(f"Top Terms by {time_unit}")
                
                # Generate WordCloud for each period
                wordcloud_cols = st.columns(min(len(periods_to_compare), 3))
                term_tables = []
                
                for i, period in enumerate(periods_to_compare):
                    period_df = df[df[group_col] == period]
                    
                    # Skip periods with too few videos
                    if len(period_df) < 5:
                        continue
                    
                    # Display word cloud in columns
                    col_idx = i % len(wordcloud_cols)
                    with wordcloud_cols[col_idx]:
                        st.write(f"### {format_func(period)}")
                        
                        # Generate text data
                        text_data = ' '.join(period_df[text_col].fillna('').astype(str))
                        
                        if text_data:
                            try:
                                # Get WordCloud settings from session state or use defaults
                                colormap = st.session_state.get('theme_color_map', 'viridis')
                                
                                wordcloud = WordCloud(
                                    width=300, 
                                    height=200,
                                    max_words=50,
                                    background_color='white',
                                    colormap=colormap
                                ).generate(text_data)
                                
                                # Display WordCloud
                                fig, ax = plt.subplots(figsize=(5, 3))
                                ax.imshow(wordcloud, interpolation='bilinear')
                                ax.axis('off')
                                plt.tight_layout()
                                st.pyplot(fig)
                            except Exception as e:
                                st.error(f"Error generating wordcloud: {str(e)}")
                    
                    # Extract top terms using TF-IDF
                    try:
                        vectorizer = TfidfVectorizer(max_features=20, stop_words='english')
                        X = vectorizer.fit_transform(period_df[text_col].fillna(''))
                        
                        # Get top terms
                        feature_names = vectorizer.get_feature_names_out()
                        tfidf_sums = X.sum(axis=0).A1
                        
                        # Sort by importance
                        sorted_indices = tfidf_sums.argsort()[::-1]
                        top_terms = [(feature_names[i], float(tfidf_sums[i])) 
                                    for i in sorted_indices[:10]]
                        
                        # Create period DataFrame
                        period_df = pd.DataFrame(
                            [(term, score) for term, score in top_terms],
                            columns=[f"{format_func(period)}", "Score"]
                        )
                        term_tables.append(period_df)
                    except Exception as e:
                        st.error(f"Error extracting terms for period {period}: {str(e)}")
                
                # Display term comparison table
                if term_tables:
                    st.subheader(f"Top Terms Comparison by {time_unit}")
                    term_table = pd.concat(term_tables, axis=1)
                    st.dataframe(term_table, hide_index=True)
        
        # Display pre-computed temporal analysis if available
        if analysis_results and 'temporal_analysis' in analysis_results:
            with st.expander("Pre-computed Temporal Analysis", expanded=False):
                temporal_data = analysis_results['temporal_analysis']
                
                # Display selected period data
                for period, data in temporal_data.items():
                    if period in periods_to_compare:
                        st.subheader(f"Period: {period}")
                        st.write(f"Video count: {data.get('count', 'N/A')}")
                        
                        if 'top_terms' in data:
                            st.write("Top terms:")
                            terms_df = pd.DataFrame(data['top_terms'], columns=['Term', 'Score'])
                            st.dataframe(terms_df.head(10), hide_index=True)
    
    except Exception as e:
        st.error(f"Error in temporal trends: {str(e)}")