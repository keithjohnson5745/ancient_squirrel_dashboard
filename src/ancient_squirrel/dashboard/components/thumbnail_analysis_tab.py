import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, Any, Optional, List, Tuple
import json
import os
from PIL import Image
import base64
from io import BytesIO
import math
import re


def render(df: pd.DataFrame, analysis_results: Optional[Dict[str, Any]] = None, 
           nlp_results: Optional[Dict[str, Any]] = None):
    """
    Render the thumbnail analysis tab
    
    Args:
        df: DataFrame with video data
        analysis_results: Optional analysis results dictionary
        nlp_results: Optional NLP analysis results
    """
    st.header("Thumbnail Analysis")
    
    # Check if thumbnail data is available
    if 'thumbnail_path' not in df.columns:
        st.warning("No thumbnail paths found in dataset. Run the analysis with '--thumbnails' to download thumbnails.")
        return
    
    # Check if thumbnail analysis is available
    thumbnail_analysis_col = None
    if 'thumbnail_analysis' in df.columns:
        thumbnail_analysis_col = 'thumbnail_analysis'
    
    # Check if joint analysis is available
    joint_analysis_col = None
    if 'title_thumbnail_analysis' in df.columns:
        joint_analysis_col = 'title_thumbnail_analysis'
    
    # Check if LLM analysis is available
    llm_analysis_col = None
    if 'thumbnail_llm_analysis' in df.columns:
        llm_analysis_col = 'thumbnail_llm_analysis'
    elif 'title_thumbnail_llm_analysis' in df.columns:
        llm_analysis_col = 'title_thumbnail_llm_analysis'
    
    # Check for analysis results data
    has_tn_results = False
    has_joint_results = False
    
    if analysis_results:
        if 'thumbnail_analysis' in analysis_results:
            has_tn_results = True
        if 'title_thumbnail_analysis' in analysis_results:
            has_joint_results = True
    
    # If no analysis data found, suggest running the analysis
    if not (thumbnail_analysis_col or joint_analysis_col or llm_analysis_col or has_tn_results or has_joint_results):
        st.warning("""
        No thumbnail analysis found. Run the analysis with these options:
        ```
        ancient-analyze --input data/youtube_videos.csv --thumbnails --analyze-thumbnails --joint-analysis
        ```
        """)
    
    # Create tabs for different analysis sections
    tabs = st.tabs([
        "Title-Thumbnail Alignment", 
        "Clickbait Scoring", 
        "Composition Analysis", 
        "Color Analysis",
        "Thumbnail Browser"
    ])
    
    # 1. Title-Thumbnail Alignment Tab
    with tabs[0]:
        render_alignment_section(df, analysis_results, joint_analysis_col, llm_analysis_col)
    
    # 2. Clickbait Scoring Tab
    with tabs[1]:
        render_clickbait_section(df, analysis_results, joint_analysis_col, thumbnail_analysis_col)
    
    # 3. Composition Analysis Tab
    with tabs[2]:
        render_composition_section(df, analysis_results, thumbnail_analysis_col, llm_analysis_col)
    
    # 4. Color Analysis Tab
    with tabs[3]:
        render_color_section(df, analysis_results, thumbnail_analysis_col)
        
    # 5. Thumbnail Browser Tab
    with tabs[4]:
        render_thumbnail_browser(df, analysis_results, thumbnail_analysis_col, joint_analysis_col, 'thumbnail_path')

def render_alignment_section(df: pd.DataFrame, analysis_results: Optional[Dict[str, Any]], 
                           joint_analysis_col: Optional[str], llm_analysis_col: Optional[str]):
    """
    Render the title-thumbnail alignment section
    
    Args:
        df: DataFrame with video data
        analysis_results: Analysis results dictionary
        joint_analysis_col: Column containing joint analysis
        llm_analysis_col: Column containing LLM analysis
    """
    st.subheader("Title-Thumbnail Alignment Analysis")
    
    # Check if we have the necessary data
    if not joint_analysis_col and ('title_thumbnail_analysis' not in (analysis_results or {})):
        st.info("No title-thumbnail alignment analysis available. Run the analysis with the '--joint-analysis' option.")
        return
    
    # Get alignment scores if available in DataFrame
    alignment_scores = []
    
    if joint_analysis_col and joint_analysis_col in df.columns:
        # Extract alignment scores
        for idx, row in df.iterrows():
            if pd.isna(row[joint_analysis_col]):
                continue
                
            try:
                analysis = json.loads(row[joint_analysis_col])
                score = analysis.get('text_visual_alignment', None)
                
                if score is not None:
                    alignment_scores.append({
                        'index': idx,
                        'title': row['title'] if 'title' in row else '',
                        'score': score,
                        'thumbnail_path': row['thumbnail_path'] if 'thumbnail_path' in row else None
                    })
            except Exception as e:
                continue
    
    # Get alignment data from analysis results if available
    elif analysis_results and 'title_thumbnail_analysis' in analysis_results:
        joint_results = analysis_results['title_thumbnail_analysis']
        
        if 'pattern_statistics' in joint_results and 'text_visual_alignment' in joint_results['pattern_statistics']:
            alignment_stats = joint_results['pattern_statistics']['text_visual_alignment']
            
            # Display overall statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Average Alignment Score", f"{alignment_stats.get('mean', 0):.2f}")
            
            with col2:
                st.metric("Median Alignment Score", f"{alignment_stats.get('median', 0):.2f}")
            
            with col3:
                st.metric("High Alignment %", f"{alignment_stats.get('high_percentage', 0):.1f}%")
    
    # If we have alignment scores from the DataFrame, display visualizations
    if alignment_scores:
        # Convert to DataFrame for easier manipulation
        scores_df = pd.DataFrame(alignment_scores)
        
        # Display overall statistics
        mean_score = scores_df['score'].mean()
        median_score = scores_df['score'].median()
        high_pct = (scores_df['score'] > 0.7).mean() * 100
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Average Alignment Score", f"{mean_score:.2f}")
        
        with col2:
            st.metric("Median Alignment Score", f"{median_score:.2f}")
        
        with col3:
            st.metric("High Alignment %", f"{high_pct:.1f}%")
        
        # Distribution of alignment scores
        st.subheader("Alignment Score Distribution")
        
        fig = px.histogram(
            scores_df, 
            x='score',
            nbins=20,
            title="Distribution of Title-Thumbnail Alignment Scores",
            labels={"score": "Alignment Score"},
            color_discrete_sequence=[px.colors.qualitative.Safe[0]]
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Examples of high and low alignment
        st.subheader("Examples by Alignment Level")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### High Alignment Examples")
            high_examples = scores_df.sort_values('score', ascending=False).head(5)
            
            for _, example in high_examples.iterrows():
                with st.expander(f"Score: {example['score']:.2f}"):
                    st.write(f"**Title:** {example['title']}")
                    
                    if example['thumbnail_path'] and os.path.exists(example['thumbnail_path']):
                        try:
                            st.image(example['thumbnail_path'], width=300)
                        except Exception as e:
                            st.error(f"Error displaying image: {e}")
        
        with col2:
            st.write("### Low Alignment Examples")
            low_examples = scores_df.sort_values('score').head(5)
            
            for _, example in low_examples.iterrows():
                with st.expander(f"Score: {example['score']:.2f}"):
                    st.write(f"**Title:** {example['title']}")
                    
                    if example['thumbnail_path'] and os.path.exists(example['thumbnail_path']):
                        try:
                            st.image(example['thumbnail_path'], width=300)
                        except Exception as e:
                            st.error(f"Error displaying image: {e}")

def render_alignment_section_continued(df, analysis_results, joint_analysis_col, llm_analysis_col, scores_df=None, alignment_scores=None):
    """This is a continuation of the render_alignment_section function"""
    
    # Display LLM insights if available
    if llm_analysis_col and llm_analysis_col in df.columns:
        st.subheader("LLM Insights on Title-Thumbnail Alignment")
        
        # Sample a few LLM analyses
        llm_samples = []
        
        for idx, row in df.iterrows():
            if pd.isna(row[llm_analysis_col]):
                continue
                
            try:
                analysis = json.loads(row[llm_analysis_col])
                
                # Check if this is a title-thumbnail analysis
                if 'visual_reinforcement' in analysis or 'content_strategy' in analysis:
                    llm_samples.append({
                        'title': row['title'] if 'title' in row else '',
                        'thumbnail_path': row['thumbnail_path'] if 'thumbnail_path' in row else None,
                        'analysis': analysis
                    })
                    
                    # Limit to a few samples
                    if len(llm_samples) >= 3:
                        break
            except Exception as e:
                continue
        
        # Display samples
        for sample in llm_samples:
            with st.expander(f"Analysis for: {sample['title'][:50]}..."):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    if sample['thumbnail_path'] and os.path.exists(sample['thumbnail_path']):
                        try:
                            st.image(sample['thumbnail_path'], width=200)
                        except Exception as e:
                            st.error(f"Error displaying image: {e}")
                
                with col2:
                    st.write(f"**Title:** {sample['title']}")
                    
                    if 'content_strategy' in sample['analysis']:
                        st.write(f"**Strategy:** {sample['analysis']['content_strategy']}")
                    
                    if 'visual_reinforcement' in sample['analysis']:
                        st.write(f"**Visual Reinforcement:** {sample['analysis']['visual_reinforcement']}")
                    
                    if 'effectiveness_score' in sample['analysis']:
                        st.metric("Effectiveness Score", f"{sample['analysis']['effectiveness_score']}/10")
    
    # If no individual scores were found, but we have pattern statistics
    elif not alignment_scores and analysis_results and 'title_thumbnail_analysis' in analysis_results:
        joint_results = analysis_results['title_thumbnail_analysis']
        
        if 'pattern_statistics' in joint_results and 'patterns' in joint_results['pattern_statistics']:
            patterns = joint_results['pattern_statistics']['patterns']
            
            # Get patterns related to alignment
            alignment_patterns = {k: v for k, v in patterns.items() 
                                if k in ['thumbnail_text', 'thumbnail_faces']}
            
            if alignment_patterns:
                st.subheader("Thumbnail Elements Supporting Alignment")
                
                # Create bar chart of alignment-related patterns
                pattern_df = pd.DataFrame([
                    {'Pattern': k, 'Percentage': v.get('percentage', 0)}
                    for k, v in alignment_patterns.items()
                ])
                
                if not pattern_df.empty:
                    fig = px.bar(
                        pattern_df,
                        y='Pattern',
                        x='Percentage',
                        orientation='h',
                        title="Percentage of Thumbnails with Text/Faces",
                        labels={"Pattern": "Element", "Percentage": "% of Thumbnails"},
                        color='Percentage',
                        color_continuous_scale=st.session_state.get('theme_color_map', 'viridis')
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    # Influence correlation if available
    if 'influence' in df.columns and joint_analysis_col and joint_analysis_col in df.columns:
        st.subheader("Alignment vs. Influence")
        
        # Extract alignment scores and influences
        alignment_influence = []
        
        for idx, row in df.iterrows():
            if pd.isna(row[joint_analysis_col]) or pd.isna(row['influence']):
                continue
                
            try:
                analysis = json.loads(row[joint_analysis_col])
                score = analysis.get('text_visual_alignment', None)
                
                if score is not None:
                    alignment_influence.append({
                        'alignment': score,
                        'influence': row['influence']
                    })
            except Exception as e:
                continue
        
        if alignment_influence:
            # Convert to DataFrame
            ai_df = pd.DataFrame(alignment_influence)
            
            # Calculate correlation
            correlation = ai_df['alignment'].corr(ai_df['influence'])
            
            # Display correlation
            st.metric("Correlation with Influence", f"{correlation:.2f}")
            
            # Scatter plot
            fig = px.scatter(
                ai_df,
                x='alignment',
                y='influence',
                title="Alignment Score vs. Influence",
                labels={"alignment": "Alignment Score", "influence": "Influence"},
                trendline="ols",
                opacity=0.7
            )
            st.plotly_chart(fig, use_container_width=True)

def render_clickbait_section(df: pd.DataFrame, analysis_results: Optional[Dict[str, Any]], 
                          joint_analysis_col: Optional[str], thumbnail_analysis_col: Optional[str]):
    """
    Render the clickbait scoring section
    
    Args:
        df: DataFrame with video data
        analysis_results: Analysis results dictionary
        joint_analysis_col: Column containing joint analysis
        thumbnail_analysis_col: Column containing thumbnail analysis
    """
    st.subheader("Clickbait Analysis")
    
    # Check if we have the necessary data
    if not joint_analysis_col and ('title_thumbnail_analysis' not in (analysis_results or {})):
        st.info("No clickbait scoring analysis available. Run the analysis with the '--joint-analysis' option.")
        return
    
    # Get clickbait scores if available in DataFrame
    clickbait_scores = []
    
    if joint_analysis_col and joint_analysis_col in df.columns:
        # Extract clickbait scores
        for idx, row in df.iterrows():
            if pd.isna(row[joint_analysis_col]):
                continue
                
            try:
                analysis = json.loads(row[joint_analysis_col])
                score = analysis.get('clickbait_score', None)
                patterns = analysis.get('patterns', [])
                
                if score is not None:
                    clickbait_scores.append({
                        'index': idx,
                        'title': row['title'] if 'title' in row else '',
                        'score': score,
                        'patterns': patterns,
                        'thumbnail_path': row['thumbnail_path'] if 'thumbnail_path' in row else None
                    })
            except Exception as e:
                continue
    
    # Get clickbait data from analysis results if available
    elif analysis_results and 'title_thumbnail_analysis' in analysis_results:
        joint_results = analysis_results['title_thumbnail_analysis']
        
        if 'pattern_statistics' in joint_results and 'clickbait_score' in joint_results['pattern_statistics']:
            clickbait_stats = joint_results['pattern_statistics']['clickbait_score']
            
            # Display overall statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Average Clickbait Score", f"{clickbait_stats.get('mean', 0):.2f}")
            
            with col2:
                st.metric("Median Clickbait Score", f"{clickbait_stats.get('median', 0):.2f}")
            
            with col3:
                st.metric("High Clickbait %", f"{clickbait_stats.get('high_percentage', 0):.1f}%")

def render_clickbait_section_continued(df, analysis_results, joint_analysis_col, clickbait_scores=None):
    """This is a continuation of the render_clickbait_section function"""
    
    # If we have clickbait scores from the DataFrame, display visualizations
    if clickbait_scores:
        # Convert to DataFrame for easier manipulation
        scores_df = pd.DataFrame(clickbait_scores)
        
        # Display overall statistics
        mean_score = scores_df['score'].mean()
        median_score = scores_df['score'].median()
        high_pct = (scores_df['score'] > 0.6).mean() * 100
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Average Clickbait Score", f"{mean_score:.2f}")
        
        with col2:
            st.metric("Median Clickbait Score", f"{median_score:.2f}")
        
        with col3:
            st.metric("High Clickbait %", f"{high_pct:.1f}%")
        
        # Distribution of clickbait scores
        st.subheader("Clickbait Score Distribution")
        
        fig = px.histogram(
            scores_df, 
            x='score',
            nbins=20,
            title="Distribution of Clickbait Scores",
            labels={"score": "Clickbait Score"},
            color_discrete_sequence=[px.colors.qualitative.Set1[0]]
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Clickbait patterns
        st.subheader("Clickbait Patterns")
        
        # Extract pattern frequencies
        pattern_counts = {}
        
        for patterns in scores_df['patterns']:
            for pattern in patterns:
                if pattern in pattern_counts:
                    pattern_counts[pattern] += 1
                else:
                    pattern_counts[pattern] = 1
        
        # Convert to DataFrame
        patterns_df = pd.DataFrame([
            {'Pattern': pattern, 'Count': count, 'Percentage': (count / len(scores_df)) * 100}
            for pattern, count in pattern_counts.items()
        ]).sort_values('Count', ascending=False)
        
        # Display pattern bar chart
        if not patterns_df.empty:
            fig = px.bar(
                patterns_df,
                y='Pattern',
                x='Percentage',
                orientation='h',
                title="Clickbait Patterns in Thumbnails",
                labels={"Pattern": "Pattern", "Percentage": "% of Thumbnails"},
                color='Percentage',
                color_continuous_scale=st.session_state.get('theme_color_map', 'viridis')
            )
            st.plotly_chart(fig, use_container_width=True)

def render_clickbait_examples(scores_df):
    """Render examples of high and low clickbait thumbnails"""
    
    # Examples of high and low clickbait
    st.subheader("Examples by Clickbait Level")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### High Clickbait Examples")
        high_examples = scores_df.sort_values('score', ascending=False).head(5)
        
        for _, example in high_examples.iterrows():
            with st.expander(f"Score: {example['score']:.2f}"):
                st.write(f"**Title:** {example['title']}")
                st.write(f"**Patterns:** {', '.join(example['patterns'])}")
                
                if example['thumbnail_path'] and os.path.exists(example['thumbnail_path']):
                    try:
                        st.image(example['thumbnail_path'], width=300)
                    except Exception as e:
                        st.error(f"Error displaying image: {e}")
    
    with col2:
        st.write("### Low Clickbait Examples")
        low_examples = scores_df.sort_values('score').head(5)
        
        for _, example in low_examples.iterrows():
            with st.expander(f"Score: {example['score']:.2f}"):
                st.write(f"**Title:** {example['title']}")
                st.write(f"**Patterns:** {', '.join(example['patterns'])}")
                
                if example['thumbnail_path'] and os.path.exists(example['thumbnail_path']):
                    try:
                        st.image(example['thumbnail_path'], width=300)
                    except Exception as e:
                        st.error(f"Error displaying image: {e}")


def render_clickbait_influence_correlation(df, joint_analysis_col):
    """Render correlation between clickbait score and influence"""
    
    # Influence correlation if available
    if 'influence' in df.columns and joint_analysis_col and joint_analysis_col in df.columns:
        st.subheader("Clickbait vs. Influence")
        
        # Extract clickbait scores and influences
        clickbait_influence = []
        
        for idx, row in df.iterrows():
            if pd.isna(row[joint_analysis_col]) or pd.isna(row['influence']):
                continue
                
            try:
                analysis = json.loads(row[joint_analysis_col])
                score = analysis.get('clickbait_score', None)
                
                if score is not None:
                    clickbait_influence.append({
                        'clickbait': score,
                        'influence': row['influence']
                    })
            except Exception as e:
                continue
        
        if clickbait_influence:
            # Convert to DataFrame
            ci_df = pd.DataFrame(clickbait_influence)
            
            # Calculate correlation
            correlation = ci_df['clickbait'].corr(ci_df['influence'])
            
            # Display correlation
            st.metric("Correlation with Influence", f"{correlation:.2f}")
            
            # Scatter plot
            fig = px.scatter(
                ci_df,
                x='clickbait',
                y='influence',
                title="Clickbait Score vs. Influence",
                labels={"clickbait": "Clickbait Score", "influence": "Influence"},
                trendline="ols",
                opacity=0.7
            )
            st.plotly_chart(fig, use_container_width=True)

def render_influence_patterns(analysis_results):
    """Render patterns by influence level"""
    
    # Show influence patterns if available in analysis results
    if analysis_results and 'title_thumbnail_analysis' in analysis_results:
        joint_results = analysis_results['title_thumbnail_analysis']
        
        if 'influence_patterns' in joint_results:
            influence_patterns = joint_results['influence_patterns']
            
            if 'high_influence_patterns' in influence_patterns and 'low_influence_patterns' in influence_patterns:
                st.subheader("Patterns by Influence Level")
                
                # High influence patterns
                high_patterns = influence_patterns['high_influence_patterns']
                low_patterns = influence_patterns['low_influence_patterns']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("### High Influence Patterns")
                    
                    high_df = pd.DataFrame([
                        {'Pattern': pattern, 'Difference': diff}
                        for pattern, diff in high_patterns.items()
                    ]).sort_values('Difference', ascending=False)
                    
                    if not high_df.empty:
                        fig = px.bar(
                            high_df,
                            y='Pattern',
                            x='Difference',
                            orientation='h',
                            title="Patterns More Common in High-Influence Videos",
                            labels={"Pattern": "Pattern", "Difference": "% Difference"},
                            color='Difference',
                            color_continuous_scale='Greens'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.write("### Low Influence Patterns")
                    
                    low_df = pd.DataFrame([
                        {'Pattern': pattern, 'Difference': diff}
                        for pattern, diff in low_patterns.items()
                    ]).sort_values('Difference', ascending=False)
                    
                    if not low_df.empty:
                        fig = px.bar(
                            low_df,
                            y='Pattern',
                            x='Difference',
                            orientation='h',
                            title="Patterns More Common in Low-Influence Videos",
                            labels={"Pattern": "Pattern", "Difference": "% Difference"},
                            color='Difference',
                            color_continuous_scale='Reds'
                        )
                        st.plotly_chart(fig, use_container_width=True)


def render_composition_section(df: pd.DataFrame, analysis_results: Optional[Dict[str, Any]], 
                             thumbnail_analysis_col: Optional[str], llm_analysis_col: Optional[str]):
    """
    Render the composition analysis section
    
    Args:
        df: DataFrame with video data
        analysis_results: Analysis results dictionary
        thumbnail_analysis_col: Column containing thumbnail analysis
        llm_analysis_col: Column containing LLM analysis
    """
    st.subheader("Thumbnail Composition Analysis")
    
    # Check if we have the necessary data
    if not thumbnail_analysis_col and ('thumbnail_analysis' not in (analysis_results or {})):
        st.info("No thumbnail composition analysis available. Run the analysis with the '--analyze-thumbnails' option.")
        return
    
    # Get composition data if available in DataFrame
    composition_data = []
    
    if thumbnail_analysis_col and thumbnail_analysis_col in df.columns:
        # Extract composition data
        for idx, row in df.iterrows():
            if pd.isna(row[thumbnail_analysis_col]):
                continue
                
            try:
                analysis = json.loads(row[thumbnail_analysis_col])
                
                # Check for composition data
                if 'composition' in analysis:
                    comp = analysis['composition']
                    
                    composition_data.append({
                        'index': idx,
                        'title': row['title'] if 'title' in row else '',
                        'has_text': comp.get('has_text', False),
                        'has_faces': comp.get('has_faces', False),
                        'thumbnail_path': row['thumbnail_path'] if 'thumbnail_path' in row else None
                    })
            except Exception as e:
                continue
def render_composition_section_summary(analysis_results):
    """Render composition summary statistics from analysis results"""
    
    # Get composition data from analysis results if available
    if analysis_results and 'thumbnail_analysis' in analysis_results:
        tn_results = analysis_results['thumbnail_analysis']
        
        if 'summary' in tn_results:
            summary = tn_results['summary']
            
            # Display overall statistics
            col1, col2 = st.columns(2)
            
            with col1:
                if 'has_text_percentage' in summary:
                    st.metric("Thumbnails with Text", f"{summary['has_text_percentage']:.1f}%")
            
            with col2:
                if 'has_faces_percentage' in summary:
                    st.metric("Thumbnails with Faces", f"{summary['has_faces_percentage']:.1f}%")
            
            # Display composition types if available
            if 'composition_types' in summary:
                st.subheader("Composition Types")
                
                comp_types = summary['composition_types']
                
                # Convert to DataFrame
                comp_df = pd.DataFrame([
                    {'Type': comp_type, 'Percentage': pct}
                    for comp_type, pct in comp_types.items()
                ]).sort_values('Percentage', ascending=False)
                
                if not comp_df.empty:
                    fig = px.bar(
                        comp_df,
                        y='Type',
                        x='Percentage',
                        orientation='h',
                        title="Composition Types in Thumbnails",
                        labels={"Type": "Composition Type", "Percentage": "% of Thumbnails"},
                        color='Percentage',
                        color_continuous_scale=st.session_state.get('theme_color_map', 'viridis')
                    )
                    st.plotly_chart(fig, use_container_width=True)


def render_composition_visualizations(composition_data):
    """Render visualizations for composition data"""
    
    # If we have composition data from the DataFrame, display visualizations
    if composition_data:
        # Convert to DataFrame for easier manipulation
        comp_df = pd.DataFrame(composition_data)
        
        # Display overall statistics
        text_pct = comp_df['has_text'].mean() * 100
        faces_pct = comp_df['has_faces'].mean() * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Thumbnails with Text", f"{text_pct:.1f}%")
        
        with col2:
            st.metric("Thumbnails with Faces", f"{faces_pct:.1f}%")
        
        # Composition element combinations
        st.subheader("Composition Element Combinations")
        
        # Get combination counts
        combinations = comp_df.groupby(['has_text', 'has_faces']).size().reset_index(name='count')
        total = len(comp_df)
        combinations['percentage'] = (combinations['count'] / total) * 100
        
        # Create labels for combinations
        combinations['combination'] = combinations.apply(
            lambda x: f"{'Text' if x['has_text'] else 'No Text'}, {'Faces' if x['has_faces'] else 'No Faces'}", 
            axis=1
        )
        
        # Sort combinations
        combinations = combinations.sort_values('count', ascending=False)
        
        # Display combinations
        fig = px.bar(
            combinations,
            x='combination',
            y='percentage',
            title="Composition Element Combinations",
            labels={"combination": "Elements", "percentage": "% of Thumbnails"},
            color='percentage',
            color_continuous_scale=st.session_state.get('theme_color_map', 'viridis')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        return comp_df
    
    return None

def render_composition_examples(comp_df):
    """Render examples for each composition type"""
    
    # Examples of each composition type
    st.subheader("Examples by Composition Type")
    
    # Group by composition type
    groups = [
        ('Text & Faces', comp_df[comp_df['has_text'] & comp_df['has_faces']]),
        ('Text Only', comp_df[comp_df['has_text'] & ~comp_df['has_faces']]),
        ('Faces Only', comp_df[~comp_df['has_text'] & comp_df['has_faces']]),
        ('Neither Text nor Faces', comp_df[~comp_df['has_text'] & ~comp_df['has_faces']])
    ]
    
    # Display examples for each group
    total = len(comp_df)
    for group_name, group_df in groups:
        if len(group_df) > 0:
            with st.expander(f"{group_name} ({len(group_df)} thumbnails, {len(group_df)/total*100:.1f}%)"):
                # Sample a few examples
                examples = group_df.sample(min(3, len(group_df)))
                
                cols = st.columns(min(3, len(examples)))
                
                for i, (_, example) in enumerate(examples.iterrows()):
                    with cols[i]:
                        st.write(f"**Title:** {example['title'][:50]}...")
                        
                        if example['thumbnail_path'] and os.path.exists(example['thumbnail_path']):
                            try:
                                st.image(example['thumbnail_path'], width=200)
                            except Exception as e:
                                st.error(f"Error displaying image: {e}")


def render_llm_composition_insights(df, llm_analysis_col):
    """Render LLM insights on composition"""
    
    # Display LLM insights if available
    if llm_analysis_col and llm_analysis_col in df.columns:
        st.subheader("LLM Insights on Composition")
        
        # Sample a few LLM analyses
        llm_samples = []
        
        for idx, row in df.iterrows():
            if pd.isna(row[llm_analysis_col]):
                continue
                
            try:
                analysis = json.loads(row[llm_analysis_col])
                
                # Check if this has visual elements section
                if ('VISUAL_ELEMENTS' in analysis or 
                    'visual_elements' in analysis or 
                    'THEMATIC_ELEMENTS' in analysis):
                    
                    llm_samples.append({
                        'title': row['title'] if 'title' in row else '',
                        'thumbnail_path': row['thumbnail_path'] if 'thumbnail_path' in row else None,
                        'analysis': analysis
                    })
                    
                    # Limit to a few samples
                    if len(llm_samples) >= 3:
                        break
            except Exception as e:
                continue
        
        # Display samples
        for sample in llm_samples:
            with st.expander(f"Composition Analysis for: {sample['title'][:50]}..."):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    if sample['thumbnail_path'] and os.path.exists(sample['thumbnail_path']):
                        try:
                            st.image(sample['thumbnail_path'], width=200)
                        except Exception as e:
                            st.error(f"Error displaying image: {e}")
                
                with col2:
                    st.write(f"**Title:** {sample['title']}")
                    
                    # Extract relevant visual elements
                    visual_elements = None
                    if 'VISUAL_ELEMENTS' in sample['analysis']:
                        visual_elements = sample['analysis']['VISUAL_ELEMENTS']
                    elif 'visual_elements' in sample['analysis']:
                        visual_elements = sample['analysis']['visual_elements']
                    
                    if visual_elements:
                        # Display composition info
                        st.write("**Visual Elements:**")
                        st.json(visual_elements, expanded=False)
                    
                    # Extract thematic elements if available
                    thematic_elements = None
                    if 'THEMATIC_ELEMENTS' in sample['analysis']:
                        thematic_elements = sample['analysis']['THEMATIC_ELEMENTS']
                    elif 'thematic_elements' in sample['analysis']:
                        thematic_elements = sample['analysis']['thematic_elements']
                    
                    if thematic_elements:
                        st.write("**Thematic Elements:**")
                        st.json(thematic_elements, expanded=False)

def render_composition_influence_correlation(df, thumbnail_analysis_col):
    """Render correlation between composition elements and influence"""
    
    # Influence correlation if available
    if 'influence' in df.columns and thumbnail_analysis_col and thumbnail_analysis_col in df.columns:
        st.subheader("Composition vs. Influence")
        
        # Extract composition elements and influences
        composition_influence = []
        
        for idx, row in df.iterrows():
            if pd.isna(row[thumbnail_analysis_col]) or pd.isna(row['influence']):
                continue
                
            try:
                analysis = json.loads(row[thumbnail_analysis_col])
                
                if 'composition' in analysis:
                    comp = analysis['composition']
                    
                    composition_influence.append({
                        'has_text': comp.get('has_text', False),
                        'has_faces': comp.get('has_faces', False),
                        'influence': row['influence']
                    })
            except Exception as e:
                continue
        
        if composition_influence:
            # Convert to DataFrame
            ci_df = pd.DataFrame(composition_influence)
            
            # Group by composition elements
            grouped = ci_df.groupby(['has_text', 'has_faces'])['influence'].agg(['mean', 'count']).reset_index()
            
            # Create labels
            grouped['combination'] = grouped.apply(
                lambda x: f"{'Text' if x['has_text'] else 'No Text'}, {'Faces' if x['has_faces'] else 'No Faces'}", 
                axis=1
            )
            
            # Sort by mean influence
            grouped = grouped.sort_values('mean', ascending=False)
            
            # Display influence by composition type
            fig = px.bar(
                grouped,
                x='combination',
                y='mean',
                title="Average Influence by Composition Type",
                labels={"combination": "Elements", "mean": "Average Influence"},
                color='mean',
                color_continuous_scale='Viridis',
                text='count'
            )
            
            fig.update_traces(texttemplate='%{text} videos', textposition='outside')
            
            st.plotly_chart(fig, use_container_width=True)


def render_color_section(df: pd.DataFrame, analysis_results: Optional[Dict[str, Any]], 
                        thumbnail_analysis_col: Optional[str]):
    """
    Render the color analysis section
    
    Args:
        df: DataFrame with video data
        analysis_results: Analysis results dictionary
        thumbnail_analysis_col: Column containing thumbnail analysis
    """
    st.subheader("Thumbnail Color Analysis")
    
    # Check if we have the necessary data
    if not thumbnail_analysis_col and ('thumbnail_analysis' not in (analysis_results or {})):
        st.info("No color analysis available. Run the analysis with the '--analyze-thumbnails' option.")
        return
    
    # Get color data if available in DataFrame
    color_data = []
    
    if thumbnail_analysis_col and thumbnail_analysis_col in df.columns:
        # Extract color data
        for idx, row in df.iterrows():
            if pd.isna(row[thumbnail_analysis_col]):
                continue
                
            try:
                analysis = json.loads(row[thumbnail_analysis_col])
                
                # Check for colors data
                if 'colors' in analysis:
                    colors = analysis['colors']
                    
                    # Get dominant colors
                    dominants = colors.get('dominant', [])
                    
                    # Get the most dominant color
                    if dominants:
                        top_color = dominants[0]
                        
                        color_data.append({
                            'index': idx,
                            'title': row['title'] if 'title' in row else '',
                            'color_name': top_color.get('name', 'unknown'),
                            'color_hex': top_color.get('hex', '#000000'),
                            'color_percentage': top_color.get('percentage', 0),
                            'brightness': colors.get('brightness', 0.5),
                            'contrast': colors.get('contrast', 0.5),
                            'thumbnail_path': row['thumbnail_path'] if 'thumbnail_path' in row else None,
                            'influence': row.get('influence', None)
                        })
            except Exception as e:
                continue

def render_color_summary_from_results(analysis_results):
    """Render color summary from analysis results"""
    
    # Get color data from analysis results if available
    if analysis_results and 'thumbnail_analysis' in analysis_results:
        tn_results = analysis_results['thumbnail_analysis']
        
        if 'summary' in tn_results:
            summary = tn_results['summary']
            
            # Display color distribution if available
            if 'color_distribution' in summary:
                st.subheader("Color Distribution")
                
                color_dist = summary['color_distribution']
                
                # Convert to DataFrame
                color_df = pd.DataFrame([
                    {'Color': color, 'Percentage': pct}
                    for color, pct in color_dist.items()
                ]).sort_values('Percentage', ascending=False)
                
                if not color_df.empty:
                    # Create a color map for the bar chart
                    color_map = {
                        'red': '#FF0000',
                        'orange': '#FFA500',
                        'yellow': '#FFFF00',
                        'green': '#00FF00',
                        'blue': '#0000FF',
                        'purple': '#800080',
                        'pink': '#FFC0CB',
                        'brown': '#A52A2A',
                        'black': '#000000',
                        'white': '#FFFFFF',
                        'gray': '#808080',
                        'cyan': '#00FFFF'
                    }
                    
                    # Get bar colors based on color names
                    bar_colors = color_df['Color'].map(lambda x: color_map.get(x.lower(), '#CCCCCC'))
                    
                    fig = px.bar(
                        color_df,
                        y='Color',
                        x='Percentage',
                        orientation='h',
                        title="Color Distribution in Thumbnails",
                        labels={"Color": "Color Name", "Percentage": "% of Thumbnails"},
                        color='Color',
                        color_discrete_map=dict(zip(color_df['Color'], bar_colors))
                    )
                    st.plotly_chart(fig, use_container_width=True)


def render_color_distribution(color_data):
    """Render color distribution from color data"""
    
    # If we have color data from the DataFrame, display visualizations
    if color_data:
        # Convert to DataFrame for easier manipulation
        color_df = pd.DataFrame(color_data)
        
        # Display color distribution
        st.subheader("Color Distribution")
        
        # Group by color name
        color_counts = color_df['color_name'].value_counts().reset_index()
        color_counts.columns = ['Color', 'Count']
        color_counts['Percentage'] = (color_counts['Count'] / len(color_df)) * 100
        
        # Sort by count
        color_counts = color_counts.sort_values('Count', ascending=False)
        
        # Create a color map for the bar chart
        color_map = {
            'red': '#FF0000',
            'orange': '#FFA500',
            'yellow': '#FFFF00',
            'green': '#00FF00',
            'blue': '#0000FF',
            'purple': '#800080',
            'pink': '#FFC0CB',
            'brown': '#A52A2A',
            'black': '#000000',
            'white': '#FFFFFF',
            'gray': '#808080',
            'cyan': '#00FFFF'
        }
        
        # Get bar colors based on color names
        bar_colors = color_counts['Color'].map(lambda x: color_map.get(x.lower(), '#CCCCCC'))
        
        # Display color distribution
        fig = px.bar(
            color_counts,
            y='Color',
            x='Percentage',
            orientation='h',
            title="Color Distribution in Thumbnails",
            labels={"Color": "Color Name", "Percentage": "% of Thumbnails"},
            color='Color',
            color_discrete_map=dict(zip(color_counts['Color'], bar_colors))
        )
        st.plotly_chart(fig, use_container_width=True)
        
        return color_df, color_counts, color_map
    
    return None, None, None

def render_brightness_contrast(color_df):
    """Render brightness and contrast histograms"""
    
    # Brightness and contrast distribution
    st.subheader("Brightness and Contrast")
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        # Brightness histogram
        fig = px.histogram(
            color_df,
            x='brightness',
            nbins=20,
            title="Brightness Distribution",
            labels={"brightness": "Brightness (0-1)"},
            color_discrete_sequence=[px.colors.qualitative.Set3[0]]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Contrast histogram
        fig = px.histogram(
            color_df,
            x='contrast',
            nbins=20,
            title="Contrast Distribution",
            labels={"contrast": "Contrast (0-1)"},
            color_discrete_sequence=[px.colors.qualitative.Set3[1]]
        )
        st.plotly_chart(fig, use_container_width=True)


def render_color_examples(color_df, color_counts):
    """Render example thumbnails by color"""
    
    # Examples of thumbnails by color
    st.subheader("Examples by Color")
    
    # Get top colors
    top_colors = color_counts['Color'].head(6).tolist()
    
    # Create tabs for top colors
    color_tabs = st.tabs(top_colors)
    
    for i, color in enumerate(top_colors):
        with color_tabs[i]:
            # Filter by color
            color_examples = color_df[color_df['color_name'] == color].sample(min(6, len(color_df[color_df['color_name'] == color])))
            
            # Create a grid of examples
            cols = st.columns(3)
            
            for j, (_, example) in enumerate(color_examples.iterrows()):
                with cols[j % 3]:
                    st.write(f"**Title:** {example['title'][:50]}...")
                    
                    if example['thumbnail_path'] and os.path.exists(example['thumbnail_path']):
                        try:
                            st.image(example['thumbnail_path'], width=200)
                        except Exception as e:
                            st.error(f"Error displaying image: {e}")
                    
                    st.write(f"Brightness: {example['brightness']:.2f}, Contrast: {example['contrast']:.2f}")

def render_color_influence_correlation(color_df, color_map):
    """Render correlation between color and influence"""
    
    # Influence correlation if available
    if 'influence' in color_df.columns and not color_df['influence'].isna().all():
        st.subheader("Color vs. Influence")
        
        # Group by color
        color_influence = color_df.groupby('color_name')['influence'].agg(['mean', 'count']).reset_index()
        
        # Sort by mean influence
        color_influence = color_influence.sort_values('mean', ascending=False)
        
        # Display influence by color
        fig = px.bar(
            color_influence,
            y='color_name',
            x='mean',
            orientation='h',
            title="Average Influence by Dominant Color",
            labels={"color_name": "Color", "mean": "Average Influence"},
            color='color_name',
            color_discrete_map=dict(zip(color_influence['color_name'], 
                                      [color_map.get(c.lower(), '#CCCCCC') for c in color_influence['color_name']])),
            text='count'
        )
        
        fig.update_traces(texttemplate='%{text} videos', textposition='outside')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation with brightness and contrast
        st.subheader("Correlation with Brightness and Contrast")
        
        # Calculate correlations
        brightness_corr = color_df['brightness'].corr(color_df['influence'])
        contrast_corr = color_df['contrast'].corr(color_df['influence'])
        
        # Display correlations
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Brightness-Influence Correlation", f"{brightness_corr:.2f}")
        
        with col2:
            st.metric("Contrast-Influence Correlation", f"{contrast_corr:.2f}")
        
        # Scatter plots
        col1, col2 = st.columns(2)
        
        with col1:
            # Brightness vs. Influence
            fig = px.scatter(
                color_df,
                x='brightness',
                y='influence',
                title="Brightness vs. Influence",
                labels={"brightness": "Brightness", "influence": "Influence"},
                trendline="ols",
                opacity=0.7
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Contrast vs. Influence
            fig = px.scatter(
                color_df,
                x='contrast',
                y='influence',
                title="Contrast vs. Influence",
                labels={"contrast": "Contrast", "influence": "Influence"},
                trendline="ols",
                opacity=0.7
            )
            st.plotly_chart(fig, use_container_width=True)

def render_thumbnail_browser(df: pd.DataFrame, analysis_results: Optional[Dict[str, Any]], 
                           thumbnail_analysis_col: Optional[str], joint_analysis_col: Optional[str],
                           thumbnail_path_col: str):
    """
    Render the thumbnail browser section
    
    Args:
        df: DataFrame with video data
        analysis_results: Analysis results dictionary
        thumbnail_analysis_col: Column containing thumbnail analysis
        joint_analysis_col: Column containing joint analysis
        thumbnail_path_col: Column containing thumbnail path
    """
    st.subheader("Thumbnail Browser")
    
    # Check if we have thumbnail paths
    if thumbnail_path_col not in df.columns:
        st.info("No thumbnail paths found in dataset. Run the analysis with '--thumbnails' to download thumbnails.")
        return
    
    # Filter to thumbnails that exist
    valid_df = df[~df[thumbnail_path_col].isna()].copy()
    
    # Check if there are any valid thumbnails
    if len(valid_df) == 0:
        st.warning("No valid thumbnails found in the dataset.")
        return
    
    # Filter options
    st.write("### Filter Thumbnails")
    
    # Create filter options
    filter_options = {}
    
    # Add community filter if available
    community_col = None
    if 'community' in df.columns:
        community_col = 'community'
    elif 'cluster' in df.columns:
        community_col = 'cluster'
    
    if community_col:
        communities = sorted(valid_df[community_col].unique())
        selected_community = st.selectbox(
            f"Filter by {community_col.title()}", 
            options=["All"] + list(communities),
            index=0
        )
        
        if selected_community != "All":
            valid_df = valid_df[valid_df[community_col] == selected_community]
            filter_options[community_col] = selected_community

def thumbnail_browser_channel_title_filter(valid_df, filter_options):
    """Handle channel and title filters for thumbnail browser"""
    
    # Add channel filter if available
    if 'channel' in valid_df.columns:
        channels = sorted(valid_df['channel'].unique())
        
        if len(channels) > 100:
            # Too many channels, use a text input filter
            channel_filter = st.text_input("Filter by Channel Name")
            
            if channel_filter:
                valid_df = valid_df[valid_df['channel'].str.contains(channel_filter, case=False, na=False)]
                filter_options['channel'] = channel_filter
        else:
            # Manageable number of channels, use a select box
            selected_channel = st.selectbox(
                "Filter by Channel", 
                options=["All"] + list(channels),
                index=0
            )
            
            if selected_channel != "All":
                valid_df = valid_df[valid_df['channel'] == selected_channel]
                filter_options['channel'] = selected_channel
    
    # Add title filter
    title_filter = st.text_input("Filter by Title")
    if title_filter:
        valid_df = valid_df[valid_df['title'].str.contains(title_filter, case=False, na=False)]
        filter_options['title'] = title_filter
        
    return valid_df, filter_options

def thumbnail_browser_visual_filters(valid_df, thumbnail_analysis_col, filter_options):
    """Handle visual feature filters for thumbnail browser"""
    
    has_visual_features = False
    
    if thumbnail_analysis_col and thumbnail_analysis_col in valid_df.columns:
        has_visual_features = True
        
        # Add composition filters
        st.write("Visual Elements:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            has_text_filter = st.radio("Text in Thumbnail", ["Any", "Yes", "No"], index=0)
            
            if has_text_filter != "Any":
                has_text = has_text_filter == "Yes"
                
                # Filter based on composition.has_text
                filtered_df = []
                
                for idx, row in valid_df.iterrows():
                    if pd.isna(row[thumbnail_analysis_col]):
                        continue
                    
                    try:
                        analysis = json.loads(row[thumbnail_analysis_col])
                        
                        if 'composition' in analysis and analysis['composition'].get('has_text', False) == has_text:
                            filtered_df.append(idx)
                    except:
                        continue
                
                if filtered_df:
                    valid_df = valid_df.loc[filtered_df]
                    filter_options['has_text'] = has_text
        
        with col2:
            has_faces_filter = st.radio("Faces in Thumbnail", ["Any", "Yes", "No"], index=0)
            
            if has_faces_filter != "Any":
                has_faces = has_faces_filter == "Yes"
                
                # Filter based on composition.has_faces
                filtered_df = []
                
                for idx, row in valid_df.iterrows():
                    if pd.isna(row[thumbnail_analysis_col]):
                        continue
                    
                    try:
                        analysis = json.loads(row[thumbnail_analysis_col])
                        
                        if 'composition' in analysis and analysis['composition'].get('has_faces', False) == has_faces:
                            filtered_df.append(idx)
                    except:
                        continue
                
                if filtered_df:
                    valid_df = valid_df.loc[filtered_df]
                    filter_options['has_faces'] = has_faces
        
        # Add color filter
        color_options = ["Any"]
        color_counts = {}
        
        # Get all unique color names
        for idx, row in valid_df.iterrows():
            if pd.isna(row[thumbnail_analysis_col]):
                continue
            
            try:
                analysis = json.loads(row[thumbnail_analysis_col])
                
                if 'colors' in analysis and 'dominant' in analysis['colors'] and analysis['colors']['dominant']:
                    color = analysis['colors']['dominant'][0]['name']
                    
                    if color not in color_counts:
                        color_counts[color] = 0
                    
                    color_counts[color] += 1
            except:
                continue
        
        # Add colors to options
        color_options.extend(sorted(color_counts.keys()))
        
        color_filter = st.selectbox("Filter by Dominant Color", options=color_options, index=0)
        
        if color_filter != "Any":
            # Filter based on dominant color
            filtered_df = []
            
            for idx, row in valid_df.iterrows():
                if pd.isna(row[thumbnail_analysis_col]):
                    continue
                
                try:
                    analysis = json.loads(row[thumbnail_analysis_col])
                    
                    if ('colors' in analysis and 'dominant' in analysis['colors'] and 
                        analysis['colors']['dominant'] and analysis['colors']['dominant'][0]['name'] == color_filter):
                        filtered_df.append(idx)
                except:
                    continue
            
            if filtered_df:
                valid_df = valid_df.loc[filtered_df]
                filter_options['color'] = color_filter
                
    return valid_df, filter_options, has_visual_features

def thumbnail_browser_joint_filters(valid_df, joint_analysis_col, has_visual_features, filter_options):
    """Handle joint analysis filters for thumbnail browser"""
    
    # Add joint analysis filters if available
    if joint_analysis_col and joint_analysis_col in valid_df.columns:
        has_visual_features = True
        
        # Add clickbait filter
        st.write("Content Strategy:")
        
        clickbait_options = ["Any", "High Clickbait", "Medium Clickbait", "Low Clickbait"]
        clickbait_filter = st.selectbox("Filter by Clickbait Level", options=clickbait_options, index=0)
        
        if clickbait_filter != "Any":
            # Define clickbait thresholds
            if clickbait_filter == "High Clickbait":
                threshold = 0.6
                is_high = True
            elif clickbait_filter == "Medium Clickbait":
                threshold_low = 0.3
                threshold_high = 0.6
                is_medium = True
            else:  # Low Clickbait
                threshold = 0.3
                is_high = False
            
            # Filter based on clickbait score
            filtered_df = []
            
            for idx, row in valid_df.iterrows():
                if pd.isna(row[joint_analysis_col]):
                    continue
                
                try:
                    analysis = json.loads(row[joint_analysis_col])
                    score = analysis.get('clickbait_score', None)
                    
                    if score is not None:
                        if 'is_medium' in locals():
                            if threshold_low <= score < threshold_high:
                                filtered_df.append(idx)
                        elif (is_high and score >= threshold) or (not is_high and score < threshold):
                            filtered_df.append(idx)
                except:
                    continue
            
            if filtered_df:
                valid_df = valid_df.loc[filtered_df]
                filter_options['clickbait'] = clickbait_filter
        
        # Add alignment filter
        alignment_options = ["Any", "High Alignment", "Medium Alignment", "Low Alignment"]
        alignment_filter = st.selectbox("Filter by Title-Thumbnail Alignment", options=alignment_options, index=0)
        
        if alignment_filter != "Any":
            # Define alignment thresholds
            if alignment_filter == "High Alignment":
                threshold = 0.7
                is_high = True
            elif alignment_filter == "Medium Alignment":
                threshold_low = 0.3
                threshold_high = 0.7
                is_medium = True
            else:  # Low Alignment
                threshold = 0.3
                is_high = False
            
            # Filter based on alignment score
            filtered_df = []
            
            for idx, row in valid_df.iterrows():
                if pd.isna(row[joint_analysis_col]):
                    continue
                
                try:
                    analysis = json.loads(row[joint_analysis_col])
                    score = analysis.get('text_visual_alignment', None)
                    
                    if score is not None:
                        if 'is_medium' in locals():
                            if threshold_low <= score < threshold_high:
                                filtered_df.append(idx)
                        elif (is_high and score >= threshold) or (not is_high and score < threshold):
                            filtered_df.append(idx)
                except:
                    continue
            
            if filtered_df:
                valid_df = valid_df.loc[filtered_df]
                filter_options['alignment'] = alignment_filter
                
    return valid_df, filter_options, has_visual_features

def thumbnail_browser_influence_filter(df, valid_df, filter_options):
    """Handle influence filter for thumbnail browser"""
    
    # Add influence filter if available
    if 'influence' in valid_df.columns:
        st.write("Performance:")
        
        influence_options = ["Any", "High Influence", "Medium Influence", "Low Influence"]
        influence_filter = st.selectbox("Filter by Influence Level", options=influence_options, index=0)
        
        if influence_filter != "Any":
            # Calculate thresholds
            high_threshold = np.percentile(df['influence'], 75)
            low_threshold = np.percentile(df['influence'], 25)
            
            # Filter based on influence level
            if influence_filter == "High Influence":
                valid_df = valid_df[valid_df['influence'] >= high_threshold]
            elif influence_filter == "Medium Influence":
                valid_df = valid_df[(valid_df['influence'] >= low_threshold) & (valid_df['influence'] < high_threshold)]
            else:  # Low Influence
                valid_df = valid_df[valid_df['influence'] < low_threshold]
            
            filter_options['influence'] = influence_filter
            
    return valid_df, filter_options


def thumbnail_browser_display_options(valid_df, filter_options, has_visual_features, 
                                     joint_analysis_col, thumbnail_analysis_col):
    """Handle display options for thumbnail browser"""
    
    # Display filter summary
    if filter_options:
        st.write("Active filters:")
        filter_text = ", ".join([f"{k}: {v}" for k, v in filter_options.items()])
        st.info(filter_text)
    
    # Check if we have any remaining thumbnails
    if len(valid_df) == 0:
        st.warning("No thumbnails match the selected filters.")
        return None
    
    # Display options
    st.write("### Display Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Number of thumbnails to display
        display_count = st.slider("Number of Thumbnails", 10, 100, 30, step=10)
    
    with col2:
        # Sort options
        sort_options = ["Random"]
        
        if 'influence' in valid_df.columns:
            sort_options.extend(["Highest Influence", "Lowest Influence"])
        
        if 'views' in valid_df.columns:
            sort_options.extend(["Most Views", "Least Views"])
        
        if has_visual_features:
            if joint_analysis_col and joint_analysis_col in valid_df.columns:
                sort_options.extend(["Highest Clickbait", "Lowest Clickbait", 
                                   "Highest Alignment", "Lowest Alignment"])
            
            if thumbnail_analysis_col and thumbnail_analysis_col in valid_df.columns:
                sort_options.extend(["Brightest", "Darkest", "Highest Contrast", "Lowest Contrast"])
        
        sort_by = st.selectbox("Sort Thumbnails By", options=sort_options, index=0)
        
    return valid_df, display_count, sort_by

def thumbnail_browser_sort_by_metrics(valid_df, sort_by, display_count):
    """Handle basic metrics sorting for thumbnail browser"""
    
    # Apply sorting
    if sort_by == "Random":
        valid_df = valid_df.sample(min(display_count, len(valid_df)))
    elif sort_by == "Highest Influence":
        valid_df = valid_df.sort_values('influence', ascending=False).head(display_count)
    elif sort_by == "Lowest Influence":
        valid_df = valid_df.sort_values('influence').head(display_count)
    elif sort_by == "Most Views":
        valid_df = valid_df.sort_values('views', ascending=False).head(display_count)
    elif sort_by == "Least Views":
        valid_df = valid_df.sort_values('views').head(display_count)
        
    return valid_df


def thumbnail_browser_sort_by_analysis(valid_df, sort_by, display_count, joint_analysis_col, thumbnail_analysis_col):
    """Handle analysis-based sorting for thumbnail browser"""
    
    if sort_by in ["Highest Clickbait", "Lowest Clickbait", 
                    "Highest Alignment", "Lowest Alignment"]:
        
        # Sort by clickbait or alignment score
        sorted_indices = []
        score_dict = {}
        
        for idx, row in valid_df.iterrows():
            if pd.isna(row[joint_analysis_col]):
                continue
                
            try:
                analysis = json.loads(row[joint_analysis_col])
                
                if "Clickbait" in sort_by:
                    score = analysis.get('clickbait_score', None)
                else:
                    score = analysis.get('text_visual_alignment', None)
                
                if score is not None:
                    score_dict[idx] = score
                    sorted_indices.append(idx)
            except:
                continue
        
        if sorted_indices:
            # Sort by score
            if "Highest" in sort_by:
                sorted_indices = sorted(sorted_indices, key=lambda x: score_dict[x], reverse=True)
            else:
                sorted_indices = sorted(sorted_indices, key=lambda x: score_dict[x])
            
            # Get display count
            sorted_indices = sorted_indices[:display_count]
            
            # Filter DataFrame
            valid_df = valid_df.loc[sorted_indices]
            
    elif sort_by in ["Brightest", "Darkest", "Highest Contrast", "Lowest Contrast"]:
        # Sort by brightness or contrast
        sorted_indices = []
        value_dict = {}
        
        for idx, row in valid_df.iterrows():
            if pd.isna(row[thumbnail_analysis_col]):
                continue
                
            try:
                analysis = json.loads(row[thumbnail_analysis_col])
                
                if 'colors' in analysis:
                    colors = analysis['colors']
                    
                    if "Brightest" in sort_by or "Darkest" in sort_by:
                        value = colors.get('brightness', None)
                    else:
                        value = colors.get('contrast', None)
                    
                    if value is not None:
                        value_dict[idx] = value
                        sorted_indices.append(idx)
            except:
                continue
        
        if sorted_indices:
            # Sort by value
            if "Brightest" in sort_by or "Highest" in sort_by:
                sorted_indices = sorted(sorted_indices, key=lambda x: value_dict[x], reverse=True)
            else:
                sorted_indices = sorted(sorted_indices, key=lambda x: value_dict[x])
            
            # Get display count
            sorted_indices = sorted_indices[:display_count]
            
            # Filter DataFrame
            valid_df = valid_df.loc[sorted_indices]
            
    return valid_df

def display_thumbnails(valid_df, display_count, thumbnail_path_col, thumbnail_analysis_col, joint_analysis_col):
    """Display the filtered and sorted thumbnails"""
    
    # Display the thumbnails
    st.write(f"### Showing {len(valid_df)} Thumbnails")
    
    # Calculate number of columns based on display count
    num_cols = 3 if display_count <= 30 else 4
    
    # Create a grid of thumbnails
    cols = st.columns(num_cols)
    
    # Display thumbnails
    for i, (idx, row) in enumerate(valid_df.iterrows()):
        with cols[i % num_cols]:
            # Display thumbnail image
            if row[thumbnail_path_col] and os.path.exists(row[thumbnail_path_col]):
                try:
                    st.image(row[thumbnail_path_col], use_column_width=True)
                except Exception as e:
                    st.error(f"Error displaying image: {e}")
            
            # Display video title
            st.write(f"**{row['title'][:50]}{'...' if len(row['title']) > 50 else ''}**")
            
            # Display additional info
            info_text = []
            
            if 'channel' in row:
                info_text.append(f"Channel: {row['channel']}")
            
            if 'influence' in row and not pd.isna(row['influence']):
                info_text.append(f"Influence: {row['influence']:.2f}")
            
            if 'views' in row and not pd.isna(row['views']):
                info_text.append(f"Views: {row['views']:,}")
            
            # Display any analysis info
            if thumbnail_analysis_col and thumbnail_analysis_col in row and not pd.isna(row[thumbnail_analysis_col]):
                try:
                    analysis = json.loads(row[thumbnail_analysis_col])
                    
                    if 'composition' in analysis:
                        comp = analysis['composition']
                        comp_text = []
                        
                        if comp.get('has_text', False):
                            comp_text.append("Has Text")
                        
                        if comp.get('has_faces', False):
                            comp_text.append("Has Faces")
                        
                        if comp_text:
                            info_text.append(f"Composition: {', '.join(comp_text)}")
                    
                    if 'colors' in analysis and 'dominant' in analysis['colors'] and analysis['colors']['dominant']:
                        top_color = analysis['colors']['dominant'][0]
                        info_text.append(f"Color: {top_color.get('name', 'unknown')}")
                        
                        if 'brightness' in analysis['colors']:
                            info_text.append(f"Brightness: {analysis['colors']['brightness']:.2f}")
                except:
                    pass
            
            if joint_analysis_col and joint_analysis_col in row and not pd.isna(row[joint_analysis_col]):
                try:
                    analysis = json.loads(row[joint_analysis_col])
                    
                    if 'clickbait_score' in analysis:
                        info_text.append(f"Clickbait: {analysis['clickbait_score']:.2f}")
                    
                    if 'text_visual_alignment' in analysis:
                        info_text.append(f"Alignment: {analysis['text_visual_alignment']:.2f}")
                except:
                    pass
            
            # Display info text
            for text in info_text:
                st.write(text)
            
            # Add a separator between thumbnails
            st.markdown("---")
    
    # Add pagination if needed
    if len(valid_df) >= display_count:
        st.write("Note: Showing the first", display_count, "thumbnails matching your filters.")
        st.write("Adjust the filters or sort options to see different thumbnails.")