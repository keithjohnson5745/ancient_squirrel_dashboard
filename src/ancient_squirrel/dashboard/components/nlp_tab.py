import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional

def render(df: pd.DataFrame, analysis_results: Optional[Dict[str, Any]] = None, 
           nlp_results: Optional[Dict[str, Any]] = None):
    """
    Render the NLP analysis tab
    
    Args:
        df: DataFrame with video data
        analysis_results: Optional analysis results dictionary
        nlp_results: Optional NLP analysis results
    """
    st.header("Natural Language Analysis")
    
    # Check if NLP data columns exist
    nlp_data_available = (
        'nmf_topic' in df.columns or 
        'is_question' in df.columns or
        (nlp_results is not None)
    )
    
    if not nlp_data_available:
        st.warning("""
        No NLP analysis data found. Run the analysis with NLP features enabled:
        ```
        ancient-analyze --input data/youtube_videos.csv --nlp --topics 20
        ```
        """)
        return
    
    # Create sub-tabs for different NLP analyses
    nlp_tabs = st.tabs([
        "Topic Modeling", 
        "Linguistic Patterns", 
        "Title Insights"
    ])
    
    # Topic Modeling tab
    with nlp_tabs[0]:
        st.subheader("Topic Modeling Results")
        
        # Check for NMF topic column in dataframe
        if 'nmf_topic' in df.columns:
            st.write("### Topic Distribution")
            
            topic_counts = df['nmf_topic'].value_counts().reset_index()
            topic_counts.columns = ['Topic', 'Count']
            topic_counts = topic_counts.sort_values('Count', ascending=False)
            
            fig = px.bar(
                topic_counts.head(20),
                x='Topic',
                y='Count',
                title="Videos per Topic",
                color='Count',
                color_continuous_scale=st.session_state.get('theme_color_map', 'viridis')
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show example titles by topic
            st.subheader("Example Titles by Topic")
            
            topic_to_view = st.selectbox(
                "Select topic to view examples",
                options=sorted(df['nmf_topic'].unique()),
                format_func=lambda x: f"Topic {x}"
            )
            
            if topic_to_view is not None:
                topic_df = df[df['nmf_topic'] == topic_to_view]
                
                st.write(f"### Example titles in Topic {topic_to_view}")
                st.dataframe(
                    topic_df[['title', 'channel']].head(10),
                    use_container_width=True,
                    hide_index=True
                )
        
        # Check if NMF topic results are available in analysis results
        nlp_topic_data = None
        
        # First check in analysis_results
        if analysis_results and 'nlp_analysis' in analysis_results:
            # Try to get from analysis summary
            nlp_summary = analysis_results['nlp_analysis'].get('summary', {})
            if nlp_summary.get('results_file'):
                st.info(f"Detailed NLP results available in: {nlp_summary.get('results_file')}")
        
        # Then check in dedicated nlp_results
        if nlp_results and 'topic_modeling' in nlp_results and 'nmf' in nlp_results['topic_modeling']:
            nlp_topic_data = nlp_results['topic_modeling']['nmf']
        
        if nlp_topic_data:
            st.write("### Topic Term Analysis")
            
            # Display topic terms
            if 'topic_terms' in nlp_topic_data:
                topic_terms = nlp_topic_data['topic_terms']
                
                # Topic selector
                if isinstance(topic_terms, list):
                    selected_topic = st.selectbox(
                        "Select topic to explore terms",
                        options=range(len(topic_terms)),
                        format_func=lambda x: f"Topic {x}",
                        key="topic_terms_selector"
                    )
                    
                    if selected_topic is not None and selected_topic < len(topic_terms):
                        # Display topic terms
                        st.write(f"### Top terms in Topic {selected_topic}")
                        
                        # Get the terms for this topic
                        topic_term_data = topic_terms[selected_topic]
                        
                        try:
                            # Convert to DataFrame for better display
                            term_data = []
                            
                            # Handle different possible formats of topic_term_data
                            if isinstance(topic_term_data, list):
                                for item in topic_term_data:
                                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                                        term, weight = item
                                        term_data.append({"Term": term, "Weight": weight})
                                    elif isinstance(item, dict) and 'term' in item and 'weight' in item:
                                        term_data.append({"Term": item['term'], "Weight": item['weight']})
                            elif isinstance(topic_term_data, dict):
                                for term, weight in topic_term_data.items():
                                    term_data.append({"Term": term, "Weight": weight})
                            
                            if term_data:
                                terms_df = pd.DataFrame(term_data)
                                
                                # Bar chart
                                fig = px.bar(
                                    terms_df,
                                    y='Term',
                                    x='Weight',
                                    orientation='h',
                                    title=f"Key Terms in Topic {selected_topic}",
                                    color='Weight',
                                    color_continuous_scale=st.session_state.get('theme_color_map', 'viridis')
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning(f"Could not extract term data from topic {selected_topic}")
                        except Exception as e:
                            st.error(f"Error processing topic terms: {str(e)}")
        
        if not ('nmf_topic' in df.columns or nlp_topic_data):
            st.info("No topic modeling data found. Run the processor with the NLP analysis option to generate topic models.")
    
    # Linguistic Patterns tab
    with nlp_tabs[1]:
        st.subheader("Linguistic Pattern Analysis")
        
        # Check for linguistic pattern columns
        patterns_available = 'is_question' in df.columns or 'is_imperative' in df.columns
        
        # Check for linguistic analysis in results
        linguistic_results = None
        if nlp_results and 'linguistic_analysis' in nlp_results:
            linguistic_results = nlp_results['linguistic_analysis']
        elif analysis_results and 'nlp_analysis' in analysis_results:
            nlp_data = analysis_results['nlp_analysis']
            if 'linguistic_analysis' in nlp_data:
                linguistic_results = nlp_data['linguistic_analysis']
        
        if patterns_available or linguistic_results is not None:
            # Display basic metrics
            col1, col2 = st.columns(2)
            
            with col1:
                question_pct = 0
                if 'is_question' in df.columns:
                    question_pct = df['is_question'].mean() * 100
                elif linguistic_results and 'question_pct' in linguistic_results:
                    question_pct = linguistic_results['question_pct']
                
                st.metric("Question Titles", f"{question_pct:.1f}%")
            
            with col2:
                imperative_pct = 0
                if 'is_imperative' in df.columns:
                    imperative_pct = df['is_imperative'].mean() * 100
                elif linguistic_results and 'imperative_pct' in linguistic_results:
                    imperative_pct = linguistic_results['imperative_pct']
                
                st.metric("Imperative Titles", f"{imperative_pct:.1f}%")
            
            # Show examples if the relevant columns exist
            if 'is_question' in df.columns:
                st.subheader("Question Title Examples")
                try:
                    question_df = df[df['is_question'] == True].head(10)
                    st.dataframe(question_df[['title', 'channel']], 
                                use_container_width=True, 
                                hide_index=True)
                except Exception as e:
                    st.error(f"Error displaying question titles: {str(e)}")
            
            if 'is_imperative' in df.columns:
                st.subheader("Imperative Title Examples")
                try:
                    imperative_df = df[df['is_imperative'] == True].head(10)
                    st.dataframe(imperative_df[['title', 'channel']], 
                                use_container_width=True,
                                hide_index=True)
                except Exception as e:
                    st.error(f"Error displaying imperative titles: {str(e)}")
            
            # Display pattern statistics
            if linguistic_results and 'pattern_counts' in linguistic_results:
                st.subheader("Title Pattern Distribution")
                
                pattern_data = []
                # Check if pattern_counts is a dict before proceeding
                if isinstance(linguistic_results['pattern_counts'], dict):
                    for pattern, count in linguistic_results['pattern_counts'].items():
                        pattern_data.append({
                            "Pattern": pattern,
                            "Count": count,
                            "Percentage": (count / df.shape[0]) * 100
                        })
                    
                    # Only create and display DataFrame if we have data
                    if pattern_data:
                        pattern_df = pd.DataFrame(pattern_data).sort_values('Count', ascending=False)
                        
                        fig = px.bar(
                            pattern_df,
                            y='Pattern',
                            x='Percentage',
                            orientation='h',
                            title="Title Pattern Distribution",
                            labels={"Pattern": "Pattern", "Percentage": "% of Titles"},
                            color='Percentage',
                            color_continuous_scale=st.session_state.get('theme_color_map', 'viridis')
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No pattern data available")
                else:
                    st.info("Pattern counts data is not in the expected format")
    
    # Title Insights tab
    with nlp_tabs[2]:
        st.subheader("Title Insights")
        
        # Check for LLM insights in different places
        llm_insights = None
        if nlp_results and 'llm_insights' in nlp_results:
            llm_insights = nlp_results['llm_insights']
        elif analysis_results and 'nlp_analysis' in analysis_results:
            nlp_data = analysis_results['nlp_analysis']
            if 'llm_insights' in nlp_data:
                llm_insights = nlp_data['llm_insights']
        
        if llm_insights:
            # Title patterns
            if 'title_patterns' in llm_insights:
                pattern_analysis = llm_insights['title_patterns']
                
                st.subheader("Title Pattern Analysis")
                
                # Display LLM analysis
                if 'pattern_analysis' in pattern_analysis:
                    st.markdown(pattern_analysis['pattern_analysis'])
                    
                    # Sample info
                    sample_size = pattern_analysis.get('sample_size', 0)
                    st.info(f"Analysis based on a sample of {sample_size} titles")
            else:
                st.info("No title pattern analysis found. Run the processor with the '--llm' flag to generate these insights.")
        else:
            st.info("No LLM insights found. Run the processor with the '--llm' flag to generate title insights.")