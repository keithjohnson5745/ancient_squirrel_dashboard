import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional

def render(df: pd.DataFrame, analysis_results: Optional[Dict[str, Any]] = None):
    """
    Render the network visualization tab
    
    Args:
        df: DataFrame with video data
        analysis_results: Optional analysis results dictionary
    """
    st.header("Network Visualization")
    
    community_col = None
    if 'community' in df.columns:
        community_col = 'community'
    elif 'cluster' in df.columns:
        community_col = 'cluster'
    
    if community_col is None:
        st.warning("No community or cluster column found in the data. Network visualization requires community assignments.")
        return
    
    try:
        # Get minimum community size from session state or use default
        min_community_size = st.session_state.get('min_community_size', 5)
        
        # Filter communities by minimum size
        community_sizes = df[community_col].value_counts()
        valid_communities = community_sizes[community_sizes >= min_community_size].index
        
        if len(valid_communities) == 0:
            st.warning(f"No {community_col}s found with at least {min_community_size} videos.")
            return
        
        filtered_df = df[df[community_col].isin(valid_communities)]
        
        st.write(f"Showing {len(valid_communities)} {community_col}s with at least {min_community_size} videos")
        
        # Create a graph with communities as nodes
        G = nx.Graph()
        
        # Add community nodes
        for community in valid_communities:
            community_df = filtered_df[filtered_df[community_col] == community]
            size = len(community_df)
            
            # Calculate influence if available
            influence = 1.0
            if 'influence' in community_df.columns:
                influence = community_df['influence'].mean()
            
            G.add_node(f"C{community}", 
                    size=size, 
                    influence=influence,
                    type='community')
        
        # Add edges between communities based on shared channels
        if 'channel' in filtered_df.columns:
            communities = list(valid_communities)
            for i, c1 in enumerate(communities):
                c1_channels = set(filtered_df[filtered_df[community_col] == c1]['channel'])
                
                for j in range(i+1, len(communities)):
                    c2 = communities[j]
                    c2_channels = set(filtered_df[filtered_df[community_col] == c2]['channel'])
                    
                    shared = c1_channels.intersection(c2_channels)
                    if shared:
                        G.add_edge(f"C{c1}", f"C{c2}", weight=len(shared))
        
        # Set node sizes based on community size
        node_sizes = [G.nodes[node]['size'] * 5 for node in G.nodes]
        
        # Set node colors based on average influence
        node_colors = [G.nodes[node]['influence'] for node in G.nodes]
        
        # Create network visualization
        fig, ax = plt.subplots(figsize=(12, 10))
        pos = nx.spring_layout(G, seed=42, k=0.3)
        
        nx.draw_networkx_nodes(G, pos, 
                            node_size=node_sizes,
                            node_color=node_colors, 
                            cmap=st.session_state.get('theme_color_map', 'viridis'),
                            alpha=0.8)
        
        nx.draw_networkx_edges(G, pos, 
                            width=[G[u][v]['weight'] * 0.2 for u, v in G.edges],
                            alpha=0.5)
        
        nx.draw_networkx_labels(G, pos)
        
        plt.title(f"{community_col.title()} Network by Shared Channels")
        plt.axis('off')
        
        # Display network
        st.pyplot(fig)
        
        # Community connections table
        if len(G.edges) > 0:
            st.subheader(f"{community_col.title()} Connections")
            
            edge_data = []
            for u, v, data in G.edges(data=True):
                edge_data.append({
                    f"{community_col.title()} 1": u.replace("C", ""),
                    f"{community_col.title()} 2": v.replace("C", ""),
                    "Shared Channels": data['weight']
                })
            
            edge_df = pd.DataFrame(edge_data).sort_values('Shared Channels', ascending=False)
            st.dataframe(edge_df, hide_index=True)
        
        # Community stats
        st.subheader(f"{community_col.title()} Statistics")
        
        community_stats = []
        for community in valid_communities:
            community_df = filtered_df[filtered_df[community_col] == community]
            
            stats = {
                f"{community_col.title()}": community,
                "Videos": len(community_df),
                "Channels": community_df['channel'].nunique() if 'channel' in community_df.columns else 0
            }
            
            if 'channel' in community_df.columns:
                stats["Top Channel"] = community_df['channel'].value_counts().idxmax()
            
            if 'influence' in community_df.columns:
                stats["Avg Influence"] = community_df['influence'].mean()
                stats["Max Influence"] = community_df['influence'].max()
            
            community_stats.append(stats)
        
        stats_df = pd.DataFrame(community_stats).sort_values('Videos', ascending=False)
        st.dataframe(stats_df, hide_index=True)
    
    except Exception as e:
        st.error(f"Error in network visualization: {str(e)}")