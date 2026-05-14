"""
Network Analysis Module
Chapter 12: Data-Led Technology Roadmapping

This module provides functions for building and analyzing
keyword co-occurrence networks to reveal technology relationships.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations
from collections import Counter


def create_cooccurrence_network(df: pd.DataFrame,
                                keyword_column: str = 'Keywords',
                                min_cooccurrence: int = 3,
                                separator: str = ';') -> nx.Graph:
    """
    Create keyword co-occurrence network.
    
    Parameters:
        df: DataFrame with keyword column
        keyword_column: Name of column containing keywords
        min_cooccurrence: Minimum co-occurrence threshold
        separator: Keyword separator character
        
    Returns:
        NetworkX Graph with keywords as nodes and co-occurrences as edges
    """
    # Extract keyword pairs
    pairs = []
    
    for keywords in df[keyword_column].dropna():
        # Split and clean keywords
        kw_list = [k.strip().lower() for k in str(keywords).split(separator)]
        kw_list = [k for k in kw_list if len(k) > 2]  # Filter short terms
        
        if len(kw_list) > 1:
            # Create all pairwise combinations
            pairs.extend(list(combinations(sorted(kw_list), 2)))
    
    # Count co-occurrences
    pair_counts = Counter(pairs)
    
    # Create network
    G = nx.Graph()
    
    for (kw1, kw2), count in pair_counts.items():
        if count >= min_cooccurrence:
            G.add_edge(kw1, kw2, weight=count)
    
    print(f"Network created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"(min_cooccurrence threshold: {min_cooccurrence})")
    
    return G


def visualize_network(G: nx.Graph, 
                      title: str = 'Keyword Co-occurrence Network',
                      node_scale: int = 100,
                      save_path: str = 'network_visualization.png'):
    """
    Visualize co-occurrence network with node sizing by degree.
    
    Parameters:
        G: NetworkX Graph
        title: Plot title
        node_scale: Scaling factor for node sizes
        save_path: Path to save figure
    """
    if G.number_of_nodes() == 0:
        print("Empty network - nothing to visualize")
        return
    
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Layout algorithm
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    
    # Node sizes based on degree
    degrees = dict(G.degree())
    node_sizes = [degrees[node] * node_scale for node in G.nodes()]
    
    # Edge widths based on weight
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [w / max_weight * 3 for w in edge_weights]
    
    # Draw network
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                           node_color='lightblue', alpha=0.7, ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_widths, 
                           alpha=0.3, edge_color='gray', ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return pos


def analyze_network_centrality(G: nx.Graph, top_n: int = 10) -> pd.DataFrame:
    """
    Calculate and compare centrality measures for network nodes.
    
    Parameters:
        G: NetworkX Graph
        top_n: Number of top nodes to display
        
    Returns:
        DataFrame with degree, betweenness, and eigenvector centrality
    """
    if G.number_of_nodes() == 0:
        print("Empty network - no centrality to calculate")
        return pd.DataFrame()
    
    # Calculate centrality measures
    degree_cent = nx.degree_centrality(G)
    betweenness_cent = nx.betweenness_centrality(G)
    
    try:
        eigenvector_cent = nx.eigenvector_centrality(G, max_iter=1000)
    except:
        eigenvector_cent = {node: 0 for node in G.nodes()}
        print("Warning: Eigenvector centrality calculation failed")
    
    # Create comparison dataframe
    centrality_df = pd.DataFrame({
        'Keyword': list(G.nodes()),
        'Degree': [degree_cent[n] for n in G.nodes()],
        'Betweenness': [betweenness_cent[n] for n in G.nodes()],
        'Eigenvector': [eigenvector_cent[n] for n in G.nodes()],
        'Connections': [G.degree(n) for n in G.nodes()]
    })
    
    # Sort by degree centrality
    centrality_df = centrality_df.sort_values('Degree', ascending=False)
    
    # Print top keywords
    print(f"\n{'='*70}")
    print(f"Top {top_n} Keywords by Centrality Measures")
    print(f"{'='*70}")
    print(centrality_df.head(top_n).to_string(index=False))
    
    return centrality_df


def detect_communities(G: nx.Graph) -> dict:
    """
    Detect communities in network using greedy modularity optimization.
    
    Parameters:
        G: NetworkX Graph
        
    Returns:
        Dictionary mapping nodes to community IDs
    """
    if G.number_of_nodes() == 0:
        print("Empty network - no communities to detect")
        return {}
    
    # Detect communities using greedy modularity
    from networkx.algorithms import community
    communities = community.greedy_modularity_communities(G)
    
    # Create node-to-community mapping
    node_community = {}
    for idx, comm in enumerate(communities):
        for node in comm:
            node_community[node] = idx
    
    # Print community summary
    print(f"\n{'='*60}")
    print(f"Community Detection Results")
    print(f"{'='*60}")
    print(f"Detected {len(communities)} communities:\n")
    
    for idx, comm in enumerate(communities):
        comm_list = list(comm)
        print(f"Community {idx+1} ({len(comm)} members):")
        print(f"  Top members: {', '.join(comm_list[:5])}")
        if len(comm_list) > 5:
            print(f"  ... and {len(comm_list)-5} more")
        print()
    
    return node_community


def visualize_communities(G: nx.Graph, 
                          node_community: dict,
                          save_path: str = 'network_communities.png'):
    """
    Visualize network with community coloring.
    
    Parameters:
        G: NetworkX Graph
        node_community: Dictionary mapping nodes to community IDs
        save_path: Path to save figure
    """
    if G.number_of_nodes() == 0:
        return
    
    fig, ax = plt.subplots(figsize=(16, 12))
    
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    
    # Color by community
    colors = [node_community.get(node, 0) for node in G.nodes()]
    
    # Node sizes by degree
    node_sizes = [G.degree(node) * 100 for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                           node_color=colors, cmap=plt.cm.tab10,
                           alpha=0.7, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.2, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)
    
    ax.set_title('Network Communities', fontsize=16, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def get_network_statistics(G: nx.Graph) -> dict:
    """
    Calculate comprehensive network statistics.
    
    Parameters:
        G: NetworkX Graph
        
    Returns:
        Dictionary with network metrics
    """
    if G.number_of_nodes() == 0:
        return {'nodes': 0, 'edges': 0}
    
    stats = {
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'density': nx.density(G),
        'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes(),
        'avg_clustering': nx.average_clustering(G),
    }
    
    # Check if connected
    if nx.is_connected(G):
        stats['avg_path_length'] = nx.average_shortest_path_length(G)
        stats['diameter'] = nx.diameter(G)
    else:
        # Use largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
        subgraph = G.subgraph(largest_cc)
        stats['avg_path_length'] = nx.average_shortest_path_length(subgraph)
        stats['diameter'] = nx.diameter(subgraph)
        stats['num_components'] = nx.number_connected_components(G)
    
    print(f"\n{'='*50}")
    print("Network Statistics")
    print(f"{'='*50}")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    return stats


def find_bridge_keywords(G: nx.Graph, top_n: int = 10) -> list:
    """
    Find keywords that bridge different communities (high betweenness).
    
    Parameters:
        G: NetworkX Graph
        top_n: Number of bridge keywords to return
        
    Returns:
        List of (keyword, betweenness_score) tuples
    """
    betweenness = nx.betweenness_centrality(G)
    sorted_nodes = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nTop {top_n} Bridge Keywords (High Betweenness):")
    for keyword, score in sorted_nodes[:top_n]:
        print(f"  {keyword}: {score:.4f}")
    
    return sorted_nodes[:top_n]


# Example usage
if __name__ == "__main__":
    print("Network Analysis Module - Chapter 12")
    print("Functions: create_cooccurrence_network(), analyze_network_centrality(), detect_communities()")
    print("See README.md for complete documentation")
