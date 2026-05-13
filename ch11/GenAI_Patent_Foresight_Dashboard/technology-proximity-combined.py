import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import numpy as np

def render_scientific_dual_network():
    st.set_page_config(layout="wide")
    st.write("### 🕸️ CPC CO-OCCURRENCE NETWORK ANALYSIS")
    
    # 1. Data Preparation
    file_path = "generative-ai-patent.csv"
    try:
        df = pd.read_csv(file_path)
        df.columns = [c.lower().replace(' ', '_').replace('.', '_') for c in df.columns]
    except Exception:
        # Realistic Mock Data for Demonstration
        df = pd.DataFrame({
            'lens_id': np.repeat(range(1, 500), 3),
            'publication_year': np.random.randint(2010, 2026, 1497),
            'cpc_classifications': ['G06N;G06F;G06Q', 'G16H;G06V', 'C10L;H01B', 'C08J;H02M', 'H10N;G06N'] * 299 + ['G06N'] * 2
        })

    def build_publication_style_network(data_subset, weight_threshold, title, is_filtered=False):
        # Extract CPC Groups
        rows = []
        target_groups = {"G06N", "G06Q", "G06F", "G16H", "G06V"}
        
        for _, row in data_subset.iterrows():
            cpcs = str(row['cpc_classifications']).split(';')
            for cpc in cpcs:
                group = cpc.strip()[:4]
                if group and (not is_filtered or group in target_groups):
                    rows.append({'lens_id': row['lens_id'], 'cpc_group': group})
        
        clean_df = pd.DataFrame(rows).drop_duplicates()
        merged = pd.merge(clean_df, clean_df, on='lens_id')
        edges = merged[merged['cpc_group_x'] < merged['cpc_group_y']]
        edge_list = edges.groupby(['cpc_group_x', 'cpc_group_y']).size().reset_index(name='weight')
        edge_list = edge_list[edge_list['weight'] > weight_threshold]

        # NetworkX Graph Structure
        G = nx.Graph()
        for _, r in edge_list.iterrows():
            G.add_edge(r['cpc_group_x'], r['cpc_group_y'], weight=r['weight'])
        
        # Stress/Spring Layout
        pos = nx.spring_layout(G, seed=42, k=1.5 if is_filtered else 0.5)
        
        fig = go.Figure()

        # Edge Drawing
        for u, v, d in G.edges(data=True):
            w = d['weight']
            width_scaled = (w / 2500) * 2 if not is_filtered else (w / 1000) * 5
            
            fig.add_trace(go.Scatter(
                x=[pos[u][0], pos[v][0], None], y=[pos[u][1], pos[v][1], None],
                mode='lines',
                line=dict(width=width_scaled, color='rgba(220, 220, 220, 0.6)'),
                showlegend=False, hoverinfo='skip'
            ))

        # Node Drawing
        node_x = [pos[n][0] for n in G.nodes()]
        node_y = [pos[n][1] for n in G.nodes()]
        
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=list(G.nodes()),
            textposition="top center",
            marker=dict(size=8 if not is_filtered else 12, color='darkorange', line=dict(width=0.5, color='white')),
            textfont=dict(size=10, family="Arial Black"),
            name='CPC Groups'
        ))

        # Responsive Layout Adjustments
        fig.update_layout(
            template="plotly_white",
            title={
                'text': f"<b>{title}</b>",
                'x': 0.5,
                'xanchor': 'center',
                'y': 0.97,
                'font': {'size': 14 if is_filtered else 22, 'color': "black"}
            },
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600,
            margin=dict(l=10, r=10, t=100, b=10) # Higher top margin for the long title
        )
        return fig

    # --- Side-by-Side View ---
    col1, col2 = st.columns([1.1, 1])

    with col1:
        # A: Full Map
        fig_a = build_publication_style_network(df, 1, "Panel A: Global CPC Landscape", is_filtered=False)
        st.plotly_chart(fig_a, use_container_width=True)

    with col2:
        # B: Strategic Map
        target_years = df[(df['publication_year'] >= 2015) & (df['publication_year'] <= 2025)]
        long_title = "Filtered for Computing (G06F/N) and Strategic Adoption (G06Q)"
        fig_b = build_publication_style_network(target_years, 5, long_title, is_filtered=True)
        st.plotly_chart(fig_b, width='stretch')

if __name__ == "__main__":
    render_scientific_dual_network()