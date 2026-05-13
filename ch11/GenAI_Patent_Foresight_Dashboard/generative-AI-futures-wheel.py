import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go

def render_fixed_futures_wheel():
    st.write("### 🎡 FUTURES WHEEL: GENAI ADOPTION 2028")

    # 1. futures_edges
    edges_data = [
        ("GenAI Interface 2028", "Democratization"), ("GenAI Interface 2028", "LLM Integration"),
        ("GenAI Interface 2028", "Productivity Gains"), ("GenAI Interface 2028", "Data Center Demand"),
        ("GenAI Interface 2028", "Regulatory Frameworks"), ("Democratization", "Expertise Shift"),
        ("Expertise Shift", "Authenticity Crisis"), ("LLM Integration", "Agentic AI"),
        ("Agentic AI", "Model Collapse"), ("Productivity Gains", "Labor Restructuring"),
        ("Labor Restructuring", "Wealth Concentration"), ("Data Center Demand", "High Resource Use"),
        ("High Resource Use", "Decarbonization Conflict"), ("Regulatory Frameworks", "IP Litigation"),
        ("IP Litigation", "Compute Wars")
    ]

    # 2. NetworkX Graph
    G = nx.DiGraph()
    G.add_edges_from(edges_data)

    # 3. CircularLayout
    levels = {
        "GenAI Interface 2028": 0,
        "Democratization": 1, "LLM Integration": 1, "Productivity Gains": 1, "Data Center Demand": 1, "Regulatory Frameworks": 1,
        "Expertise Shift": 2, "Agentic AI": 2, "Labor Restructuring": 2, "High Resource Use": 2, "IP Litigation": 2,
        "Authenticity Crisis": 3, "Model Collapse": 3, "Wealth Concentration": 3, "Decarbonization Conflict": 3, "Compute Wars": 3
    }

    pos = {}
    for node, level in levels.items():
        nodes_at_level = [n for n, l in levels.items() if l == level]
        idx = nodes_at_level.index(node)
        angle = (2 * np.pi * idx) / len(nodes_at_level)
        radius = level * 1.5  
        pos[node] = (radius * np.cos(angle), radius * np.sin(angle))

    # 4. Color and Size Mapping
    color_map = {0: "gold", 1: "deepskyblue", 2: "springgreen", 3: "tomato"}
    size_map = {0: 45, 1: 30, 2: 25, 3: 20}
    
    # Legend Etiketleri
    label_map = {
        0: "Central Trend",
        1: "1st Order Effects",
        2: "2nd Order Effects",
        3: "3rd Order Effects"
    }

    fig = go.Figure()

    # Edges
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        fig.add_trace(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            # Grouping edges under one legend item or hiding them
            legendgroup="Edges",
            name="Impact Paths",
            mode='lines',
            line=dict(width=1, color='#D3D3D3'),
            hoverinfo='skip',
            showlegend=False 
        ))

    # Nodes
    for level_idx in range(4):
        level_nodes = [n for n, l in levels.items() if l == level_idx]
        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in level_nodes],
            y=[pos[n][1] for n in level_nodes],
            mode='markers+text',
            name=label_map[level_idx], # Using our descriptive labels
            text=level_nodes,
            textposition="top center",
            textfont=dict(size=10, color="black", family="Arial Black"),
            marker=dict(
                size=size_map[level_idx],
                color=color_map[level_idx],
                line=dict(width=1, color='gray')
            ),
            hovertemplate="<b>%{text}</b><br>Level: " + str(level_idx) + "<extra></extra>"
        ))

    # 5. Graph Layout (Light Theme)
    fig.update_layout(
        template="plotly_white",
        title=dict(text="Fixed Futures Wheel: GenAI Adoption 2028", x=0.5, font=dict(size=18)),
        showlegend=True,
        # Enhanced Legend Styling
        legend=dict(
            title="<b>Impact Hierarchy</b>",
            orientation="v", # Vertical usually looks cleaner for categorizing levels
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255, 255, 255, 0.5)",
            bordercolor="Gray",
            borderwidth=1
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=750,
        margin=dict(l=40, r=150, t=80, b=40) # Increased right margin for the legend
    )

    st.plotly_chart(fig, use_container_width=True)

render_fixed_futures_wheel()