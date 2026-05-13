import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def render_extended_strategic_cone():
    st.write("### 🗼 GENERATIVE AI FUTURES CONE (EXTENDED)")

    fig = go.Figure()

    # 1. Cone Geometry (Possible, Plausible, Probable)
    zones = [
        {"name": "Possible", "x": [2026, 2040, 2040, 2026], "y": [0, 100, -100, 0], "color": "rgba(69, 123, 157, 0.15)"},
        {"name": "Plausible", "x": [2026, 2040, 2040, 2026], "y": [0, 60, -60, 0], "color": "rgba(168, 218, 220, 0.35)"},
        {"name": "Probable", "x": [2026, 2040, 2040, 2026], "y": [0, 25, -25, 0], "color": "rgba(29, 53, 87, 0.45)"}
    ]

    for zone in zones:
        fig.add_trace(go.Scatter(
            x=zone["x"], y=zone["y"],
            fill="toself",
            fillcolor=zone["color"],
            line=dict(color='rgba(255,255,255,0)'),
            name=zone["name"],
            hoverinfo='skip'
        ))

    # 2. Milestones DataFrame
    milestones = pd.DataFrame([
        dict(year=2030, y_pos=15, label="Agentic AI Standard"),
        dict(year=2030, y_pos=-45, label="AI-Generated Full Movies"),
        dict(year=2035, y_pos=5, label="NPU-Only Hardware"),
        dict(year=2035, y_pos=80, label="Personalized AI Medicine"),
        dict(year=2040, y_pos=-15, label="Physical AI Elderly Care"),
        dict(year=2040, y_pos=50, label="Bespoke Protein Design") 
    ])

    # Milestones Scatter
    fig.add_trace(go.Scatter(
        x=milestones["year"],
        y=milestones["y_pos"],
        mode="markers+text",
        text=milestones["label"],
        textposition=["top center", "bottom center", "top center", "top center", "bottom left", "top left"],
        marker=dict(size=10, color="white", line=dict(width=2, color="#1D3557")),
        name="Milestones",
        textfont=dict(size=11, color="#1D3557", family="Arial")
    ))

    # 3. Reference Lines
    fig.add_shape(type="line", x0=2026, y0=0, x1=2040, y1=0, line=dict(color="#2c3e50", width=2))
    
    for yr in [2030, 2035, 2040]:
        fig.add_shape(type="line", x0=yr, y0=-110, x1=yr, y1=110, line=dict(color="lightgray", width=1, dash="dot"))

    # 4. Layout Customization
    fig.update_layout(
        template="plotly_white",
        
        xaxis=dict(
            title="Time Horizon (Strategic Years)",
            range=[2025, 2042], 
            tickmode='linear', 
            tick0=2026, 
            dtick=4,
            showgrid=False
        ),
        yaxis=dict(title="Strategic Uncertainty", showticklabels=False, range=[-120, 120]),
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
        height=600,
        margin=dict(l=40, r=80, t=60, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)

render_extended_strategic_cone()