import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def render_strategic_trend_radar():
    st.write("### 📡 STRATEGIC AI TREND RADAR (2026-2040)")

    # 1. Veri Hazırlama (R: trends_data)
    current_year = 2026
    data = {
        "trend": ["Agentic AI Workflows", "AI-Native Drug Discovery", "Synthetic Data Regulation", 
                  "On-Device SLMs", "Hyper-Personalized Learning", "Sovereign AI Infrastructure", 
                  "Post-Labor Economy Debates", "AI Carbon Accounting"],
        "category": ["Technological", "Environmental", "Political", "Social", "Social", "Political", "Economic", "Environmental"],
        "maturity_year": [2026, 2035, 2027, 2026, 2030, 2028, 2040, 2027],
        "impact": [10, 10, 6, 8, 7, 9, 8, 5]
    }
    df = pd.DataFrame(data)
    
    # Mesafe hesaplama (R: distance)
    df['distance'] = df['maturity_year'].apply(lambda x: min(x - current_year + 0.5, 10))

    # 2. Geometri ve Açı Hesaplamaları
    categories = ["Social", "Technological", "Economic", "Environmental", "Political", "Values"]
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
    angle_map = dict(zip(categories, angles))
    
    df['angle'] = df['category'].map(angle_map)
    df['x'] = df['distance'] * np.cos(df['angle'])
    df['y'] = df['distance'] * np.sin(df['angle'])

    # 3. Plotly Figürü Oluşturma
    fig = go.Figure()

    # Arka Plan Halkaları (Horizon Circles)
    horizons = [(1, "Now (2026)"), (4, "Next (2027-2030)"), (10, "Beyond (2031+)")]
    for r, label in horizons:
        theta = np.linspace(0, 2*np.pi, 100)
        fig.add_trace(go.Scatter(
            x=r * np.cos(theta), y=r * np.sin(theta),
            mode='lines', line=dict(color='rgba(200, 200, 200, 0.5)', dash='solid'),
            showlegend=False, hoverinfo='skip'
        ))
        fig.add_annotation(x=0.2, y=r + 0.3, text=label, showarrow=False, font=dict(size=10, color="gray"))

    # Kategori Çizgileri (Spoke Lines)
    for cat, angle in angle_map.items():
        fig.add_trace(go.Scatter(
            x=[0, 11 * np.cos(angle)], y=[0, 11 * np.sin(angle)],
            mode='lines', line=dict(color='rgba(180, 180, 180, 0.4)', dash='dash'),
            showlegend=False, hoverinfo='skip'
        ))
        fig.add_annotation(
            x=13 * np.cos(angle), y=13 * np.sin(angle),
            text=f"<b>{cat}</b>", showarrow=False, font=dict(size=12, color="black")
        )

    # Trend Noktaları (Trends)
    colors = px.colors.qualitative.Safe
    for i, cat in enumerate(categories):
        cat_df = df[df['category'] == cat]
        if not cat_df.empty:
            fig.add_trace(go.Scatter(
                x=cat_df['x'], y=cat_df['y'],
                mode='markers+text',
                name=cat,
                text=cat_df['trend'],
                textposition="top center",
                marker=dict(
                    size=cat_df['impact'] * 3,
                    color=colors[i % len(colors)],
                    opacity=0.7,
                    line=dict(width=1, color='white')
                ),
                hovertemplate="<b>%{text}</b><br>Maturity: %{customdata}<extra></extra>",
                customdata=cat_df['maturity_year']
            ))

    # 4. Legend and Layout Optimization
    fig.update_layout(
        template="plotly_white",
        title=dict(text="Strategic AI Trend Radar (2026-2040)", x=0.5, font=dict(size=20)),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-18, 18]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-15, 15]),
        height=800, # Increased height to accommodate legend
        
        # KEY CHANGES HERE:
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2, # Moves legend further below the X-axis
            xanchor="center",
            x=0.5,
            title=dict(text="Trend Categories", font=dict(size=12))
        ),
        margin=dict(l=20, r=20, t=60, b=100) # Increased bottom margin (b=100)
    )

    st.plotly_chart(fig, use_container_width=True)
    

# Call the function
render_strategic_trend_radar()

# 5. Create Impact Legend (Manual Legend Construction)
st.write("---")
impact_vals = [6, 7, 8, 9, 10]

# Create a small horizontal figure for the Impact Legend
impact_fig = go.Figure()

impact_fig.add_trace(go.Scatter(
    x=impact_vals,
    y=[0] * len(impact_vals),
    mode='markers+text',
    marker=dict(
        size=[v * 3 for v in impact_vals], # Match your radar's size logic
        color='gray',
        opacity=0.7
    ),
    text=impact_vals,
    textposition="bottom center",
    hoverinfo='skip'
))

impact_fig.update_layout(
    title=dict(text="Impact Score (Marker Size)", font=dict(size=14), x=0.5),
    height=120,
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[5, 11]),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1, 1]),
    template="plotly_white",
    showlegend=False
)

st.plotly_chart(impact_fig, use_container_width=True)
