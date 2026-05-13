import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# PROPERLY WORKING GRAPHS

# Page Settings
st.set_page_config(page_title="Strategic Technology Foresight Hub", layout="wide")

# --- Style Settings  ---
st.markdown("""
    <style>
    .main { background-color: #fcfcfc; }
    h1, h2, h3 { color: #1e3a8a; font-family: 'Inter', sans-serif; }
    .stMarkdown { font-size: 1.1rem; }
    </style>
    """, unsafe_allow_html=True)

# --- Data Upload ---
@st.cache_data
def load_data():
    file_path = "generative-ai-patent.csv"
    try:
        df = pd.read_csv(file_path)

        df.columns = [c.replace(' ', '.') for c in df.columns]
    except FileNotFoundError:

        years = np.arange(2010, 2026)
        df = pd.DataFrame({
            'IPCR.Classifications': ['G06F;G06N', 'H04L', 'G06T'] * 40,
            'Applicants': ['Google', 'Microsoft', 'IBM', 'Samsung'] * 30,
            'Publication.Year': np.random.choice(years, 120),
            'Jurisdiction': ['US', 'CN', 'EP', 'JP'] * 30
        })
    return df

df = load_data()


st.title("🌐 STRATEGIC TECHNOLOGY FORESIGHT HUB (2026-2040)")
st.caption("Strategic Clarity and Innovation Roadmap")
st.divider()

# --- ROW 1: STRATEGIC ROADMAP (PANEL 3) ---
st.subheader("🚀 STRATEGIC ROADMAP (2026-2030)")
base_date = datetime(2026, 1, 1)
road_df = pd.DataFrame([
    dict(Task="Phase 1: Compliance", Start=base_date, End=base_date + timedelta(days=4*120), Color="#4A90E2"),
    dict(Task="Phase 2: R&D Logic", Start=base_date + timedelta(days=2*120), End=base_date + timedelta(days=9*120), Color="#50E3C2"),
    dict(Task="Phase 3: Pilot", Start=base_date + timedelta(days=7*120), End=base_date + timedelta(days=13*120), Color="#F5A623"),
    dict(Task="Phase 4: Scaling", Start=base_date + timedelta(days=12*120), End=base_date + timedelta(days=18*120), Color="#B8E986"),
])

fig_roadmap = px.timeline(road_df, x_start="Start", x_end="End", y="Task", color="Color",
                         color_discrete_map="identity", template="plotly_white")
fig_roadmap.update_layout(showlegend=False, height=300, yaxis_title=None, xaxis_title="Timeline")
st.plotly_chart(fig_roadmap, width="stretch")

st.divider()

# --- ROW 2: TREND RADAR (PANEL 1) & FUTURES CONE (PANEL 4) ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📡 STRATEGIC TREND RADAR")
    # 1. Strategic Data Preparation (from generative-AI-trend-radar.py)
    current_year = 2026
    radar_data_dict = {
        "trend": ["Agentic AI Workflows", "AI-Native Drug Discovery", "Synthetic Data Regulation", 
                  "On-Device SLMs", "Hyper-Personalized Learning", "Sovereign AI Infrastructure", 
                  "Post-Labor Economy Debates", "AI Carbon Accounting"],
        "category": ["Technological", "Environmental", "Political", "Social", "Social", "Political", "Economic", "Environmental"],
        "maturity_year": [2026, 2035, 2027, 2026, 2030, 2028, 2040, 2027],
        "impact": [10, 10, 6, 8, 7, 9, 8, 5]
    }
    df_radar = pd.DataFrame(radar_data_dict)
    
    # Calculate radar distance and coordinates
    df_radar['distance'] = df_radar['maturity_year'].apply(lambda x: min(x - current_year + 0.5, 10))
    categories = ["Social", "Technological", "Economic", "Environmental", "Political", "Values"]
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
    angle_map = dict(zip(categories, angles))
    
    df_radar['angle'] = df_radar['category'].map(angle_map)
    df_radar['x'] = df_radar['distance'] * np.cos(df_radar['angle'])
    df_radar['y'] = df_radar['distance'] * np.sin(df_radar['angle'])

    # 2. Build Advanced Radar Figure
    fig_radar = go.Figure()

    # Background Horizon Rings
    horizons = [(1, "Now (2026)"), (4, "Next (2027-30)"), (10, "Beyond (2031+)")]
    for r, label in horizons:
        theta = np.linspace(0, 2*np.pi, 100)
        fig_radar.add_trace(go.Scatter(
            x=r * np.cos(theta), y=r * np.sin(theta),
            mode='lines', line=dict(color='rgba(200, 200, 200, 0.5)', dash='solid'),
            showlegend=False, hoverinfo='skip'
        ))
        fig_radar.add_annotation(x=0.2, y=r + 0.4, text=label, showarrow=False, font=dict(size=10, color="gray"))

    # Category Spokes
    for cat, angle in angle_map.items():
        fig_radar.add_trace(go.Scatter(
            x=[0, 11 * np.cos(angle)], y=[0, 11 * np.sin(angle)],
            mode='lines', line=dict(color='rgba(180, 180, 180, 0.4)', dash='dash'),
            showlegend=False, hoverinfo='skip'
        ))
        fig_radar.add_annotation(
            x=13 * np.cos(angle), y=13 * np.sin(angle),
            text=f"<b>{cat}</b>", showarrow=False, font=dict(size=11, color="black")
        )

    # Trend Markers
    colors = px.colors.qualitative.Safe
    for i, cat in enumerate(categories):
        cat_df = df_radar[df_radar['category'] == cat]
        if not cat_df.empty:
            fig_radar.add_trace(go.Scatter(
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
                customdata=cat_df['maturity_year'],
                hovertemplate="<b>%{text}</b><br>Maturity: %{customdata}<extra></extra>"
            ))

    fig_radar.update_layout(
        template="plotly_white",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-18, 18]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-15, 15]),
        height=600,
        margin=dict(l=10, r=10, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # 3. Impact Score Legend
    impact_vals = [6, 7, 8, 9, 10]
    impact_fig = go.Figure()
    impact_fig.add_trace(go.Scatter(
        x=impact_vals, y=[0] * len(impact_vals),
        mode='markers+text',
        marker=dict(size=[v * 3 for v in impact_vals], color='gray', opacity=0.7),
        text=impact_vals, textposition="bottom center", hoverinfo='skip'
    ))
    impact_fig.update_layout(
        title=dict(text="Impact Score (Marker Size)", font=dict(size=14), x=0.5),
        height=120, margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[5, 11]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1, 1]),
        template="plotly_white", showlegend=False
    )
    st.plotly_chart(impact_fig, use_container_width=True)

with col2:
    st.subheader("🗼 FUTURES CONE (2026-2040)")
    x_cone = np.linspace(0, 14, 100)
    fig_cone = go.Figure()
    
    # Cone Geometry (Preposterous & Plausible)
    fig_cone.add_trace(go.Scatter(x=x_cone, y=2.2*x_cone, fill=None, mode='lines', line_color='rgba(200,200,200,0.1)', showlegend=False))
    fig_cone.add_trace(go.Scatter(x=x_cone, y=-2.2*x_cone, fill='tonexty', mode='lines', line_color='rgba(200,200,200,0.1)', name='Preposterous'))
    fig_cone.add_trace(go.Scatter(x=x_cone, y=0.8*x_cone, fill=None, mode='lines', line_color='rgba(74,144,226,0.2)', showlegend=False))
    fig_cone.add_trace(go.Scatter(x=x_cone, y=-0.8*x_cone, fill='tonexty', mode='lines', line_color='rgba(74,144,226,0.2)', name='Plausible'))
    
    # Strategic Milestones (Wildcards) from futures-cone-GEN-AI.py
    milestones = pd.DataFrame([
        dict(x=4, y=4.5, label="Agentic AI Standard"),
        dict(x=4, y=-13.5, label="AI-Generated Full Movies"),
        dict(x=9, y=1.5, label="NPU-Only Hardware"),
        dict(x=9, y=24.0, label="Personalized AI Medicine"),
        dict(x=14, y=-4.5, label="Physical AI Elderly Care"),
        dict(x=14, y=15.0, label="Bespoke Protein Design")
    ])
    fig_cone.add_trace(go.Scatter(
        x=milestones["x"], y=milestones["y"],
        mode="markers+text", text=milestones["label"],
        textposition="top center",
        marker=dict(size=10, color="white", line=dict(width=2, color="#1D3557")),
        name="Wildcards/Milestones", textfont=dict(size=11, color="#1D3557")
    ))

    # Projection Lines
    fig_cone.add_trace(go.Scatter(x=[0, 14], y=[0, 0], mode='lines', line=dict(color='#4A90E2', width=3), name='Projected'))
    fig_cone.add_trace(go.Scatter(x=[0, 14], y=[0, 10], mode='lines', line=dict(color='#B8E986', dash='dash'), name='Preferable'))

    fig_cone.update_layout(
            template="plotly_white", 
            yaxis_range=[-35, 35], 
            xaxis=dict(tickmode='array', tickvals=[0, 4, 9, 14], ticktext=['2026', '2030', '2035', '2040']),
            showlegend=True, 
            height=600, 
            # Increase 'b' (bottom) margin to 100 to accommodate the legend
            margin=dict(l=40, r=40, t=60, b=100), 
            legend=dict(
                orientation="h",      # Horizontal legend
                yanchor="bottom",     # Anchored to the bottom
                y=-0.3,               # Positioned below the x-axis
                xanchor="center",     # Centered horizontally
                x=0.5
            )
        )
    st.plotly_chart(fig_cone, use_container_width=True)

st.divider()

# --- ROW 3: S-CURVE (PANEL 5), JURISDICTIONS (PANEL 6), APPLICANTS (PANEL 2) ---
col3, col4, col5 = st.columns(3)

with col3:
    st.subheader("📈 MATURITY (S-CURVE)")
    s_data = df.groupby('Publication.Year').size().sort_index().cumsum().reset_index(name='cum')
    
    fig_s = go.Figure()
    # Mevcut Veri
    fig_s.add_trace(go.Scatter(x=s_data['Publication.Year'], y=s_data['cum'], 
                              mode='lines+markers', name='Historical', line=dict(color='#4A90E2')))
    
    last_year = int(s_data['Publication.Year'].max())
    last_val = s_data['cum'].iloc[-1]
    fig_s.add_trace(go.Scatter(x=[last_year, last_year + 8], y=[last_val, last_val * 4],
                              mode='lines', name='Forecast', line=dict(color='#FF3D00', dash='dash')))
    
    fig_s.update_layout(template="plotly_white", xaxis_title="Publication.Year", yaxis_title="Innovation Index")
    st.plotly_chart(fig_s, width="stretch")

with col4:
    st.subheader("🌍 TOP JURISDICTIONS")
    jur_counts = df['Jurisdiction'].value_counts().head(6).reset_index()
    fig_jur = px.bar(jur_counts, x='Jurisdiction', y='count', template="plotly_white",
                    color_discrete_sequence=["#F5A623"])
    st.plotly_chart(fig_jur, width="stretch")

with col5:
    st.subheader("🏢 TOP APPLICANTS")
    # value_counts() sonrası sütun isimlerini manuel netleştiriyoruz
    app_counts = df['Applicants'].value_counts().head(6).reset_index()
    app_counts.columns = ['Applicants', 'count'] 
    
    app_counts['Applicants_Display'] = app_counts['Applicants'].apply(
        lambda x: str(x)[:17] + "..." if len(str(x)) > 20 else str(x)
    )
    
    fig_app = px.bar(app_counts, 
                    y='Applicants_Display', 
                    x='count', 
                    orientation='h', 
                    template="plotly_white", 
                    color_discrete_sequence=["#50E3C2"])
    
    fig_app.update_layout(
        yaxis={'categoryorder':'total ascending'},
        yaxis_title=None
    )
    st.plotly_chart(fig_app, width="stretch")