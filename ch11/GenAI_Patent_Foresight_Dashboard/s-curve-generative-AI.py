import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.optimize import curve_fit

# --- App Configuration ---
st.set_page_config(page_title="Patent S-Curve Tracker", layout="wide")

st.title("S-Curve: Cumulative Generative AI Patent Growth")

# --- 1. Robust Sigmoid Function ---
def sigmoid(x, L, k, x0):
    # np.clip prevents overflow/large image errors
    return L / (1 + np.exp(-np.clip(k * (x - x0), -50, 50)))

# --- 2. Data Loading ---
@st.cache_data
def load_data():
    try:
        return pd.read_csv("generative-ai-patent.csv")
    except:
        return None

patent_data = load_data()

if patent_data is not None:
    # --- 3. Preprocessing ---
    s_curve_data = (
        patent_data.groupby('Publication Year')
        .size()
        .reset_index(name='Yearly_Count')
        .sort_values('Publication Year')
    )
    s_curve_data['Cumulative_Patents'] = s_curve_data['Yearly_Count'].cumsum()
    
    # --- 4. Fitting Logic ---
    x_raw = s_curve_data['Publication Year'].values
    y_data = s_curve_data['Cumulative_Patents'].values
    x_min = x_raw.min()
    x_norm = x_raw - x_min 

    success = False
    try:
        # Smart Guesses:
        # L (Max) = 2x current max, k (Growth) = 1.0, x0 (Midpoint) = middle of year range
        initial_guesses = [max(y_data) * 2, 1.0, np.median(x_norm)]
        
        # Set bounds to prevent the optimizer from wandering into impossible numbers
        # (L must be > 0, k must be > 0, etc.)
        lower_bounds = [max(y_data), 0.01, -20]
        upper_bounds = [max(y_data) * 100, 5.0, 40]

        popt, _ = curve_fit(sigmoid, x_norm, y_data, p0=initial_guesses, 
                            bounds=(lower_bounds, upper_bounds), maxfev=10000)
        
        x_smooth_norm = np.linspace(x_norm.min(), x_norm.max() + 1, 100)
        y_smooth = sigmoid(x_smooth_norm, *popt)
        x_smooth_years = x_smooth_norm + x_min
        success = True
        curve_label = "Sigmoid S-Curve"
        
    except Exception:
        # FALLBACK: If Sigmoid fails, use a 2nd degree polynomial (smooth curve)
        z = np.polyfit(x_norm, y_data, 2)
        p = np.poly1d(z)
        x_smooth_norm = np.linspace(x_norm.min(), x_norm.max() + 1, 100)
        y_smooth = p(x_smooth_norm)
        x_smooth_years = x_smooth_norm + x_min
        success = False
        curve_label = "Polynomial Trend (Sigmoid Fit Failed)"

    # --- 5. UI Layout ---
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Data Summary")
        st.dataframe(s_curve_data, use_container_width=True)
        if success:
            st.success("Mathematical S-Curve Found")
            st.metric("Estimated Saturation", f"{int(popt[0])} Patents")
        else:
            st.warning("Data is too linear for a Sigmoid fit. Showing polynomial trend.")

    with col2:
        st.subheader("Growth Visualization")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.set_style("whitegrid")

        # Plot raw points
        ax.scatter(x_raw, y_data, color='#2c3e50', s=100, label='Actual Data', zorder=5)
        
        # Plot trend line
        ax.plot(x_smooth_years, y_smooth, color='#e74c3c', linewidth=3, label=curve_label)

        ax.set_xlabel('Year')
        ax.set_ylabel('Total Patents')
        ax.legend()
        st.pyplot(fig)

else:
    st.error("CSV file not found. Please check 'generative-ai-patent.csv'.")