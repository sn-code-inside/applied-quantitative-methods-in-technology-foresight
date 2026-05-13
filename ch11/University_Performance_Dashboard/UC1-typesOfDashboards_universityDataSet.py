# --- HOW TO RUN THE CODE ---
# 1. Ensure you have the required libraries installed:
#    pip install pandas streamlit plotly
# 2. Make sure the data file 'UC1-typesOfDashboards_universityDataSet.csv' is in the same directory.
# 3. Run the application from your terminal:
#    streamlit run dashboard_onepage.py
# ---------------------------

# Import the pandas library for data manipulation and analysis
import pandas as pd
# Import the streamlit library for building the web application
import streamlit as st
# Import the plotly.express library for creating interactive visualizations
import plotly.express as px
# Import the warnings module to manage runtime warnings
import warnings

# Optional: hide benign Streamlit context warnings that can sometimes clutter the console
warnings.filterwarnings("ignore", message="missing ScriptRunContext")

# Load dataset (The try/except block is a common technique to load data, 
# though here it's redundant as it retries the same action)
try:
    # Attempt to read the CSV file, assuming a semicolon (;) separator
    df = pd.read_csv("UC1-typesOfDashboards_universityDataSet.csv", sep=';')
except:
    # If the first attempt fails (e.g., due to file not found or corrupted), 
    # it tries again with the same parameters (this specific structure should be reviewed/corrected if the intent was to try a different separator)
    df = pd.read_csv("UC1-typesOfDashboards_universityDataSet.csv", sep=',')

# Configure the Streamlit page settings:
# - page_title: Sets the title displayed in the browser tab
# - layout: 'wide' utilizes the full width of the screen for the dashboard
st.set_page_config(page_title="University Performance Dashboard", layout="wide")

# --- Sidebar Navigation Setup ---
# Add a title to the sidebar for clarity
st.sidebar.title("📊 Dashboard Navigation")
# Create a radio button widget in the sidebar for selecting the dashboard view
page = st.sidebar.radio(
    "Choose a dashboard view:",
    ["Main Overview", "Strategic", "Analytical", "Operational", "Tactical"]
)

# --- Main Page Title and Caption ---
st.title("🎓 University Performance Dashboard")
# Add a smaller subtitle/caption for context
st.caption("Data-driven decision support for university management")

# --- Shared Chart Components (Functions for Reusability) ---
# Function to generate charts for the Strategic dashboard
def strategic_charts():
    # Fig 1: Line chart showing Student Satisfaction trends over time, separated by faculty
    fig1 = px.line(df, x="month", y="satisfaction", color="faculty", title="Student Satisfaction Trends")
    # Fig 2: Line chart showing Publication growth over time, separated by faculty
    fig2 = px.line(df, x="month", y="publications", color="faculty", title="Publication Growth Over Time")
    # Return both Plotly figures in a list
    return [fig1, fig2]

# Function to generate charts for the Analytical dashboard
def analytical_charts():
    # Fig 3: Scatter plot to explore the correlation between 'budget' and 'publications'.
    # Color is based on 'faculty', and marker size represents 'students' (enrollment).
    fig3 = px.scatter(df, x="budget", y="publications", color="faculty", size="students",
                      title="Budget vs. Publications by Faculty")
    # Return the Plotly figure in a list
    return [fig3]

# Function to generate charts for the Operational dashboard
def operational_charts():
    # Fig 4: Bar chart showing the average attendance rate for each faculty
    fig4 = px.bar(df, x="faculty", y="attendance_rate", color="faculty",
                  title="Average Attendance Rate by Faculty")
    # Return the Plotly figure in a list
    return [fig4]

# Function to generate charts for the Tactical dashboard
def tactical_charts():
    # Fig 5: Box plot showing the distribution (spread, median, quartiles) of student satisfaction 
    # across different faculties. Good for mid-term performance comparison.
    fig5 = px.box(df, x="faculty", y="satisfaction", title="Faculty-Level Satisfaction Distribution")
    # Return the Plotly figure in a list
    return [fig5]

# --- CONDITIONAL PAGE RENDERING ---

# Check if the 'Main Overview' page is selected
if page == "Main Overview":
    st.subheader("📈 Main Overview Dashboard")
    st.write("A consolidated view combining strategic, analytical, operational, and tactical insights.")

    # Create two columns to display charts side-by-side
    col1, col2 = st.columns(2)
    # Use the first column
    with col1:
        # Display the first Strategic chart (Satisfaction Trends)
        st.plotly_chart(strategic_charts()[0], use_container_width=True)
    # Use the second column
    with col2:
        # Display the second Strategic chart (Publication Growth)
        st.plotly_chart(strategic_charts()[1], use_container_width=True)

    # Create another two columns for the next set of charts
    col3, col4 = st.columns(2)
    # Use the third column
    with col3:
        # Display the Analytical chart (Budget vs. Publications)
        st.plotly_chart(analytical_charts()[0], use_container_width=True)
    # Use the fourth column
    with col4:
        # Display the Operational chart (Attendance Rate)
        st.plotly_chart(operational_charts()[0], use_container_width=True)

    # Display the Tactical chart (Satisfaction Distribution) across the full width
    st.plotly_chart(tactical_charts()[0], use_container_width=True)

# Check if the 'Strategic' page is selected
elif page == "Strategic":
    st.subheader("Strategic Dashboard")
    st.write("Long-term performance indicators and institutional trends.")
    # Loop through and display all charts defined in the strategic_charts function
    for fig in strategic_charts():
        st.plotly_chart(fig, use_container_width=True)

# Check if the 'Analytical' page is selected
elif page == "Analytical":
    st.subheader("Analytical Dashboard")
    st.write("Explore relationships and correlations between different indicators.")
    # Loop through and display all charts defined in the analytical_charts function
    for fig in analytical_charts():
        st.plotly_chart(fig, use_container_width=True)

# Check if the 'Operational' page is selected
elif page == "Operational":
    st.subheader("Operational Dashboard")
    st.write("Track short-term operational metrics and daily activities.")
    # Loop through and display all charts defined in the operational_charts function
    for fig in operational_charts():
        st.plotly_chart(fig, use_container_width=True)

# Check if the 'Tactical' page is selected
elif page == "Tactical":
    st.subheader("Tactical Dashboard")
    st.write("Measure mid-term faculty and department performance.")
    # Loop through and display all charts defined in the tactical_charts function
    for fig in tactical_charts():
        st.plotly_chart(fig, use_container_width=True)

# --- Sidebar Footer/Information ---
# Add a horizontal line for separation in the sidebar
st.sidebar.markdown("---")
# Add a small information box to the sidebar
st.sidebar.info("Dashboard types: Strategic | Analytical | Operational | Tactical")