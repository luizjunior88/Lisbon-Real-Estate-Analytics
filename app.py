import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# --- Page Config ---
st.set_page_config(
    page_title="Lisbon Real Estate Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Header ---
st.title("🏡 Lisbon Real Estate: Market Valuation Model")
st.markdown("### Ridge Regression Analysis ($L_2$ Regularization)")

col_pt, col_en = st.columns(2)
with col_pt:
    st.info("""
    **🇵🇹 Resumo Executivo:**
    Modelagem preditiva para isolar o **Valor Intrínseco** da especulação imobiliária.
    A abordagem utiliza regularização para minimizar o impacto de *outliers* e multicolinearidade.
    """)
with col_en:
    st.warning("""
    **🇬🇧 Executive Summary:**
    Predictive modeling designed to isolate **Intrinsic Value** from market speculation.
    The approach uses regularization to minimize the impact of outliers and multicollinearity.
    """)

st.divider()

# --- Data Loading ---
@st.cache_data
def load_data():
    return pd.read_csv('portfolio_data.csv')

try:
    df = load_data()
except FileNotFoundError:
    st.error("CRITICAL: Data artifact 'portfolio_data.csv' not found.")
    st.stop()

# --- Sidebar Filters ---
st.sidebar.header("⚙️ Parameters")

parishes = sorted(df['Parish'].unique())
selected_parishes = st.sidebar.multiselect(
    "Parish / Freguesia:",
    options=parishes,
    default=parishes[:3]
)

conditions = sorted(df['Condition'].unique())
selected_conditions = st.sidebar.multiselect(
    "Condition / Condição:",
    options=conditions,
    default=conditions
)

if not selected_parishes:
    st.stop()

df_filtered = df[
    (df['Parish'].isin(selected_parishes)) &
    (df['Condition'].isin(selected_conditions))
]

if df_filtered.empty:
    st.warning("No data available for current selection.")
    st.stop()

# --- Section 1: Metrics & Math ---
st.subheader("1. Mathematical Framework & Performance Metrics")

col_math, col_metrics = st.columns([1.5, 1])

with col_math:
    st.markdown("**Cost Function Optimization (Ridge):**")
    st.latex(r'''
    J(\beta) = ||y - X\beta||_2^2 + \lambda ||\beta||_2^2
    ''')
    st.caption("Objective function minimizing Residual Sum of Squares (RSS) with L2 penalty.")

with col_metrics:
    mae = df_filtered['Abs_Error'].mean()
    rmse = np.sqrt((df_filtered['Error']**2).mean())
    
    st.metric("Mean Absolute Error (MAE)", f"€ {mae:,.0f}")
    st.metric("Root Mean Squared Error (RMSE)", f"€ {rmse:,.0f}")

st.divider()

# --- Section 2: Visual Analytics ---
st.subheader("2. Market Dynamics & Residual Analysis")

# Map
st.markdown("**Geospatial Residual Distribution:** Red markers indicate high model deviation (Potential Anomalies).")
try:
    fig_map = px.scatter_mapbox(
        df_filtered,
        lat="Latitude",
        lon="Longitude",
        color="Abs_Error",
        size="Real_Price",
        color_continuous_scale=px.colors.sequential.Bluered,
        hover_name="Parish",
        hover_data={"Real_Price": ":,.0f", "Predicted_Price": ":,.0f", "Abs_Error": ":,.0f"},
        zoom=11,
        height=550
    )
    fig_map.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)
except Exception as e:
    st.error(f"Map Rendering Error: {e}")

# Scatter
st.markdown("**Linearity Assessment:** Area Gross vs. Real Price")
try:
    fig_scatter = px.scatter(
        df_filtered,
        x="AreaGross",
        y="Real_Price",
        color="Condition",
        trendline="ols", # Requires statsmodels
        labels={"AreaGross": "Gross Area (m²)", "Real_Price": "Price (€)"},
        height=500
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
except:
    st.warning("Trendline unavailable. Displaying raw scatter plot.")
    fig_scatter = px.scatter(df_filtered, x="AreaGross", y="Real_Price", color="Condition")
    st.plotly_chart(fig_scatter, use_container_width=True)

st.caption("© 2024 Portfolio Project. Built with Python & Streamlit.")
