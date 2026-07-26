import streamlit as st
import pandas as pd
from eda_logic import run_eda, detect_column_types, apply_filters
from ai_insights import generate_insights

st.set_page_config(
    page_title="Auto-EDA App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
    /* Overall padding */
    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Page background gradient */
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at top left, #131722 0%, #0e1117 60%);
    }

    /* Hero section */
    .hero-box {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 0.5rem;
    }
    .hero-icon {
        font-size: 2.8rem;
        line-height: 1;
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(90deg, #2dd4bf, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 1.08rem;
        color: #9ca3af;
        margin: 0.3rem 0 2rem 0;
        padding-left: 0.1rem;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #1c1f26, #171a21);
        border: 1px solid #2a2e37;
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 4px 14px rgba(0,0,0,0.35);
        transition: transform 0.15s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: #14b8a6;
    }
    div[data-testid="stMetricLabel"] {
        color: #9ca3af;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetricValue"] {
        color: #2dd4bf;
        font-size: 1.8rem;
        font-weight: 700;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        font-weight: 600;
        font-size: 0.95rem;
        border-radius: 8px 8px 0 0;
    }
    div[data-baseweb="tab-highlight"] {
        background-color: #2dd4bf !important;
        height: 3px !important;
    }

    /* File uploader */
    div[data-testid="stFileUploaderDropzone"] {
        border-radius: 14px;
        border: 2px dashed #2dd4bf55;
        background: #14171e;
    }

    /* Section headers */
    h3 {
        border-left: 4px solid #14b8a6;
        padding-left: 0.7rem;
        margin-top: 2rem !important;
        font-weight: 700 !important;
    }

    /* Dataframe container */
    div[data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #2a2e37;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown("""
<div class="hero-box">
    <div class="hero-icon">📊</div>
    <div class="hero-title">Auto-EDA</div>
</div>
<div class="hero-subtitle">Upload any CSV, Excel, or JSON file and get instant charts, stats, and AI-generated insights — no code required.</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload your data file",
    type=["csv", "xlsx", "xls", "json"],
    label_visibility="collapsed"
)

if uploaded_file is not None:
    file_name = uploaded_file.name.lower()

    try:
        if file_name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif file_name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        elif file_name.endswith(".json"):
            df = pd.read_json(uploaded_file)
        else:
            st.error("Unsupported file type.")
            st.stop()
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

    st.markdown("### 🎛️ Filters (optional)")
    with st.expander("Click to filter your data before analysis"):
        column_types_preview = detect_column_types(df)
        filters = {}

        filter_cols = st.multiselect(
            "Choose columns to filter by",
            options=df.columns.tolist()
        )

        for col in filter_cols:
            col_type = column_types_preview.get(col)

            if col_type == "categorical":
                unique_vals = df[col].dropna().unique().tolist()
                selected_vals = st.multiselect(f"Filter '{col}'", options=unique_vals, default=unique_vals)
                filters[col] = {"type": "categorical", "values": selected_vals}

            elif col_type == "numeric":
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                selected_range = st.slider(f"Filter '{col}' range", min_val, max_val, (min_val, max_val))
                filters[col] = {"type": "numeric", "range": selected_range}

        if filters:
            df = apply_filters(df, filters)
            st.success(f"Showing {len(df)} rows after filtering.")

    results = run_eda(df)

    st.markdown("### 🔍 Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.markdown("### 📈 Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{results['summary']['n_rows']:,}")
    col2.metric("Columns", results["summary"]["n_cols"])
    col3.metric("Duplicates", results["summary"]["duplicates"])
    col4.metric("Missing %", f"{results['summary']['missing_total_pct']}%")

    st.markdown("###")
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Numeric Columns", "🗂️ Categorical Columns", "🔗 Correlations", "🤖 AI Insights"])

    with tab1:
        if results["numeric_charts"]:
            for col, fig in results["numeric_charts"].items():
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric columns found.")

    with tab2:
        if results["categorical_charts"]:
            for col, fig in results["categorical_charts"].items():
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No categorical columns found.")

    with tab3:
        if results["correlation_chart"]:
            st.plotly_chart(results["correlation_chart"], use_container_width=True)
        else:
            st.info("Not enough numeric columns for correlation.")

    with tab4:
        with st.spinner("Generating insights..."):
            insights = generate_insights(
                results["summary"],
                results["column_types"],
                results["numeric_stats"],
                results["categorical_stats"]
            )
        st.markdown(insights)
else:
    st.info("👆 Upload a CSV, Excel, or JSON file to get started.")