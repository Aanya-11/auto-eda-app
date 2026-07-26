import streamlit as st
import pandas as pd
import plotly.express as px
from eda_logic import run_eda, detect_column_types, apply_filters, apply_imputation, apply_transformations
from ai_insights import generate_insights, ask_data_question

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
            st.dataframe(df, use_container_width=True, height=350)

    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        st.markdown("### 🧹 Handle Missing Values (optional)")
        with st.expander("Click to fill or drop missing values"):
            strategy_map = {}
            impute_cols = st.multiselect(
                "Choose columns to clean",
                options=missing_cols
            )

            col_types_for_impute = detect_column_types(df)

            for col in impute_cols:
                col_type = col_types_for_impute.get(col)
                if col_type == "numeric":
                    strategy = st.selectbox(
                        f"Strategy for '{col}' (numeric)",
                        options=["mean", "median", "drop_rows"],
                        key=f"impute_{col}"
                    )
                else:
                    strategy = st.selectbox(
                        f"Strategy for '{col}' (categorical/text)",
                        options=["mode", "drop_rows"],
                        key=f"impute_{col}"
                    )
                strategy_map[col] = strategy

            if strategy_map and st.button("Apply Cleaning"):
                df = apply_imputation(df, strategy_map)
                st.success(f"Cleaned! {len(df)} rows remain.")
                st.dataframe(df, use_container_width=True, height=350)

    st.markdown("### 🔧 Transform Columns (optional)")
    with st.expander("Click to scale, normalize, or encode columns"):
        transform_map = {}
        col_types_for_transform = detect_column_types(df)
        transform_cols = st.multiselect(
            "Choose columns to transform",
            options=df.columns.tolist(),
            key="transform_cols"
        )

        for col in transform_cols:
            col_type = col_types_for_transform.get(col)
            if col_type == "numeric":
                method = st.selectbox(
                    f"Method for '{col}' (numeric)",
                    options=["log", "normalize"],
                    key=f"transform_{col}"
                )
            else:
                unique_count = df[col].nunique()
                if unique_count > 50:
                    st.caption(
                        f"⚠️ '{col}' has {unique_count} unique values — "
                        f"one-hot encoding will be skipped for this column (too many values, would use too much memory)."
                    )
                method = st.selectbox(
                    f"Method for '{col}' (categorical)",
                    options=["onehot"],
                    key=f"transform_{col}"
                )
            transform_map[col] = method

        if transform_map and st.button("Apply Transformations"):
            df, skipped_cols = apply_transformations(df, transform_map)
            applied_count = len(transform_map) - len(skipped_cols)
            if applied_count > 0:
                st.success(f"Transformed {applied_count} column(s).")
            if skipped_cols:
                for col, count in skipped_cols:
                    st.warning(
                        f"⚠️ Skipped '{col}' for one-hot encoding — it has {count} unique values "
                        f"(too many, would use too much memory). Try a column with fewer unique values."
                    )
            st.dataframe(df, use_container_width=True, height=350)

    results = run_eda(df)

    st.markdown("### 🔍 Data Preview")
    show_full_data = st.toggle(f"Show full data ({len(df):,} rows)", value=False)
    if show_full_data:
        st.dataframe(df, use_container_width=True, height=500)
    else:
        st.dataframe(df.head(100), use_container_width=True, height=400)
        st.caption(f"Showing first 100 of {len(df):,} rows. Toggle above to see all rows.")

    st.markdown("### 📈 Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{results['summary']['n_rows']:,}")
    col2.metric("Columns", results["summary"]["n_cols"])
    col3.metric("Duplicates", results["summary"]["duplicates"])
    col4.metric("Missing %", f"{results['summary']['missing_total_pct']}%")

    st.markdown("###")
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Numeric Columns",
        "🗂️ Categorical Columns",
        "🔗 Correlations",
        "🕳️ Missing Data",
        "🤖 AI Insights",
        "💬 Ask Your Data"
    ])

    with tab1:
        if results["outliers"]:
            with st.expander(f"⚠️ Outliers detected in {len(results['outliers'])} column(s)"):
                for col, info in results["outliers"].items():
                    st.write(f"**{col}**: {info['count']} outliers ({info['pct']}%) — outside range [{info['lower_bound']}, {info['upper_bound']}]")

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
        if results["missing_matrix_chart"]:
            st.plotly_chart(results["missing_matrix_chart"], use_container_width=True)
        else:
            st.success("No missing values found in this dataset! 🎉")

    with tab5:
        with st.spinner("Generating insights..."):
            insights = generate_insights(
                results["summary"],
                results["column_types"],
                results["numeric_stats"],
                results["categorical_stats"]
            )
        st.markdown(insights)

    with tab6:
        st.markdown("Ask a question about your data in plain English, and get an instant chart or answer.")
        st.caption("Examples: \"show me sales trend over time\", \"top 5 categories by count\", \"distribution of price\"")

        user_question = st.text_input("Your question", placeholder="e.g. What are the top 5 values by count?", key="ask_data_input")

        if st.button("Ask", key="ask_data_button") and user_question.strip():
            with st.spinner("Thinking..."):
                try:
                    result = ask_data_question(df, user_question)
                except Exception as e:
                    st.error(f"Something went wrong talking to the AI: {e}")
                    result = None

            if result:
                st.markdown(f"**Answer:** {result.get('answer_text', '')}")

                chart_type = result.get("chart_type")
                x_col = result.get("x")
                y_col = result.get("y")
                agg = result.get("agg")

                try:
                    if chart_type == "bar" and x_col:
                        if y_col and agg:
                            agg_df = df.groupby(x_col)[y_col].agg(agg).reset_index()
                            fig = px.bar(agg_df, x=x_col, y=y_col, title=f"{agg.title()} of {y_col} by {x_col}")
                        else:
                            counts = df[x_col].value_counts().reset_index()
                            counts.columns = [x_col, "count"]
                            fig = px.bar(counts, x=x_col, y="count", title=f"Count of {x_col}")
                        st.plotly_chart(fig, use_container_width=True)

                    elif chart_type == "line" and x_col and y_col:
                        line_df = df.sort_values(by=x_col)
                        if agg:
                            line_df = line_df.groupby(x_col)[y_col].agg(agg).reset_index()
                        fig = px.line(line_df, x=x_col, y=y_col, title=f"{y_col} over {x_col}")
                        st.plotly_chart(fig, use_container_width=True)

                    elif chart_type == "scatter" and x_col and y_col:
                        fig = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
                        st.plotly_chart(fig, use_container_width=True)

                    elif chart_type == "histogram" and x_col:
                        fig = px.histogram(df, x=x_col, title=f"Distribution of {x_col}")
                        st.plotly_chart(fig, use_container_width=True)

                    elif chart_type == "pie" and x_col:
                        counts = df[x_col].value_counts().reset_index()
                        counts.columns = [x_col, "count"]
                        fig = px.pie(counts, names=x_col, values="count", title=f"Share of {x_col}")
                        st.plotly_chart(fig, use_container_width=True)

                    elif chart_type == "none":
                        pass  # answer_text already shown above

                except Exception as e:
                    st.warning(f"Couldn't build a chart for that: {e}")
else:
    st.info("👆 Upload a CSV, Excel, or JSON file to get started.")