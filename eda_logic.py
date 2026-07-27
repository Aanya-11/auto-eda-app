import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio


def detect_column_types(df):
    """Classify each column as numeric, categorical, datetime, or text."""
    column_types = {}
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            column_types[col] = "datetime"
            continue

        if df[col].dtype == "object":
            try:
                pd.to_datetime(df[col], errors="raise")
                column_types[col] = "datetime"
                continue
            except (ValueError, TypeError):
                pass

        if pd.api.types.is_numeric_dtype(df[col]):
            column_types[col] = "numeric"
            continue

        unique_ratio = df[col].nunique() / len(df[col])
        if unique_ratio < 0.05 or df[col].nunique() < 20:
            column_types[col] = "categorical"
        else:
            column_types[col] = "text"

    return column_types


def apply_filters(df, filters):
    """Apply a dictionary of filters to the dataframe."""
    filtered_df = df.copy()

    for col, condition in filters.items():
        if condition["type"] == "categorical":
            if condition["values"]:
                filtered_df = filtered_df[filtered_df[col].isin(condition["values"])]
        elif condition["type"] == "numeric":
            low, high = condition["range"]
            filtered_df = filtered_df[(filtered_df[col] >= low) & (filtered_df[col] <= high)]

    return filtered_df


def apply_imputation(df, strategy_map):
    """Fill missing values in selected columns using the chosen strategy per column."""
    imputed_df = df.copy()

    for col, strategy in strategy_map.items():
        if strategy == "mean":
            imputed_df[col] = imputed_df[col].fillna(imputed_df[col].mean())
        elif strategy == "median":
            imputed_df[col] = imputed_df[col].fillna(imputed_df[col].median())
        elif strategy == "mode":
            mode_val = imputed_df[col].mode()
            if not mode_val.empty:
                imputed_df[col] = imputed_df[col].fillna(mode_val[0])
        elif strategy == "drop_rows":
            imputed_df = imputed_df.dropna(subset=[col])

    return imputed_df


def apply_transformations(df, transform_map):
    """Apply log scale, normalization, label encoding, or one-hot encoding to selected columns."""
    transformed_df = df.copy()

    for col, method in transform_map.items():
        if method == "log":
            min_val = transformed_df[col].min()
            if min_val <= 0:
                transformed_df[col] = np.log1p(transformed_df[col] - min_val + 1)
            else:
                transformed_df[col] = np.log1p(transformed_df[col])

        elif method == "normalize":
            min_val = transformed_df[col].min()
            max_val = transformed_df[col].max()
            if max_val != min_val:
                transformed_df[col] = (transformed_df[col] - min_val) / (max_val - min_val)

        elif method == "onehot":
            dummies = pd.get_dummies(transformed_df[col], prefix=col)
            transformed_df = pd.concat([transformed_df.drop(columns=[col]), dummies], axis=1)

        elif method == "label_encode":
            transformed_df[col] = transformed_df[col].astype("category").cat.codes

        elif method == "skip":
            continue

    return transformed_df


def get_summary(df):
    """Overall dataset summary."""
    return {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "duplicates": int(df.duplicated().sum()),
        "missing_total_pct": round(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100, 2)
    }


def get_numeric_stats(df, col):
    return {
        "mean": round(df[col].mean(), 2),
        "median": round(df[col].median(), 2),
        "std": round(df[col].std(), 2),
        "min": df[col].min(),
        "max": df[col].max(),
        "missing_pct": round(df[col].isnull().sum() / len(df) * 100, 2)
    }


def get_categorical_stats(df, col):
    mode_series = df[col].mode()
    return {
        "unique_count": df[col].nunique(),
        "top_value": mode_series[0] if not mode_series.empty else None,
        "top_value_pct": round(df[col].value_counts(normalize=True).iloc[0] * 100, 2) if df[col].nunique() > 0 else 0,
        "missing_pct": round(df[col].isnull().sum() / len(df) * 100, 2)
    }


# ---------------------------------------------------------------------------
# NEW: shared chart styling helper (does not alter any existing logic above)
# ---------------------------------------------------------------------------
def style_fig(fig, x_title=None, y_title=None):
    """Apply consistent, clean styling to any Plotly figure used in the app."""
    fig.update_layout(
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="#fafafa", size=13, family="Segoe UI, sans-serif"),
        title=dict(font=dict(size=18, color="#fafafa")),
        margin=dict(l=50, r=30, t=60, b=50),
        bargap=0.15,
        xaxis=dict(
            title=x_title,
            showgrid=False,
            zeroline=False,
            linecolor="#2a2e37",
        ),
        yaxis=dict(
            title=y_title,
            showgrid=True,
            gridcolor="#1c1f26",
            zeroline=False,
        ),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def plot_numeric(df, col):
    fig = px.histogram(df, x=col, title=f"Distribution of {col}", color_discrete_sequence=["#14b8a6"])
    return style_fig(fig, x_title=col, y_title="Count")


def plot_categorical(df, col, top_n=10):
    counts = df[col].value_counts().nlargest(top_n).reset_index()
    counts.columns = [col, "count"]
    fig = px.bar(counts, x=col, y="count", title=f"Top values in {col}", color_discrete_sequence=["#60a5fa"])
    return style_fig(fig, x_title=col, y_title="Count")


def plot_correlation(df):
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        return None
    corr = numeric_df.corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="Teal", title="Correlation Heatmap")
    return style_fig(fig)


def plot_missing_matrix(df):
    """Visualize missing values across all columns."""
    missing_data = df.isnull()
    if missing_data.sum().sum() == 0:
        return None

    missing_numeric = missing_data.astype(int)

    fig = px.imshow(
        missing_numeric.T,
        aspect="auto",
        color_continuous_scale=["#1c1f26", "#ef4444"],
        title="Missing Values Map (red = missing)",
        labels=dict(x="Row Index", y="Column", color="Missing")
    )
    fig.update_layout(coloraxis_showscale=False)
    return style_fig(fig)


def detect_outliers(df):
    """Flag outliers in numeric columns using the IQR method."""
    outlier_summary = {}
    numeric_cols = df.select_dtypes(include="number").columns

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_count = len(outliers)

        if outlier_count > 0:
            outlier_summary[col] = {
                "count": outlier_count,
                "pct": round(outlier_count / len(df) * 100, 2),
                "lower_bound": round(lower_bound, 2),
                "upper_bound": round(upper_bound, 2)
            }

    return outlier_summary


def run_eda(df):
    """Master function - runs everything and returns one results dictionary."""
    col_types = detect_column_types(df)
    summary = get_summary(df)

    numeric_cols = [c for c, t in col_types.items() if t == "numeric"]
    categorical_cols = [c for c, t in col_types.items() if t == "categorical"]

    numeric_stats = {}
    categorical_stats = {}
    numeric_charts = {}
    categorical_charts = {}

    for c in numeric_cols:
        try:
            numeric_stats[c] = get_numeric_stats(df, c)
            numeric_charts[c] = plot_numeric(df, c)
        except Exception:
            continue

    for c in categorical_cols:
        try:
            categorical_stats[c] = get_categorical_stats(df, c)
            categorical_charts[c] = plot_categorical(df, c)
        except Exception:
            continue

    correlation_chart = plot_correlation(df)
    missing_matrix_chart = plot_missing_matrix(df)
    outliers = detect_outliers(df)

    results = {
        "summary": summary,
        "column_types": col_types,
        "numeric_stats": numeric_stats,
        "categorical_stats": categorical_stats,
        "numeric_charts": numeric_charts,
        "categorical_charts": categorical_charts,
        "correlation_chart": correlation_chart,
        "missing_matrix_chart": missing_matrix_chart,
        "outliers": outliers
    }

    return results


def generate_html_report(df, results, ai_insights_text=""):
    """Build a single self-contained HTML report string with charts and stats embedded."""

    summary = results["summary"]

    html_parts = []

    html_parts.append(f"""
    <html>
    <head>
        <meta charset="utf-8">
        <title>Auto-EDA Report</title>
        <style>
            body {{
                background-color: #0e1117;
                color: #fafafa;
                font-family: -apple-system, Segoe UI, Roboto, sans-serif;
                padding: 2rem;
                max-width: 1000px;
                margin: auto;
            }}
            h1 {{
                background: linear-gradient(90deg, #2dd4bf, #60a5fa, #a78bfa);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }}
            h2 {{
                border-left: 4px solid #14b8a6;
                padding-left: 0.6rem;
                margin-top: 2rem;
            }}
            .metric-row {{
                display: flex;
                gap: 1rem;
                margin: 1rem 0;
            }}
            .metric-card {{
                background: #1c1f26;
                border: 1px solid #2a2e37;
                border-radius: 12px;
                padding: 1rem;
                flex: 1;
            }}
            .metric-value {{
                color: #2dd4bf;
                font-size: 1.5rem;
                font-weight: bold;
            }}
            .metric-label {{
                color: #9ca3af;
                font-size: 0.8rem;
                text-transform: uppercase;
            }}
            .insights-box {{
                background: #1c1f26;
                border: 1px solid #2a2e37;
                border-radius: 12px;
                padding: 1.2rem;
                white-space: pre-wrap;
            }}
        </style>
    </head>
    <body>
        <h1>📊 Auto-EDA Report</h1>
        <p>Generated automatically from your uploaded dataset.</p>

        <h2>Summary</h2>
        <div class="metric-row">
            <div class="metric-card"><div class="metric-value">{summary['n_rows']:,}</div><div class="metric-label">Rows</div></div>
            <div class="metric-card"><div class="metric-value">{summary['n_cols']}</div><div class="metric-label">Columns</div></div>
            <div class="metric-card"><div class="metric-value">{summary['duplicates']}</div><div class="metric-label">Duplicates</div></div>
            <div class="metric-card"><div class="metric-value">{summary['missing_total_pct']}%</div><div class="metric-label">Missing</div></div>
        </div>
    """)

    if ai_insights_text:
        html_parts.append(f"""
        <h2>🤖 AI-Generated Insights</h2>
        <div class="insights-box">{ai_insights_text}</div>
        """)

    if results["outliers"]:
        html_parts.append("<h2>⚠️ Outliers Detected</h2><ul>")
        for col, info in results["outliers"].items():
            html_parts.append(
                f"<li><b>{col}</b>: {info['count']} outliers ({info['pct']}%) "
                f"— outside range [{info['lower_bound']}, {info['upper_bound']}]</li>"
            )
        html_parts.append("</ul>")

    include_js = True

    if results["numeric_charts"]:
        html_parts.append("<h2>📈 Numeric Columns</h2>")
        for col, fig in results["numeric_charts"].items():
            chart_html = pio.to_html(fig, include_plotlyjs=("cdn" if include_js else False), full_html=False)
            html_parts.append(chart_html)
            include_js = False

    if results["categorical_charts"]:
        html_parts.append("<h2>🗂️ Categorical Columns</h2>")
        for col, fig in results["categorical_charts"].items():
            chart_html = pio.to_html(fig, include_plotlyjs=("cdn" if include_js else False), full_html=False)
            html_parts.append(chart_html)
            include_js = False

    if results["correlation_chart"]:
        html_parts.append("<h2>🔗 Correlation Heatmap</h2>")
        chart_html = pio.to_html(results["correlation_chart"], include_plotlyjs=("cdn" if include_js else False), full_html=False)
        html_parts.append(chart_html)
        include_js = False

    if results["missing_matrix_chart"]:
        html_parts.append("<h2>🕳️ Missing Values Map</h2>")
        chart_html = pio.to_html(results["missing_matrix_chart"], include_plotlyjs=("cdn" if include_js else False), full_html=False)
        html_parts.append(chart_html)
        include_js = False

    html_parts.append("</body></html>")

    return "".join(html_parts)