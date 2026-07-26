import pandas as pd
import plotly.express as px


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


def plot_numeric(df, col):
    return px.histogram(df, x=col, marginal="box", title=f"Distribution of {col}")


def plot_categorical(df, col, top_n=10):
    counts = df[col].value_counts().nlargest(top_n).reset_index()
    counts.columns = [col, "count"]
    return px.bar(counts, x=col, y="count", title=f"Top values in {col}")


def plot_correlation(df):
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        return None
    corr = numeric_df.corr()
    return px.imshow(corr, text_auto=".2f", color_continuous_scale="Teal", title="Correlation Heatmap")


def run_eda(df):
    """Master function — runs everything and returns one results dictionary."""
    col_types = detect_column_types(df)
    summary = get_summary(df)

    numeric_cols = [c for c, t in col_types.items() if t == "numeric"]
    categorical_cols = [c for c, t in col_types.items() if t == "categorical"]

    numeric_stats, categorical_stats = {}, {}
    numeric_charts, categorical_charts = {}, {}

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

    return {
        "summary": summary,
        "column_types": col_types,
        "numeric_stats": numeric_stats,
        "categorical_stats": categorical_stats,
        "numeric_charts": numeric_charts,
        "categorical_charts": categorical_charts,
        "correlation_chart": plot_correlation(df)
    }