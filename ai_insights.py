import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate_insights(summary, column_types, numeric_stats, categorical_stats):
    """Send dataset summary to Groq and get plain-English insights."""

    prompt = f"""
You are a data analyst. Based on this dataset summary, write 3-4 short, 
plain-English insights a business person would find useful. Keep each 
insight to 1-2 sentences. Do not repeat raw numbers robotically — 
explain what they mean.

Dataset summary:
- Rows: {summary['n_rows']}
- Columns: {summary['n_cols']}
- Duplicate rows: {summary['duplicates']}
- Missing data: {summary['missing_total_pct']}%

Column types: {column_types}

Numeric column stats: {numeric_stats}

Categorical column stats: {categorical_stats}

Write the insights as a bullet list.
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Could not generate AI insights: {e}"


def ask_data_question(df, question):
    """
    Takes a natural-language question about the dataframe and returns a dict:
    {
        "chart_type": "bar" | "line" | "scatter" | "histogram" | "pie" | "none",
        "x": <column name or None>,
        "y": <column name or None>,
        "agg": "sum" | "mean" | "count" | "median" | None,
        "answer_text": "<short plain-English answer/explanation>"
    }
    If the question can't be answered with a chart, chart_type will be "none"
    and answer_text will contain a direct text answer instead.
    """
    columns_info = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        sample_vals = df[col].dropna().unique()[:5].tolist()
        columns_info.append(f"- {col} ({dtype}), sample values: {sample_vals}")

    schema_text = "\n".join(columns_info)

    system_prompt = f"""You are a data analyst assistant. The user has a pandas dataframe with these columns:

{schema_text}

The dataframe has {len(df)} rows.

Given a natural-language question, respond with ONLY a JSON object (no markdown, no explanation outside the JSON) in this exact format:
{{
    "chart_type": "bar" | "line" | "scatter" | "histogram" | "pie" | "none",
    "x": "<exact column name from the list above, or null>",
    "y": "<exact column name from the list above, or null>",
    "agg": "sum" | "mean" | "count" | "median" | null,
    "answer_text": "<a short 1-2 sentence plain-English answer or explanation>"
}}

Rules:
- Only use column names that exist in the schema above, exactly as spelled.
- If the question asks for a trend/comparison over time, prefer "line" chart with the date/time column as x.
- If comparing categories, prefer "bar" chart with agg set.
- If showing distribution of one numeric column, use "histogram" with x set and y null.
- If the question cannot be visualized (e.g. "what does this dataset contain"), set chart_type to "none" and put the full answer in answer_text.
- Respond with ONLY the JSON object, nothing else.
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            temperature=0.2,
            max_tokens=500
        )
        raw_text = response.choices[0].message.content.strip()
    except Exception as e:
        return {
            "chart_type": "none",
            "x": None,
            "y": None,
            "agg": None,
            "answer_text": f"Could not reach the AI: {e}"
        }

    # Strip markdown code fences if the model added them anyway
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.lower().startswith("json"):
            raw_text = raw_text[4:].strip()

    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError:
        result = {
            "chart_type": "none",
            "x": None,
            "y": None,
            "agg": None,
            "answer_text": "Sorry, I couldn't understand how to answer that question. Try rephrasing it."
        }

    return result