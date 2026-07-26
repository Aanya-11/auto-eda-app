import os
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