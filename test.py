import pandas as pd
from eda_logic import run_eda

df = pd.read_csv("netflix_vs_amazon_clean.csv")

print("---- Preview ----")
print(df.head())

print("\n---- Full EDA Results ----")
results = run_eda(df)
print(results["summary"])
print(results["column_types"])