# 📊 Auto-EDA App

An interactive web app that automates exploratory data analysis (EDA) — upload any CSV, Excel, or JSON file and instantly get summary statistics, visualizations, and AI-generated insights. No code required.

**Live App:** [auto-eda-app-m8bwyqgbsrz7774rzchn32.streamlit.app](https://auto-eda-app-m8bwyqgbsrz7774rzchn32.streamlit.app/)

---

## ✨ Features

- **Multi-format upload** — supports CSV, Excel (`.xlsx`, `.xls`), and JSON files
- **Interactive filtering** — filter rows by categorical or numeric columns before analysis
- **Missing value handling** — impute (mean/median/mode) or drop rows with missing data, per column
- **Column transformations** — log transform, normalization, one-hot encoding, and label encoding
- **Automated EDA** — row/column counts, duplicate detection, missing value %, outlier detection
- **Rich visualizations** — distribution charts for numeric and categorical columns, correlation heatmap, missing data matrix (powered by Plotly)
- **AI-generated insights** — natural language summary of the dataset's key patterns (powered by Groq's Llama API)
- **Ask Your Data** — chat-style Q&A over the computed statistics, without running code on your uploaded data
- **Downloadable report** — export the full analysis as an HTML report (printable to PDF)

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Frontend / App Framework | [Streamlit](https://streamlit.io/) |
| Data Processing | Pandas |
| Visualizations | Plotly |
| AI Insights | Groq API (Llama) |
| Hosting | Streamlit Community Cloud |

---

## 📂 Project Structure

```
eda-analyser/
├── app.py              # Main Streamlit app (UI + orchestration)
├── eda_logic.py         # Core EDA functions (stats, charts, filters, transforms)
├── ai_insights.py        # AI-powered insight generation and Q&A
├── requirements.txt      # Python dependencies
├── .streamlit/
│   └── config.toml       # App theme configuration
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- A Groq API key ([get one here](https://console.groq.com/))

### Installation

```bash
# Clone the repository
git clone https://github.com/Aanya-11/auto-eda-app.git
cd auto-eda-app

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Set up your API key

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_api_key_here
```

### Run the app

```bash
streamlit run app.py
```

> ⚠️ Always run this command directly in a terminal with the virtual environment active — using an IDE's "Run" button can cause issues with Streamlit's execution context.

---

## 📖 How It Works

1. **Upload** a dataset (CSV, Excel, or JSON)
2. **Optionally filter, clean, and transform** the data using the sidebar controls
3. The app **automatically profiles** the dataset — column types, summary stats, missing values, outliers
4. **Explore** results across tabs: Numeric, Categorical, Correlations, Missing Data
5. Read the **AI-generated insights** summarizing key patterns
6. **Ask questions** about your data in plain English
7. **Download** a complete HTML report of the analysis

---

## 🗺️ Roadmap

- [ ] Support for larger datasets with sampling/chunking
- [ ] Custom chart export (PNG/SVG)
- [ ] Multi-dataset comparison view

---

## 👤 Author

**Aanya Dubey**
B.Tech CSE (DATA SCIENCE)

- GitHub: [@Aanya-11](https://github.com/Aanya-11)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
