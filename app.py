# ============================================================
# 📊 LLM-Powered Exploratory Data Analysis (EDA) App

# STEP 1: IMPORT LIBRARIES
# ──────────────────────────────────────────────────────────────
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ollama
import tempfile
import os

# STEP 2: SETTINGS
# ──────────────────────────────────────────────────────────────
# Following step for ollama
# 1.download ollama from website
# 2.open CMD(command prompt)
#   a.write ollama pull <modelName>
#   
# To download phi3, run in terminal:  ollama pull phi3
# To download tinyllama:              ollama pull tinyllama
#
#   note: choose below model as per your RAM memory
#   ModelName      Size    Speed       Quality
#   ----------    ----    --------    -------
#   tinyllama     600MB   Very Fast   Basic
#   phi3          2.3GB   Fast        Good  ← recommended for students
#   mistral       4.1GB   Slow        Best
#
#3.check installed model write -ollama list


MODEL_NAME = "tinyllama"   # i Used tinyllama as per my RAM

# Temp folder — works on Windows, Mac, Linux automatically
TEMP_DIR = tempfile.gettempdir()

# Max rows sent to AI — sending all rows makes it slow!
# We only send the statistics summary, not raw data rows.
MAX_SUMMARY_COLS = 10   # only describe first 10 columns if dataset is very wide

# STEP 3: HELPER FUNCTION — Clean the Data
# ──────────────────────────────────────────────────────────────
# Fill missing values so charts and stats don't break.
#   • Numbers → fill with MEDIAN
#   • Text    → fill with most common value (MODE)

def clean_data(df):
    for col in df.select_dtypes(include=['number']).columns:
        df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    return df

# STEP 4: HELPER FUNCTION — Ask Ollama for Insights
# ──────────────────────────────────────────────────────────────
# SPEED TIPS applied here:
#   1. We send only a SHORT summary (not all data)
#   2. We ask for a SHORT reply (3-4 bullet points only)
#   3. We limit columns sent to AI (MAX_SUMMARY_COLS)

def generate_ai_insights(df, column_names):
    """
    Send a compact dataset summary to Ollama and get quick insights.
    Keeping the prompt short = faster AI response!
    """

    # Only send numeric stats — cleaner and avoids long text columns
    numeric_cols = df.select_dtypes(include=['number']).columns[:MAX_SUMMARY_COLS]
    short_summary = df[numeric_cols].describe().round(2).to_string()

    # Plain column names only — no long descriptions
    col_names = ', '.join(column_names)

    # SHORT prompt = faster response from the model
    prompt = (
        f"I have a dataset with these columns: {col_names}\n\n"
        f"Numeric statistics:\n{short_summary}\n\n"
        "Answer in exactly 4 bullet points (one line each):\n"
        "- What this dataset is about\n"
        "- Most interesting pattern in the numbers\n"
        "- Any data quality concern\n"
        "- One analysis idea a student can try"
    )

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )

    return response['message']['content']


# STEP 5: HELPER FUNCTION — Draw Charts
# ──────────────────────────────────────────────────────────────
# SPEED TIP: We limit to max 6 histograms so it doesn't take
# forever on wide datasets with many columns.

def generate_visualizations(df):
    plot_paths = []
    numeric_cols = df.select_dtypes(include=['number']).columns[:6]  # max 6 charts

    # Histogram for each numeric column
    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(df[col], bins=30, kde=True, color="#4f86c6", ax=ax)
        ax.set_title(f"Distribution of '{col}'", fontsize=14, fontweight='bold')
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        plt.tight_layout()

        path = os.path.join(TEMP_DIR, f"{col}_distribution.png")
        fig.savefig(path, dpi=100)   # dpi=100 is faster than 120
        plot_paths.append(path)
        plt.close(fig)

    # Correlation Heatmap (only if 2+ numeric columns)
    all_numeric = df.select_dtypes(include=['number'])
    if len(all_numeric.columns) > 1:
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(
            all_numeric.corr(),
            annot=True,
            cmap='coolwarm',
            fmt=".2f",
            linewidths=0.5,
            ax=ax
        )
        ax.set_title("Correlation Heatmap", fontsize=14, fontweight='bold')
        plt.tight_layout()

        path = os.path.join(TEMP_DIR, "correlation_heatmap.png")
        fig.savefig(path, dpi=100)
        plot_paths.append(path)
        plt.close(fig)

    return plot_paths

# STEP 6: MAIN FUNCTION — Tie Everything Together
# ──────────────────────────────────────────────────────────────

def eda_analysis(file_obj):
    # 6a: Load
    df = pd.read_csv(file_obj)
    print(f"✅ Loaded: {df.shape[0]} rows x {df.shape[1]} columns")

    # 6b: Clean
    df = clean_data(df)

    # 6c: Statistics
    summary_str = df.describe(include='all').to_string()
    shape_info  = f"Rows: {df.shape[0]}  |  Columns: {df.shape[1]}"
    dtypes_info = df.dtypes.to_string()

    # 6d: AI Insights (short & fast)
    print(f"🤖 Asking {MODEL_NAME}...")
    insights = generate_ai_insights(df, list(df.columns))

    # 6e: Charts
    print("📊 Drawing charts...")
    plot_paths = generate_visualizations(df)

    # 6f: Build report
    report = f"""
{'='*60}
DATASET OVERVIEW
{'='*60}
{shape_info}

Columns & Data Types:
{dtypes_info}

{'='*60}
STATISTICAL SUMMARY
{'='*60}
{summary_str}

{'='*60}
AI INSIGHTS  (model: {MODEL_NAME})
{'='*60}
{insights}
"""
    return report, plot_paths

# STEP 7: GRADIO UI
# ──────────────────────────────────────────────────────────────

with gr.Blocks(title="LLM EDA App", theme=gr.themes.Soft()) as demo:

    gr.Markdown(f"""
    # 📊 LLM-Powered Exploratory Data Analysis
    **Model: `{MODEL_NAME}` via Ollama — runs locally on your PC**

    Upload any CSV to get automatic cleaning, statistics, charts, and AI insights.
    > ⚡ Tip: First run is slow because the model loads into memory. After that it's faster!
    """)

    file_input = gr.File(label="📁 Upload CSV file", file_types=[".csv"])

    analyze_btn = gr.Button("🔍 Analyze Dataset", variant="primary", size="lg")

    report_box = gr.Textbox(label="📝 EDA Report", lines=30, max_lines=60)

    gallery = gr.Gallery(label="📈 Visualizations", columns=2, height=500)

    analyze_btn.click(
        fn=eda_analysis,
        inputs=[file_input],
        outputs=[report_box, gallery]
    )

# STEP 8: LAUNCH
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch(share=False) # share=True creates a temporary public URL anyone one can open until 72 hrs
