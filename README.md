#  LLM-Powered Exploratory Data Analysis (EDA) App

An AI-powered EDA tool that runs **entirely on your local machine** — no cloud, no API keys, no cost. Upload any CSV and instantly get automatic data cleaning, statistical summaries, visualizations, and AI-generated insights.

## Features
- **Auto Data Cleaning** — fills missing numeric values with median, text values with mode
- **Statistical Summary** — full `describe()` output with shape, column types, and distributions
- **Smart Visualizations** — histogram per numeric column + correlation heatmap
- **AI Insights** — powered by a local LLM via Ollama (no internet required)
- **Simple UI** — clean Gradio web interface, just upload and click

## Demo
> Upload a CSV → Click Analyze → Get a full EDA report + charts in seconds

## Tech Stack

| Tool                                                                     | Purpose                   |
| ------------------------------------------------------------------------ | ------------------------- |
| [Gradio](https://gradio.app/)                                               | Web UI                    |
| [Pandas](https://pandas.pydata.org/)                                        | Data loading & statistics |
| [Matplotlib](https://matplotlib.org/) + [Seaborn](https://seaborn.pydata.org/) | Visualizations            |
| [Ollama](https://ollama.com/)                                               | Local LLM inference       |
| `tinyllama` / `phi3` / `mistral`                                   | AI insight generation     |

## Setup & Installation

### 1. Install Python Dependencies

```bash
pip install gradio pandas matplotlib seaborn ollama
```
### 2. Install Ollama

Download from [https://ollama.com](https://ollama.com) and install for your OS.

### 3. Pull a Local LLM Model
Choose a model based on your available RAM:

| Model         | Size   | Speed        | Quality                |
| ------------- | ------ | ------------ | ---------------------- |
| `tinyllama` | 600 MB | ⚡ Very Fast | Basic                  |
| `phi3`      | 2.3 GB | 🚀 Fast      | Good ←*recommended* |
| `mistral`   | 4.1 GB | 🐢 Slow      | Best                   |

```bash
ollama pull tinyllama   # or phi3, mistral
```
### 4. Configure the Model

In `app.py`, update line:

```python
MODEL_NAME = "tinyllama"  # change to phi3 or mistral if you have more RAM
```
### 5. Run the App

```bash
python app.py
```
Then open your browser at `http://localhost:7860`

## Project Structure

```
llm-eda-app/
│
├── app.py          # Main application
└── README.md       # This file
```
## How It Works

1. **Upload** a `.csv` file via the Gradio UI
2. **Clean** — missing values are auto-filled (median for numbers, mode for text)
3. **Analyze** — statistical summary is generated with Pandas
4. **Visualize** — up to 6 histograms + 1 correlation heatmap are rendered
5. **AI Insights** — a compact summary is sent to your local LLM, which returns 4 bullet-point insights about the dataset

## Note

- Only the **first 10 numeric columns** are sent to the AI to keep responses fast
- Charts are limited to **6 histograms** to avoid long render times on wide datasets
- All processing is done **locally** — your data never leaves your machine

## 🙋 Author

Built with ❤️ using Gradio + Ollama. Perfect **Data Science enthusiasts** exploring EDA with local AI.
