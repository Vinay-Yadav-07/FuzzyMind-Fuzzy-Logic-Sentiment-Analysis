# 🧠 FuzzyMind: Fuzzy Logic Sentiment Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-orange)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**FuzzyMind** is a comprehensive Soft Computing project that implements a **Mamdani Fuzzy Inference System (FIS)** to perform sentiment analysis on product reviews. Unlike binary "positive/negative" classification, FuzzyMind provides a nuanced sentiment score by processing linguistic variables such as positive word frequency, negative word frequency, and punctuation intensity.

---

## 🚀 Key Features

- **Mamdani FIS Engine**: Built from scratch using `scikit-fuzzy` with 15 expert-defined rules.
- **Triangular Membership Functions**: Precise fuzzification of textile features.
- **Multi-Platform UI**: 
  - 🖥️ **Streamlit Dashboard**: In-depth analysis, dataset explorer, and 3D surface plots.
  - 🔍 **Gradio Interface**: Quick review testing with real-time gauge charts.
  - 🌐 **Static Web UI**: Lightweight HTML/JS interface for browser-based inference.
- **Data Visualization**: Automated generation of 3D control surfaces, membership functions, and confusion matrices.
- **Batch Processing**: Analyze large-scale datasets (Amazon Reviews - 35k rows) with high efficiency.

---

## 📁 Project Structure

```text
fuzzy_project/
├── 📓 notebooks/          # Exploratory Data Analysis & FIS Development
├── 🖥️ app/                # UI Interfaces (Streamlit, Gradio, HTML)
├── ⚙️ utils/              # Core Fuzzy Logic Engine (fuzzy_engine.py)
├── 📊 data/               # Amazon Reviews Raw Dataset
├── 📈 outputs/            # Auto-generated Visualizations & CSV Reports
├── requirements.txt       # Project Dependencies
└── README.md              # Project Documentation
```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/fuzzy-sentiment-analysis.git
cd fuzzy_project
```

### 2. Install Dependencies
It is recommended to use a virtual environment.
```bash
pip install -r requirements.txt
```

---

## 🖥️ How to Run

### Option 1: Streamlit Dashboard (Full Experience)
```bash
streamlit run app/streamlit_gradio_app.py
```
*Features: FIS Knowledge Base, 3D Surface Visualization, Product Insights.*

### Option 2: Gradio UI (Quick Analysis)
```bash
python app/gradio_app.py
```
*Features: Single & Batch review analysis with visual gauges.*

### Option 3: Jupyter Notebook (Analysis)
```bash
jupyter notebook notebooks/Fuzzy_Sentiment_Analysis.ipynb
```

### Option 4: Web Interface (No Server Needed)
Simply open `app/index.html` in your web browser.

---

## 🧠 Fuzzy Logic Configuration

| Component | Implementation |
|---|---|
| **Input Variables** | Positive score, Negative score, Punctuation intensity |
| **Membership Functions** | Triangular (trimf) |
| **Inference Method** | Mamdani FIS |
| **Rule Operators** | MIN (AND), MAX (OR) |
| **Defuzzification** | Centroid (Center of Gravity) |
| **Output** | Sentiment score (0-100) |

---

---

## 🔧 Troubleshooting & Common Fixes

### Error: `ModuleNotFoundError: No module named 'skfuzzy'`
Run: `pip install scikit-fuzzy`

### Error: `FileNotFoundError: data/reviews.csv`
Ensure the dataset is located at `data/reviews.csv`. If you are running the project from a subdirectory, move back to the root `fuzzy_project` folder.

### Error: `Port 8501 already in use` (Streamlit)
```bash
streamlit run app/streamlit_gradio_app.py --server.port 8502
```

### Notebook imports failing?
Always launch Jupyter from the root `fuzzy_project` directory so that it can correctly see the `utils` package.

---

## 📊 Visualizations

The system automatically generates several insights located in the `outputs/` folder:
- **3D Surface Plots**: Visualizing the decision surface of the FIS.
- **Membership Functions**: Graphs of linguistic terms.
- **Confusion Matrix**: Evaluation against actual star ratings.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Developed as a **Soft Computing (SC)** project at **Vidyalankar Institute of Technology**.
