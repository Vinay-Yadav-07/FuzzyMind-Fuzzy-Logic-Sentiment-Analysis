# 🧠 FuzzyMind — Complete Run Guide
## Soft Computing Project | Fuzzy Logic Sentiment Analysis
## Dataset: Amazon Product Reviews (1429_1.csv → 34,660 rows)

---

## 📁 Project Structure

```
fuzzy_project/
│
├── 📓 notebooks/
│   └── Fuzzy_Sentiment_Analysis.ipynb   ← Main Jupyter Notebook
│
├── 🖥️ app/
│   ├── streamlit_gradio_app.py          ← Streamlit + Gradio UI
│   ├── gradio_app.py                    ← Standalone Gradio UI
│   └── index.html                       ← Website UI (open directly in browser)
│
├── ⚙️ utils/
│   ├── __init__.py
│   └── fuzzy_engine.py                  ← Core Fuzzy Logic Engine
│
├── 📊 data/
│   └── reviews.csv                      ← Amazon Reviews Dataset
│
├── 📈 outputs/                          ← Auto-generated charts & CSV
│
├── requirements.txt
└── HOW_TO_RUN.md                        ← This file
```

---

## ✅ STEP 1 — Install All Packages

```bash
pip install scikit-fuzzy numpy matplotlib pandas scikit-learn streamlit gradio seaborn notebook ipykernel
```

Or use requirements file:
```bash
pip install -r requirements.txt
```

---

## ✅ STEP 2 — Run Jupyter Notebook (Core Project)

```bash
cd fuzzy_project
jupyter notebook notebooks/Fuzzy_Sentiment_Analysis.ipynb
```

**In Jupyter: Click Kernel → Restart & Run All**

OR with VS Code: Open the .ipynb file → click "Run All"

### What the notebook does:
1. Loads the Amazon Reviews dataset (34,660 rows)
2. Defines fuzzy linguistic variables & MFs
3. Builds the Mamdani FIS with 15 rules
4. Applies fuzzy inference to 2000 sample reviews
5. Evaluates accuracy vs star ratings
6. Visualizes: MFs, results, 3D surface, confusion matrix

---

## ✅ STEP 3 — Run Streamlit UI

```bash
cd fuzzy_project
streamlit run app/streamlit_gradio_app.py
```

Opens at: **http://localhost:8501**

### Streamlit Tabs:
| Tab | Feature |
|---|---|
| 🔍 Analyze Text | Analyze any review with full FIS |
| 📊 Dataset Explorer | Browse the 34,660 Amazon reviews |
| 📈 Membership Functions | Visualize fuzzy sets + 3D surface |
| 🏆 Product Insights | Run FIS on dataset sample, rank products |
| 📚 FIS Knowledge Base | View all rules and system details |

---

## ✅ STEP 4 — Run Gradio UI

```bash
cd fuzzy_project
python app/gradio_app.py
```

Opens at: **http://localhost:7860**

### Gradio Tabs:
| Tab | Feature |
|---|---|
| 🔍 Analyze Review | Single review with score, gauge, membership charts |
| 📋 Batch Analysis | Multiple reviews + dataset sample |
| 📊 Membership Functions | MF visualization |
| 📚 About FIS | Full documentation |

---

## ✅ STEP 5 — Open Website UI

Simply open `app/index.html` in any browser — **no server needed!**

Double-click the file or:
```bash
# Windows
start app/index.html

# macOS
open app/index.html

# Linux
xdg-open app/index.html
```

### Website Features:
- Real-time fuzzy inference (runs in browser)
- Animated pipeline visualization
- Membership function canvas charts
- Sample texts from dataset categories
- Complete FIS rule table
- Dataset statistics

---

## 🧠 Soft Computing Concepts Used

| Concept | Implementation |
|---|---|
| **Fuzzy Sets** | 3 input vars (pos/neg/punct), each with linguistic terms |
| **Membership Functions** | Triangular (trimf) for all input & output vars |
| **Fuzzy Inference System** | Mamdani FIS (most common type) |
| **Fuzzification** | Convert crisp scores → fuzzy degrees |
| **Rule Evaluation** | AND (min) operator across 15 rules |
| **Aggregation** | OR (max) of all activated rule outputs |
| **Defuzzification** | Centroid / Center of Gravity method |
| **Output** | Crisp score 0-100 → 5 sentiment levels |

---

## 🔧 Common Errors & Fixes

### Error: `ModuleNotFoundError: No module named 'skfuzzy'`
```bash
pip install scikit-fuzzy
```

### Error: `ModuleNotFoundError: No module named 'gradio'`
```bash
pip install gradio
```

### Error: `FileNotFoundError: data/reviews.csv`
Make sure the CSV file is in the `data/` folder named `reviews.csv`

### Error: `Port 8501 already in use`
```bash
streamlit run app/streamlit_gradio_app.py --server.port 8502
```

### Error: `Port 7860 already in use`
Edit `gradio_app.py` and change `server_port=7860` to `server_port=7861`

### Notebook: `utils.fuzzy_engine not found`
Run the notebook from inside `fuzzy_project/` directory, not a subdirectory.

---

## 📊 Output Files Generated

| File | Description |
|---|---|
| `outputs/eda_overview.png` | Dataset EDA charts |
| `outputs/membership_functions.png` | All 4 MF plots |
| `outputs/dataset_results.png` | Score distribution, label pie, box plot |
| `outputs/confusion_matrix.png` | Accuracy evaluation |
| `outputs/surface_3d.png` | 3D FIS control surface |
| `outputs/top_products.png` | Product ranking by fuzzy score |
| `outputs/fuzzy_results.csv` | Full results table |

---

*Soft Computing Project — FuzzyMind | Mamdani FIS · Triangular MFs · Amazon Reviews*
