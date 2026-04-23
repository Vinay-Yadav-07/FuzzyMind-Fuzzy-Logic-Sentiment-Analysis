"""
Gradio UI — Fuzzy Logic Sentiment Analyzer
Soft Computing Project | Amazon Reviews Dataset
Run: python app/gradio_app.py
Opens at: http://localhost:7860
"""

import gradio as gr
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.fuzzy_engine import analyze_sentiment, build_fis, extract_features
import skfuzzy as fuzz

# ─────────────────────────────────────────────
#  Load Dataset
# ─────────────────────────────────────────────
CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "reviews.csv")
try:
    df_all = pd.read_csv(CSV, low_memory=False)
    df_all = df_all[['name','reviews.text','reviews.rating']].copy()
    df_all.columns = ['product','review_text','rating']
    df_all.dropna(subset=['review_text','rating'], inplace=True)
    df_all['rating']      = pd.to_numeric(df_all['rating'], errors='coerce')
    df_all.dropna(subset=['rating'], inplace=True)
    df_all['review_text'] = df_all['review_text'].astype(str).str.strip()
    df_all = df_all[df_all['review_text'].str.len() > 10].reset_index(drop=True)
    DATA_LOADED = True
except Exception as e:
    print(f"Dataset load error: {e}")
    DATA_LOADED = False
    df_all = pd.DataFrame(columns=['product','review_text','rating'])

# FIS & MF
fis_s, mf = build_fis()
universe   = mf["universe"]
su         = mf["sentiment_universe"]

CMAP = {"Very Negative":"#ef4444","Negative":"#f97316","Neutral":"#facc15",
        "Positive":"#86efac","Very Positive":"#22c55e"}

# ─────────────────────────────────────────────
#  HELPER PLOTS
# ─────────────────────────────────────────────
def sc_color(s):
    if s>=80: return "#22c55e"
    if s>=60: return "#86efac"
    if s>=40: return "#facc15"
    if s>=20: return "#f97316"
    return "#ef4444"

def dark_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor("#1e293b")
    if title:   ax.set_title(title, color="#38bdf8", fontweight="bold", fontsize=11)
    if xlabel:  ax.set_xlabel(xlabel, color="#94a3b8")
    if ylabel:  ax.set_ylabel(ylabel, color="#94a3b8")
    ax.tick_params(colors="#94a3b8")
    ax.grid(True, color="#334155", alpha=0.4)
    for sp in ax.spines.values(): sp.set_edgecolor("#334155")


# ─────────────────────────────────────────────
#  ANALYZE FUNCTION → returns all Gradio outputs
# ─────────────────────────────────────────────
def run_analysis(text):
    if not text or not text.strip():
        return ("⚠️ Please enter some text.", None, None, None)

    r  = analyze_sentiment(text)
    sc = r["score"]
    clr = sc_color(sc)

    # ── Text output ─────────────────────────────
    summary = f"""
## {r['emoji']} Sentiment: **{r['label']}** — Score: **{sc:.1f} / 100**

| Feature | Value |
|---|---|
| Fuzzy Score | **{sc:.2f}** / 100 |
| Sentiment Label | **{r['label']}** |
| Positive Input | **{r['features']['pos_score']:.3f}** / 10 |
| Negative Input | **{r['features']['neg_score']:.3f}** / 10 |
| Punct. Input   | **{r['features']['punct_score']:.3f}** / 10 |
| Word Count     | **{r['features']['word_count']}** words |

### Output Membership Degrees:
| Label | Degree | Bar |
|---|---|---|
""" + "\n".join([
    f"| {lbl} | {deg:.3f} | {'█'*int(deg*20)}{' '*(20-int(deg*20))} |"
    for lbl, deg in r["memberships"].items()
])

    # ── Plot 1: Membership bars ───────────────
    fig1, ax = plt.subplots(figsize=(8, 4), facecolor="#0f172a")
    ax.set_facecolor("#1e293b")
    labels = list(r["memberships"].keys())
    values = [r["memberships"][l] for l in labels]
    colors = [CMAP[l] for l in labels]
    bars = ax.barh(labels, values, color=colors, edgecolor="#0f172a", height=0.55)
    for b, v in zip(bars, values):
        ax.text(v+0.01, b.get_y()+b.get_height()/2, f"{v:.3f}", va="center",
                color="#e2e8f0", fontsize=10, fontweight="bold")
    dark_ax(ax, f"Fuzzy Output Membership Degrees  [Score: {sc:.1f}]",
            "Membership Degree μ(x)")
    ax.set_xlim(0, 1.2)
    ax.axvline(x=max(values), color="#facc15", linestyle="--", linewidth=1, alpha=0.6)
    plt.tight_layout()
    fig1.savefig("/tmp/grad_membership.png", dpi=130, bbox_inches="tight", facecolor="#0f172a")
    plt.close(fig1)

    # ── Plot 2: Input feature radar ───────────
    fig2, ax2 = plt.subplots(figsize=(7, 5), facecolor="#0f172a")
    ax2.set_facecolor("#1e293b")
    feat_names = ["Positive\nScore", "Negative\nScore", "Punct.\nScore"]
    feat_vals  = [r["features"]["pos_score"], r["features"]["neg_score"], r["features"]["punct_score"]]
    feat_clrs  = ["#4ade80", "#f87171", "#facc15"]
    brs2 = ax2.bar(feat_names, feat_vals, color=feat_clrs, edgecolor="#0f172a", width=0.5)
    for b2, v2 in zip(brs2, feat_vals):
        ax2.text(b2.get_x()+b2.get_width()/2, v2+0.1, f"{v2:.2f}",
                 ha="center", color="#e2e8f0", fontweight="bold")
    dark_ax(ax2, "Fuzzy Input Values", "Feature", "Value (0–10)")
    ax2.set_ylim(0, 11)
    plt.tight_layout()
    fig2.savefig("/tmp/grad_inputs.png", dpi=130, bbox_inches="tight", facecolor="#0f172a")
    plt.close(fig2)

    # ── Plot 3: Score gauge ───────────────────
    fig3, ax3 = plt.subplots(figsize=(7, 3.5), facecolor="#0f172a")
    ax3.set_facecolor("#0f172a")
    # Background track
    ax3.barh(["Sentiment"], [100], color="#1e293b", height=0.4, edgecolor="#334155")
    ax3.barh(["Sentiment"], [sc], color=clr, height=0.4, edgecolor="#0f172a")
    ax3.axvline(50, color="#64748b", linestyle="--", linewidth=1, alpha=0.7)
    ax3.text(sc/2, 0, f"{sc:.1f}", ha="center", va="center",
             color="#0f172a", fontsize=18, fontweight="bold")
    # Zone labels
    for zone, pos, clrz in [("Very Neg",5,"#ef4444"),("Neg",15,"#f97316"),
                             ("Neutral",35,"#facc15"),("Pos",55,"#86efac"),
                             ("Very Pos",80,"#22c55e")]:
        ax3.text(pos+5, 0.35, zone, fontsize=7, color=clrz, ha="center", alpha=0.7)
    ax3.set_xlim(0, 100); ax3.set_ylim(-0.5, 0.7)
    ax3.axis("off")
    ax3.set_title(f"{r['emoji']}  {r['label']}  —  Fuzzy Score: {sc:.1f} / 100",
                  color=clr, fontsize=14, fontweight="bold", pad=10)
    fig3.patch.set_facecolor("#0f172a")
    plt.tight_layout()
    fig3.savefig("/tmp/grad_gauge.png", dpi=130, bbox_inches="tight", facecolor="#0f172a")
    plt.close(fig3)

    return summary, "/tmp/grad_gauge.png", "/tmp/grad_membership.png", "/tmp/grad_inputs.png"


# ─────────────────────────────────────────────
#  BATCH ANALYSIS FUNCTION
# ─────────────────────────────────────────────
def run_batch(texts_input, n_from_dataset):
    rows = []
    if texts_input and texts_input.strip():
        for line in texts_input.strip().split("\n"):
            if line.strip():
                r = analyze_sentiment(line.strip())
                rows.append({"Source":"Manual", "Text":line.strip()[:60],
                             "Fuzzy Score":r["score"], "Label":r["label"]})

    if DATA_LOADED and n_from_dataset > 0:
        sample = df_all.sample(min(n_from_dataset, len(df_all)), random_state=42)
        for _, row in sample.iterrows():
            r = analyze_sentiment(str(row["review_text"]))
            rows.append({"Source":f"Dataset ★{row['rating']:.0f}",
                         "Text":str(row["review_text"])[:60],
                         "Fuzzy Score":r["score"], "Label":r["label"]})

    if not rows:
        return pd.DataFrame(), None

    df_b = pd.DataFrame(rows)

    # Chart
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="#0f172a")
    axes[0].set_facecolor("#1e293b")
    n,bins,patches = axes[0].hist(df_b["Fuzzy Score"], bins=15, edgecolor="#0f172a")
    for patch,left in zip(patches,bins[:-1]):
        c=bins[1]-bins[0]; center=left+c/2
        if center>=80: patch.set_facecolor("#22c55e")
        elif center>=60: patch.set_facecolor("#86efac")
        elif center>=40: patch.set_facecolor("#facc15")
        elif center>=20: patch.set_facecolor("#f97316")
        else:             patch.set_facecolor("#ef4444")
    dark_ax(axes[0],"Fuzzy Score Distribution","Score (0-100)","Count")

    axes[1].set_facecolor("#1e293b")
    counts = df_b["Label"].value_counts()
    order  = [o for o in ["Very Positive","Positive","Neutral","Negative","Very Negative"] if o in counts.index]
    axes[1].pie(counts[order].values, labels=order, autopct="%1.0f%%",
                colors=[CMAP[k] for k in order],
                textprops={"color":"#e2e8f0"},
                wedgeprops={"edgecolor":"#0f172a","linewidth":2})
    axes[1].set_title("Sentiment Distribution", color="#38bdf8", fontweight="bold")
    plt.tight_layout()
    fig.savefig("/tmp/grad_batch.png", dpi=130, bbox_inches="tight", facecolor="#0f172a")
    plt.close(fig)

    return df_b, "/tmp/grad_batch.png"


# ─────────────────────────────────────────────
#  MF PLOT
# ─────────────────────────────────────────────
def show_mfs():
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor="#0f172a")
    fig.suptitle("Fuzzy Membership Functions — Triangular (trimf)", color="#38bdf8", fontsize=14, fontweight="bold")
    C = {"low":"#f87171","medium":"#facc15","high":"#4ade80",
         "very_negative":"#ef4444","negative":"#f97316",
         "neutral":"#facc15","positive":"#86efac","very_positive":"#22c55e"}
    def sa(ax,t,xl):
        ax.set_facecolor("#1e293b"); ax.set_title(t,color="#38bdf8",fontweight="bold",fontsize=10)
        ax.set_xlabel(xl,color="#94a3b8"); ax.set_ylabel("μ(x)",color="#94a3b8")
        ax.set_ylim(-0.05,1.15); ax.grid(True,color="#334155",alpha=0.4); ax.tick_params(colors="#94a3b8")
        for sp in ax.spines.values(): sp.set_edgecolor("#334155")
    for t in ["low","medium","high"]:
        axes[0,0].plot(universe,mf["pos_score"][t].mf,color=C[t],lw=2.5,label=t)
        axes[0,0].fill_between(universe,mf["pos_score"][t].mf,alpha=0.12,color=C[t])
    sa(axes[0,0],"Input: Positive Score","Score (0–10)"); axes[0,0].legend()
    for t in ["low","medium","high"]:
        axes[0,1].plot(universe,mf["neg_score"][t].mf,color=C[t],lw=2.5,label=t)
        axes[0,1].fill_between(universe,mf["neg_score"][t].mf,alpha=0.12,color=C[t])
    sa(axes[0,1],"Input: Negative Score","Score (0–10)"); axes[0,1].legend()
    axes[1,0].plot(universe,mf["punct_score"]["low"].mf,color="#f87171",lw=2.5,label="low")
    axes[1,0].fill_between(universe,mf["punct_score"]["low"].mf,alpha=0.12,color="#f87171")
    axes[1,0].plot(universe,mf["punct_score"]["high"].mf,color="#4ade80",lw=2.5,label="high")
    axes[1,0].fill_between(universe,mf["punct_score"]["high"].mf,alpha=0.12,color="#4ade80")
    sa(axes[1,0],"Input: Punctuation Score","Score (0–10)"); axes[1,0].legend()
    for t in ["very_negative","negative","neutral","positive","very_positive"]:
        axes[1,1].plot(su,mf["sentiment"][t].mf,color=C[t],lw=2.5,label=t.replace("_"," ").title())
        axes[1,1].fill_between(su,mf["sentiment"][t].mf,alpha=0.1,color=C[t])
    sa(axes[1,1],"Output: Sentiment Score","Sentiment (0–100)"); axes[1,1].legend(fontsize=9)
    plt.tight_layout()
    fig.savefig("/tmp/grad_mfs.png",dpi=130,bbox_inches="tight",facecolor="#0f172a")
    plt.close(fig)
    return "/tmp/grad_mfs.png"


# ─────────────────────────────────────────────
#  GRADIO INTERFACE
# ─────────────────────────────────────────────
THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.sky,
    secondary_hue=gr.themes.colors.violet,
    neutral_hue=gr.themes.colors.slate,
    font=gr.themes.GoogleFont("Inter"),
).set(
    body_background_fill="#0f172a",
    body_text_color="#e2e8f0",
    border_color_primary="#334155",
    block_background_fill="#1e293b",
    block_border_color="#334155",
    block_label_text_color="#94a3b8",
    button_primary_background_fill="linear-gradient(135deg,#3b82f6,#6366f1)",
    button_primary_text_color="white",
    input_background_fill="#0f172a",
    input_border_color="#334155",
)

SAMPLE_TEXTS = [
    "This Amazon Fire tablet is absolutely fantastic! Super easy to use, great battery life and amazing display quality.",
    "Terrible product! Broken on arrival, horrible customer service and complete waste of money.",
    "It is a decent tablet for the price. Nothing special but does what I need.",
    "Not bad at all! Works great for reading books and watching videos. Very happy with this purchase!",
    "Disgusting quality, slow performance, annoying ads all the time. Worst purchase ever!",
]

with gr.Blocks(theme=THEME, title="🧠 FuzzyMind — Gradio UI") as demo:
    gr.HTML("""
    <div style='text-align:center;padding:2rem;background:linear-gradient(135deg,#0f172a,#1e1b4b,#0f172a);
                border:1px solid #334155;border-radius:16px;margin-bottom:1rem;'>
      <h1 style='font-size:2.2rem;font-weight:800;color:#f8fafc;margin:0;'>
        🧠 <span style='color:#38bdf8;'>FuzzyMind</span> Gradio UI
      </h1>
      <p style='color:#94a3b8;margin:.5rem 0 0;'>
        Soft Computing · Mamdani FIS · Fuzzy Sets · Amazon Reviews Dataset
      </p>
    </div>""")

    with gr.Tabs():
        # ── Tab 1: Single Analysis ────────────────
        with gr.TabItem("🔍 Analyze Review"):
            with gr.Row():
                with gr.Column(scale=1):
                    text_input = gr.Textbox(
                        label="Enter Review Text",
                        placeholder="Type or paste an Amazon product review here...",
                        lines=6,
                        value=SAMPLE_TEXTS[0]
                    )
                    analyze_btn = gr.Button("⚡ Run Fuzzy Inference", variant="primary", size="lg")

                    gr.Markdown("**Quick Samples:**")
                    sample_btns = []
                    for s in SAMPLE_TEXTS:
                        b = gr.Button(s[:65]+"...", size="sm")
                        sample_btns.append((b, s))

                with gr.Column(scale=1):
                    result_md   = gr.Markdown(label="Result")
                    gauge_img   = gr.Image(label="Sentiment Gauge", type="filepath")

            with gr.Row():
                membership_img = gr.Image(label="Membership Degrees", type="filepath")
                inputs_img     = gr.Image(label="Fuzzy Input Values", type="filepath")

            analyze_btn.click(run_analysis, inputs=text_input,
                              outputs=[result_md, gauge_img, membership_img, inputs_img])

            for btn, txt in sample_btns:
                btn.click(lambda t=txt: run_analysis(t),
                          outputs=[result_md, gauge_img, membership_img, inputs_img])

        # ── Tab 2: Batch Analysis ─────────────────
        with gr.TabItem("📋 Batch Analysis"):
            with gr.Row():
                with gr.Column():
                    batch_texts = gr.Textbox(
                        label="Enter multiple reviews (one per line)",
                        placeholder="Review 1\nReview 2\nReview 3...",
                        lines=8
                    )
                    n_dataset = gr.Slider(0, 500, value=100, step=50,
                                         label="Also pull N reviews from Amazon dataset")
                    batch_btn = gr.Button("🚀 Analyze All", variant="primary")

            batch_table = gr.DataFrame(label="Results Table")
            batch_chart = gr.Image(label="Distribution Chart", type="filepath")
            batch_btn.click(run_batch, inputs=[batch_texts, n_dataset],
                            outputs=[batch_table, batch_chart])

        # ── Tab 3: Membership Functions ───────────
        with gr.TabItem("📊 Membership Functions"):
            gr.Markdown("### Fuzzy Membership Functions (Triangular trimf)")
            mf_btn = gr.Button("📊 Show MF Plots", variant="secondary")
            mf_img = gr.Image(label="Membership Functions", type="filepath")
            mf_btn.click(show_mfs, outputs=mf_img)
            demo.load(show_mfs, outputs=mf_img)

        # ── Tab 4: About ──────────────────────────
        with gr.TabItem("📚 About FIS"):
            gr.Markdown("""
## 🧠 Soft Computing — Mamdani Fuzzy Inference System

### Fuzzy Variables

| Variable | Type | Terms |
|---|---|---|
| positive_score | Antecedent | low, medium, high |
| negative_score | Antecedent | low, medium, high |
| punctuation_score | Antecedent | low, high |
| sentiment | Consequent | very_negative, negative, neutral, positive, very_positive |

### 15 Fuzzy Rules
```
R1 : IF pos=HIGH   AND neg=LOW                 → VERY POSITIVE
R2 : IF pos=HIGH   AND neg=LOW  AND punct=HIGH → VERY POSITIVE
R3 : IF pos=HIGH   AND neg=MED                 → POSITIVE
R4 : IF pos=MED    AND neg=LOW                 → POSITIVE
R5 : IF pos=MED    AND neg=LOW  AND punct=HIGH → POSITIVE
R6 : IF pos=MED    AND neg=MED                 → NEUTRAL
R7 : IF pos=LOW    AND neg=LOW                 → NEUTRAL
R8 : IF pos=LOW    AND neg=LOW  AND punct=LOW  → NEUTRAL
R9 : IF neg=HIGH   AND pos=LOW                 → VERY NEGATIVE
R10: IF neg=HIGH   AND pos=MED                 → NEGATIVE
R11: IF neg=MED    AND pos=LOW                 → NEGATIVE
R12: IF neg=HIGH   AND punct=HIGH              → VERY NEGATIVE
R13: IF pos=HIGH   AND neg=HIGH                → NEUTRAL
R14: IF pos=LOW    AND neg=MED                 → NEGATIVE
R15: IF pos=MED    AND neg=HIGH                → NEGATIVE
```

### Pipeline
`Text → Feature Extraction → Fuzzification → Rule Evaluation → Aggregation → Defuzzification → Label`

### Dataset
Amazon Product Reviews — **34,660 reviews** across 48 products
            """)

    gr.HTML("""
    <div style='text-align:center;color:#475569;font-size:.78rem;padding:1rem;border-top:1px solid #334155;margin-top:1rem;'>
      🧠 FuzzyMind Gradio UI — Soft Computing Project | Mamdani FIS · Triangular MFs · Amazon Reviews
    </div>""")


if __name__ == "__main__":
    print("🚀 Starting FuzzyMind Gradio UI...")
    print("   Opens at: http://localhost:7860")
    demo.launch(server_port=7860, share=False, show_error=True)
