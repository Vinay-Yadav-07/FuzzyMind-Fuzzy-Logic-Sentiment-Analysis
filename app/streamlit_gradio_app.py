"""
Streamlit + Gradio UI — Fuzzy Logic Sentiment Analyzer
Soft Computing Project | Amazon Reviews Dataset
Run: streamlit run app/streamlit_gradio_app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import sys, os, io
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.fuzzy_engine import analyze_sentiment, extract_features, build_fis, rating_to_label, score_to_3class
import skfuzzy as fuzz

# ─────────────────────────────────────────────
st.set_page_config(page_title="FuzzyMind — Sentiment Analyzer",
                   page_icon="🧠", layout="wide",
                   initial_sidebar_state="expanded")

DARK = "#0f172a"; CARD = "#1e293b"; BORDER = "#334155"
ACCENT = "#38bdf8"; TEXT = "#e2e8f0"; MUTED = "#94a3b8"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
*, *::before, *::after {{ font-family:'Inter',sans-serif; box-sizing:border-box; }}
.stApp {{ background:{DARK}; }}
[data-testid="stSidebar"] {{ background:{CARD}; border-right:1px solid {BORDER}; }}
[data-testid="stSidebar"] * {{ color:{TEXT} !important; }}
.hero-wrap {{
  background:linear-gradient(135deg,#0f172a 0%,#1e1b4b 40%,#0c1a0c 100%);
  border:1px solid {BORDER}; border-radius:20px; padding:2.5rem 3rem;
  text-align:center; margin-bottom:2rem;
  box-shadow: 0 0 60px rgba(56,189,248,0.07);
}}
.hero-title {{ font-size:2.6rem; font-weight:800; color:#f8fafc; margin:0; letter-spacing:-0.02em; }}
.hero-title span {{ color:{ACCENT}; }}
.hero-sub {{ color:{MUTED}; font-size:0.95rem; margin-top:.5rem; }}
.kpi-card {{
  background:{CARD}; border:1px solid {BORDER}; border-radius:14px;
  padding:1.2rem 1.5rem; text-align:center; transition:border-color .2s;
}}
.kpi-card:hover {{ border-color:{ACCENT}; }}
.kpi-label {{ font-size:.72rem; font-weight:600; text-transform:uppercase;
              letter-spacing:.08em; color:#64748b; margin-bottom:.3rem; }}
.kpi-value {{ font-size:2rem; font-weight:800; color:#f8fafc; }}
.kpi-sub   {{ font-size:.78rem; color:{MUTED}; margin-top:.2rem; }}
.score-track {{ background:#0f172a; border-radius:999px; height:12px; margin:.5rem 0; }}
.score-fill  {{ height:100%; border-radius:999px; }}
.tag {{
  display:inline-block; padding:3px 12px; border-radius:999px;
  font-size:.75rem; font-weight:700; margin:3px 2px;
}}
.section-h {{ font-size:1.05rem; font-weight:700; color:{TEXT};
              border-left:3px solid {ACCENT}; padding-left:.7rem; margin:1.4rem 0 .8rem; }}
.rule-box {{
  background:#0f172a; border:1px solid {BORDER}; border-radius:8px;
  padding:6px 12px; font-size:.78rem; color:{MUTED}; margin:3px 0; font-family:monospace;
}}
.gradio-hint {{
  background:linear-gradient(135deg,#1e293b,#0f2d1f);
  border:1px solid #166534; border-radius:12px; padding:1.2rem 1.5rem; margin-top:.8rem;
}}
.gradio-hint h4 {{ color:#4ade80; margin:0 0 .5rem; }}
.gradio-hint code {{ background:#0f172a; padding:2px 8px; border-radius:5px;
                     color:#86efac; font-size:.85rem; }}
.stButton>button {{
  background:linear-gradient(135deg,#3b82f6,#6366f1) !important;
  color:white !important; border:none !important; border-radius:10px !important;
  padding:.6rem 1.8rem !important; font-weight:700 !important;
  font-size:.95rem !important; width:100% !important;
}}
.stButton>button:hover {{ opacity:.87 !important; transform:translateY(-1px) !important; }}
.stTextArea textarea {{
  background:#0f172a !important; border:1px solid {BORDER} !important;
  border-radius:10px !important; color:#f1f5f9 !important;
}}
.stTextArea textarea:focus {{ border-color:{ACCENT} !important; }}
.stTabs [data-baseweb="tab-list"] {{ background:{CARD}; border-radius:8px; padding:3px; }}
.stTabs [data-baseweb="tab"] {{ background:transparent; color:{MUTED}; border-radius:6px; }}
.stTabs [aria-selected="true"] {{ background:{DARK}; color:{ACCENT}; }}
#MainMenu,footer,header {{ visibility:hidden; }}
</style>""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────
def sc(s):
    if s>=80: return "#22c55e"
    if s>=60: return "#86efac"
    if s>=40: return "#facc15"
    if s>=20: return "#f97316"
    return "#ef4444"

def lb_bg(l):
    return {"Very Positive":"#14532d","Positive":"#166534","Neutral":"#713f12",
            "Negative":"#7c2d12","Very Negative":"#450a0a"}.get(l,"#334155")

CMAP = {"Very Positive":"#22c55e","Positive":"#86efac","Neutral":"#facc15",
        "Negative":"#f97316","Very Negative":"#ef4444"}

# ── Cache data & FIS ──────────────────────────────────────────
@st.cache_data
def load_data():
    csv = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "reviews.csv")
    df = pd.read_csv(csv, low_memory=False)
    df = df[['name','reviews.text','reviews.rating','reviews.doRecommend']].copy()
    df.columns = ['product','review_text','rating','recommend']
    df.dropna(subset=['review_text','rating'], inplace=True)
    df['rating']      = pd.to_numeric(df['rating'], errors='coerce')
    df.dropna(subset=['rating'], inplace=True)
    df['review_text'] = df['review_text'].astype(str).str.strip()
    df = df[df['review_text'].str.len() > 10].reset_index(drop=True)
    return df

@st.cache_resource
def get_fis_cached():
    return build_fis()

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 FuzzyMind")
    st.markdown("**Soft Computing | Fuzzy Logic FIS**")
    st.divider()

    try:
        df_all = load_data()
        data_ok = True
    except:
        data_ok = False
        df_all  = None

    if data_ok:
        st.markdown(f"""
        <div class='kpi-card' style='margin-bottom:.6rem;'>
          <div class='kpi-label'>Dataset Loaded</div>
          <div class='kpi-value' style='font-size:1.4rem;color:#22c55e;'>✅ Ready</div>
          <div class='kpi-sub'>{len(df_all):,} Amazon reviews</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.error("⚠️ Dataset not found at data/reviews.csv")

    st.markdown("""
    <div class='kpi-card' style='margin-bottom:.6rem;'>
      <div class='kpi-label'>FIS Type</div>
      <div style='color:#38bdf8;font-weight:700;font-size:1rem;'>Mamdani</div>
    </div>
    <div class='kpi-card' style='margin-bottom:.6rem;'>
      <div class='kpi-label'>Defuzzification</div>
      <div style='color:#38bdf8;font-weight:700;font-size:1rem;'>Centroid (COG)</div>
    </div>
    <div class='kpi-card' style='margin-bottom:.6rem;'>
      <div class='kpi-label'>Membership Functions</div>
      <div style='color:#38bdf8;font-weight:700;font-size:1rem;'>Triangular (trimf)</div>
    </div>
    <div class='kpi-card' style='margin-bottom:.6rem;'>
      <div class='kpi-label'>Total Rules</div>
      <div style='color:#38bdf8;font-weight:700;font-size:1rem;'>15 IF-THEN Rules</div>
    </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("### 🟢 Gradio UI")
    st.markdown("""
    <div class='gradio-hint'>
      <h4>Launch Gradio separately:</h4>
      <code>python app/gradio_app.py</code><br><br>
      Opens at <code>http://localhost:7860</code>
    </div>""", unsafe_allow_html=True)

# ── HERO ──────────────────────────────────────────────────────
st.markdown("""
<div class='hero-wrap'>
  <div class='hero-title'>🧠 <span>FuzzyMind</span> Sentiment Analyzer</div>
  <div class='hero-sub'>Soft Computing · Mamdani FIS · Fuzzy Sets · Amazon Reviews Dataset · 34,660 Reviews</div>
</div>""", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────
t1,t2,t3,t4,t5 = st.tabs([
    "🔍 Analyze Text",
    "📊 Dataset Explorer",
    "📈 Membership Functions",
    "🏆 Product Insights",
    "📚 FIS Knowledge Base"
])

# ════════════════ TAB 1: ANALYZE ═════════════════════════════
with t1:
    cl, cr = st.columns([1, 1], gap="large")

    with cl:
        st.markdown("<div class='section-h'>Enter Review Text</div>", unsafe_allow_html=True)
        user_text = st.text_area("", height=160,
            value="This Amazon Fire tablet is absolutely fantastic! Super easy to use, great screen quality and amazing battery life. My kids love it so much!",
            label_visibility="collapsed")

        if st.button("⚡ Run Fuzzy Inference", use_container_width=True):
            st.session_state["run"] = True

        st.markdown("<div class='kpi-label' style='margin-top:1rem;'>Quick Samples from Dataset</div>", unsafe_allow_html=True)
        samples = {
            "😄 5★ Very Positive": "This is the best tablet I have ever purchased! Absolutely love everything about it!",
            "🙂 4★ Positive":      "Great value for money, works well and easy to set up. Very happy with purchase.",
            "😐 3★ Neutral":       "It is a decent tablet for the price. Nothing special but gets the job done.",
            "😟 2★ Negative":      "Disappointing product. Slow performance and annoying ads everywhere.",
            "😠 1★ Very Negative": "Terrible! Broken on arrival, horrible quality, waste of money. Never buying again!",
        }
        c1,c2 = st.columns(2)
        for i,(lbl,txt) in enumerate(samples.items()):
            if (c1 if i%2==0 else c2).button(lbl, key=f"s{i}"):
                st.session_state["sample"] = txt
                st.rerun()
        if "sample" in st.session_state:
            user_text = st.session_state.pop("sample")
            st.session_state["run"] = True

    with cr:
        st.markdown("<div class='section-h'>Fuzzy Analysis Result</div>", unsafe_allow_html=True)
        if user_text.strip():
            result = analyze_sentiment(user_text)
            sc_val = result["score"]; clr = sc(sc_val); lbl = result["label"]

            st.markdown(f"""
            <div class='kpi-card' style='border-color:{clr};border-width:2px;margin-bottom:1rem;'>
              <div class='kpi-label'>Fuzzy Sentiment Score</div>
              <div class='kpi-value' style='color:{clr};font-size:2.8rem;'>{sc_val:.1f}
                <span style='font-size:1rem;color:#64748b;font-weight:400;'>/ 100</span></div>
              <div class='score-track'>
                <div class='score-fill' style='width:{sc_val}%;background:{clr};'></div>
              </div>
              <div style='font-size:1.6rem;margin-top:.5rem;'>{result["emoji"]} &nbsp;
                <span class='tag' style='background:{lb_bg(lbl)};color:{clr};font-size:.9rem;'>{lbl}</span>
              </div>
            </div>""", unsafe_allow_html=True)

            f = result["features"]
            c_p, c_n, c_u = st.columns(3)
            for col,(nm,val,color) in zip([c_p,c_n,c_u],[
                ("Positive",f["pos_score"],"#4ade80"),
                ("Negative",f["neg_score"],"#f87171"),
                ("Punct.",  f["punct_score"],"#facc15"),
            ]):
                col.markdown(f"""
                <div class='kpi-card' style='text-align:center;padding:.9rem;'>
                  <div class='kpi-label'>{nm}</div>
                  <div style='font-size:1.6rem;font-weight:700;color:{color};'>{val:.2f}</div>
                  <div style='color:#64748b;font-size:.7rem;'>/ 10</div>
                  <div class='score-track' style='height:6px;'>
                    <div class='score-fill' style='width:{val/10*100:.1f}%;background:{color};'></div>
                  </div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<div class='kpi-label' style='margin-top:.8rem;'>Output Membership Degrees</div>", unsafe_allow_html=True)
            for m_lbl, m_deg in result["memberships"].items():
                mc = CMAP[m_lbl]
                st.markdown(f"""
                <div style='margin:4px 0;'>
                  <div style='display:flex;justify-content:space-between;font-size:.78rem;
                              color:#94a3b8;margin-bottom:3px;'>
                    <span>{m_lbl}</span>
                    <span style='color:{mc};font-weight:700;'>{m_deg:.3f}</span>
                  </div>
                  <div class='score-track' style='height:8px;'>
                    <div class='score-fill' style='width:{m_deg*100:.1f}%;background:{mc};'></div>
                  </div>
                </div>""", unsafe_allow_html=True)

# ════════════════ TAB 2: DATASET EXPLORER ════════════════════
with t2:
    if not data_ok:
        st.error("Dataset not loaded. Place reviews.csv in the data/ folder.")
    else:
        st.markdown("<div class='section-h'>Dataset Overview</div>", unsafe_allow_html=True)

        c1,c2,c3,c4 = st.columns(4)
        for col,(lbl,val,sub) in zip([c1,c2,c3,c4],[
            ("Total Reviews",f"{len(df_all):,}","Amazon product reviews"),
            ("Unique Products",f"{df_all['product'].nunique()}","distinct items"),
            ("Avg Rating",f"{df_all['rating'].mean():.2f} ⭐","out of 5 stars"),
            ("5★ Reviews",f"{(df_all['rating']==5).sum():,}",f"{(df_all['rating']==5).mean()*100:.1f}% of total"),
        ]):
            col.markdown(f"""<div class='kpi-card'><div class='kpi-label'>{lbl}</div>
            <div class='kpi-value' style='font-size:1.7rem;'>{val}</div>
            <div class='kpi-sub'>{sub}</div></div>""", unsafe_allow_html=True)

        st.markdown("<div class='section-h'>Rating Distribution</div>", unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 2, figsize=(14, 4), facecolor="#0f172a")
        colors = ['#ef4444','#f97316','#facc15','#86efac','#22c55e']
        counts = df_all['rating'].value_counts().sort_index()
        axes[0].set_facecolor("#1e293b")
        bars = axes[0].bar(counts.index, counts.values, color=colors, edgecolor="#0f172a", width=0.7)
        for b in bars:
            axes[0].text(b.get_x()+b.get_width()/2, b.get_height()+150,
                        f'{int(b.get_height()):,}', ha='center', fontsize=9, color="#94a3b8")
        axes[0].set_title("Star Rating Distribution", color=ACCENT, fontweight="bold")
        axes[0].set_xlabel("Stars", color="#94a3b8"); axes[0].set_ylabel("Count", color="#94a3b8")
        axes[0].grid(True, axis='y', alpha=0.3, color="#334155")
        axes[0].tick_params(colors="#94a3b8")
        for sp in axes[0].spines.values(): sp.set_edgecolor("#334155")

        axes[1].set_facecolor("#1e293b")
        df_all['text_len'] = df_all['review_text'].str.len()
        axes[1].hist(df_all['text_len'].clip(0,800), bins=30, color="#38bdf8", edgecolor="#0f172a", alpha=0.8)
        axes[1].axvline(df_all['text_len'].median(), color="#facc15", linestyle='--', lw=1.5,
                       label=f"Median: {df_all['text_len'].median():.0f} chars")
        axes[1].set_title("Review Length Distribution", color=ACCENT, fontweight="bold")
        axes[1].set_xlabel("Characters", color="#94a3b8"); axes[1].set_ylabel("Count", color="#94a3b8")
        axes[1].legend(); axes[1].grid(True, alpha=0.3, color="#334155")
        axes[1].tick_params(colors="#94a3b8")
        for sp in axes[1].spines.values(): sp.set_edgecolor("#334155")

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

        # Sample reviews table
        st.markdown("<div class='section-h'>Sample Reviews</div>", unsafe_allow_html=True)
        rating_filter = st.selectbox("Filter by Rating", [1,2,3,4,5,None],
                                     format_func=lambda x: f"{x}★" if x else "All")
        sample_df = df_all if not rating_filter else df_all[df_all['rating']==rating_filter]
        st.dataframe(sample_df[['product','review_text','rating']].sample(
            min(15, len(sample_df))).reset_index(drop=True),
            use_container_width=True, hide_index=True)

# ════════════════ TAB 3: MEMBERSHIP FUNCTIONS ═══════════════
with t3:
    fis_s, mf = get_fis_cached()
    universe   = mf["universe"]; su = mf["sentiment_universe"]

    st.markdown("<div class='section-h'>Fuzzy Membership Functions (Triangular)</div>", unsafe_allow_html=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), facecolor="#0f172a")
    C = {"low":"#f87171","medium":"#facc15","high":"#4ade80",
         "very_negative":"#ef4444","negative":"#f97316",
         "neutral":"#facc15","positive":"#86efac","very_positive":"#22c55e"}

    def sa(ax, t, xl):
        ax.set_facecolor("#1e293b"); ax.set_title(t,color=ACCENT,fontweight="bold")
        ax.set_xlabel(xl,color="#94a3b8"); ax.set_ylabel("μ(x)",color="#94a3b8")
        ax.set_ylim(-0.05,1.15); ax.grid(True,color="#334155",alpha=0.4)
        ax.tick_params(colors="#94a3b8")
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

    plt.tight_layout(); st.pyplot(fig, use_container_width=True)

    # 3D Surface
    st.markdown("<div class='section-h'>3D FIS Control Surface</div>", unsafe_allow_html=True)
    N=15; xr=np.linspace(0.1,9.9,N); yr=np.linspace(0.1,9.9,N); Z=np.zeros((N,N))
    for i,xv in enumerate(xr):
        for j,yv in enumerate(yr):
            try:
                fis_s.input["positive_score"]=float(xv); fis_s.input["negative_score"]=float(yv)
                fis_s.input["punctuation_score"]=3.0; fis_s.compute(); Z[i,j]=fis_s.output["sentiment"]
            except: Z[i,j]=50.0
    X,Y=np.meshgrid(xr,yr)
    from mpl_toolkits.mplot3d import Axes3D
    fig3=plt.figure(figsize=(11,6),facecolor="#0f172a")
    ax3=fig3.add_subplot(111,projection="3d"); ax3.set_facecolor("#1e293b")
    surf=ax3.plot_surface(X,Y,Z.T,cmap="RdYlGn",alpha=0.9,edgecolor="none")
    ax3.set_xlabel("Positive Score",color="#94a3b8",labelpad=8)
    ax3.set_ylabel("Negative Score",color="#94a3b8",labelpad=8)
    ax3.set_zlabel("Sentiment",color="#94a3b8",labelpad=8)
    ax3.set_title("Mamdani FIS — Control Surface",color=ACCENT,fontweight="bold")
    ax3.tick_params(colors="#94a3b8"); plt.colorbar(surf,ax=ax3,shrink=0.5,label="Sentiment Score")
    plt.tight_layout(); st.pyplot(fig3, use_container_width=True)

# ════════════════ TAB 4: PRODUCT INSIGHTS ════════════════════
with t4:
    if not data_ok:
        st.error("Dataset needed."); st.stop()

    st.markdown("<div class='section-h'>Run Fuzzy Analysis on Dataset Sample</div>", unsafe_allow_html=True)
    n_samples = st.slider("Number of reviews to analyze", 200, 2000, 500, 100)

    if st.button("🚀 Run Fuzzy FIS on Dataset", use_container_width=True):
        sample = df_all.sample(n_samples, random_state=42).copy()
        prog   = st.progress(0, "Running fuzzy inference...")
        rows   = []
        for i, row in enumerate(sample.itertuples()):
            r = analyze_sentiment(row.review_text)
            rows.append({"product":row.product,"rating":row.rating,
                         "fuzzy_score":r["score"],"fuzzy_label":r["label"]})
            if i % 100 == 0: prog.progress(i/n_samples, f"Analyzed {i}/{n_samples}...")
        prog.progress(1.0, "✅ Done!")
        st.session_state["insights"] = pd.DataFrame(rows)

    if "insights" in st.session_state:
        di = st.session_state["insights"]

        # Accuracy
        di["true_class"]  = di["rating"].apply(lambda x: "Positive" if x>=4 else "Neutral" if x==3 else "Negative")
        di["fuzzy_class"] = di["fuzzy_score"].apply(score_to_3class)
        acc = (di["true_class"]==di["fuzzy_class"]).mean()*100

        c1,c2,c3,c4 = st.columns(4)
        for col,(lbl,val) in zip([c1,c2,c3,c4],[
            ("Analyzed",f"{len(di):,}"),("Accuracy",f"{acc:.1f}%"),
            ("Avg Score",f"{di['fuzzy_score'].mean():.1f}"),
            ("Most Common",di["fuzzy_label"].mode()[0]),
        ]):
            col.markdown(f"""<div class='kpi-card'><div class='kpi-label'>{lbl}</div>
            <div class='kpi-value' style='font-size:1.7rem;'>{val}</div></div>""", unsafe_allow_html=True)

        # Charts
        fig,axes=plt.subplots(1,2,figsize=(14,5),facecolor="#0f172a")
        axes[0].set_facecolor("#1e293b")
        counts = di["fuzzy_label"].value_counts()
        order  = [o for o in ["Very Positive","Positive","Neutral","Negative","Very Negative"] if o in counts.index]
        counts = counts.reindex(order)
        axes[0].pie(counts.values, labels=counts.index, autopct="%1.1f%%",
                    colors=[CMAP[k] for k in counts.index],
                    textprops={"color":"#e2e8f0"}, wedgeprops={"edgecolor":"#0f172a","linewidth":2})
        axes[0].set_title("Fuzzy Label Distribution", color=ACCENT, fontweight="bold")

        axes[1].set_facecolor("#1e293b")
        for rt, grp in di.groupby("rating"):
            if len(grp)>5:
                axes[1].boxplot(grp["fuzzy_score"], positions=[rt], widths=0.6, patch_artist=True,
                    boxprops=dict(facecolor="#38bdf8",alpha=0.45),
                    medianprops=dict(color="#facc15",linewidth=2.5),
                    whiskerprops=dict(color="#94a3b8"), capprops=dict(color="#94a3b8"),
                    flierprops=dict(marker="o",color="#64748b",markersize=2,alpha=0.4))
        axes[1].set_title("Fuzzy Score vs Star Rating", color=ACCENT, fontweight="bold")
        axes[1].set_xlabel("Star Rating",color="#94a3b8"); axes[1].set_ylabel("Fuzzy Score",color="#94a3b8")
        axes[1].set_xticks([1,2,3,4,5]); axes[1].grid(True,alpha=0.3,axis="y",color="#334155")
        axes[1].tick_params(colors="#94a3b8")
        for sp in axes[1].spines.values(): sp.set_edgecolor("#334155")

        plt.tight_layout(); st.pyplot(fig, use_container_width=True)

        # Top products
        st.markdown("<div class='section-h'>Top Products by Fuzzy Sentiment</div>", unsafe_allow_html=True)
        prod = di.groupby("product").agg(avg_score=("fuzzy_score","mean"),count=("fuzzy_score","size")).reset_index()
        prod = prod[prod["count"]>=3].sort_values("avg_score",ascending=False)
        prod["name_short"] = prod["product"].str[:50]+"..."
        top = prod.head(8)
        fig2,ax2=plt.subplots(figsize=(13,5),facecolor="#0f172a")
        ax2.set_facecolor("#1e293b")
        bcs=[sc(s) for s in top["avg_score"]]
        bars=ax2.barh(range(len(top)),top["avg_score"],color=bcs,edgecolor="#334155",height=0.65)
        for b,s in zip(bars,top["avg_score"]):
            ax2.text(s+0.3,b.get_y()+b.get_height()/2,f"{s:.1f}",va="center",color="#e2e8f0",fontsize=9,fontweight="bold")
        ax2.set_yticks(range(len(top))); ax2.set_yticklabels(top["name_short"],fontsize=8)
        ax2.set_xlabel("Avg Fuzzy Score",color="#94a3b8")
        ax2.set_title("Products Ranked by Fuzzy Sentiment",color=ACCENT,fontweight="bold",fontsize=13)
        ax2.set_xlim(0,110); ax2.grid(True,axis="x",alpha=0.3,color="#334155")
        ax2.tick_params(colors="#94a3b8")
        for sp in ax2.spines.values(): sp.set_edgecolor("#334155")
        plt.tight_layout(); st.pyplot(fig2, use_container_width=True)

# ════════════════ TAB 5: KNOWLEDGE BASE ══════════════════════
with t5:
    st.markdown("<div class='section-h'>Mamdani FIS — Full Specification</div>", unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1:
        st.markdown(f"""
        <div class='kpi-card'>
          <h4 style='color:{ACCENT};margin-top:0;'>📐 Fuzzy Variables</h4>
          <p style='color:{MUTED};font-size:.88rem;'>
          <b style='color:{TEXT};'>Input Variables (Antecedents):</b><br>
          • <code>positive_score</code> → {{low, medium, high}}<br>
          • <code>negative_score</code> → {{low, medium, high}}<br>
          • <code>punctuation_score</code> → {{low, high}}<br><br>
          <b style='color:{TEXT};'>Output Variable (Consequent):</b><br>
          • <code>sentiment</code> → {{very_negative, negative, neutral, positive, very_positive}}
          </p>
        </div>
        <div class='kpi-card' style='margin-top:.8rem;'>
          <h4 style='color:{ACCENT};margin-top:0;'>⚙️ Triangular MF Formula</h4>
          <p style='color:{MUTED};font-size:.88rem;'>
          <code style='color:#4ade80;font-size:.9rem;'>trimf(x, [a, b, c])</code><br><br>
          • <b>a</b> = left base (μ = 0)<br>
          • <b>b</b> = peak (μ = 1)<br>
          • <b>c</b> = right base (μ = 0)<br><br>
          All input MFs span universe [0, 10]<br>
          Output MFs span universe [0, 100]
          </p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='kpi-card'>
          <h4 style='color:{ACCENT};margin-top:0;'>📋 15 Fuzzy Rules</h4>""", unsafe_allow_html=True)
        rules_disp = [
            ("R1",  "pos=HIGH & neg=LOW",              "VERY POSITIVE", "#22c55e"),
            ("R2",  "pos=HIGH & neg=LOW & punct=HIGH", "VERY POSITIVE", "#22c55e"),
            ("R3",  "pos=HIGH & neg=MED",              "POSITIVE",      "#86efac"),
            ("R4",  "pos=MED & neg=LOW",               "POSITIVE",      "#86efac"),
            ("R5",  "pos=MED & neg=LOW & punct=HIGH",  "POSITIVE",      "#86efac"),
            ("R6",  "pos=MED & neg=MED",               "NEUTRAL",       "#facc15"),
            ("R7",  "pos=LOW & neg=LOW",               "NEUTRAL",       "#facc15"),
            ("R8",  "pos=LOW & neg=LOW & punct=LOW",   "NEUTRAL",       "#facc15"),
            ("R9",  "neg=HIGH & pos=LOW",              "VERY NEGATIVE", "#ef4444"),
            ("R10", "neg=HIGH & pos=MED",              "NEGATIVE",      "#f97316"),
            ("R11", "neg=MED & pos=LOW",               "NEGATIVE",      "#f97316"),
            ("R12", "neg=HIGH & punct=HIGH",           "VERY NEGATIVE", "#ef4444"),
            ("R13", "pos=HIGH & neg=HIGH",             "NEUTRAL",       "#facc15"),
            ("R14", "pos=LOW & neg=MED",               "NEGATIVE",      "#f97316"),
            ("R15", "pos=MED & neg=HIGH",              "NEGATIVE",      "#f97316"),
        ]
        for rid,cond,cons,clr in rules_disp:
            st.markdown(f"""<div class='rule-box'>
              <span style='color:#38bdf8;'>{rid}</span>: IF {cond} →
              <span style='color:{clr};font-weight:700;'>{cons}</span></div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class='kpi-card' style='margin-top:1rem;'>
      <h4 style='color:{ACCENT};margin-top:0;'>🔄 FIS Pipeline</h4>
      <div style='display:flex;gap:.7rem;flex-wrap:wrap;align-items:center;font-size:.85rem;color:{MUTED};'>
        <span style='background:#0f172a;padding:7px 14px;border-radius:8px;border:1px solid {BORDER};'>📝 Raw Review</span>
        <span style='color:{ACCENT};font-size:1.2rem;'>→</span>
        <span style='background:#0f172a;padding:7px 14px;border-radius:8px;border:1px solid {BORDER};'>🔤 Feature Extraction</span>
        <span style='color:{ACCENT};font-size:1.2rem;'>→</span>
        <span style='background:#0f172a;padding:7px 14px;border-radius:8px;border:1px solid {BORDER};'>📐 Fuzzification</span>
        <span style='color:{ACCENT};font-size:1.2rem;'>→</span>
        <span style='background:#0f172a;padding:7px 14px;border-radius:8px;border:1px solid {BORDER};'>📋 Rule Evaluation</span>
        <span style='color:{ACCENT};font-size:1.2rem;'>→</span>
        <span style='background:#0f172a;padding:7px 14px;border-radius:8px;border:1px solid {BORDER};'>🔗 Aggregation</span>
        <span style='color:{ACCENT};font-size:1.2rem;'>→</span>
        <span style='background:#0f172a;padding:7px 14px;border-radius:8px;border:1px solid {BORDER};'>📍 Defuzzification</span>
        <span style='color:{ACCENT};font-size:1.2rem;'>→</span>
        <span style='background:#166534;padding:7px 14px;border-radius:8px;border:1px solid #22c55e;color:#22c55e;font-weight:700;'>✅ Sentiment</span>
      </div>
    </div>""", unsafe_allow_html=True)

st.markdown(f"""<hr style='border-color:{BORDER};margin-top:3rem;'>
<div style='text-align:center;color:#475569;font-size:.78rem;padding:1rem;'>
  🧠 FuzzyMind — Soft Computing Project &nbsp;|&nbsp;
  Mamdani FIS · Triangular MFs · Centroid Defuzzification · Amazon Reviews Dataset
</div>""", unsafe_allow_html=True)
