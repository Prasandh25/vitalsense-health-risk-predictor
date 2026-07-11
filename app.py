import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VitalSense Ultra 2.0",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800;900&family=Rajdhani:wght@300;400;500;600;700&family=Share+Tech+Mono&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #050810 !important;
    color: #c8e0f0 !important;
    font-family: 'Rajdhani', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 90% 60% at 15% 5%, rgba(0,255,200,0.05) 0%, transparent 55%),
        radial-gradient(ellipse 70% 45% at 85% 95%, rgba(100,0,255,0.05) 0%, transparent 55%),
        radial-gradient(ellipse 50% 35% at 50% 50%, rgba(0,150,255,0.03) 0%, transparent 50%),
        #050810 !important;
}

[data-testid="stSidebar"] {
    background: rgba(8,11,20,0.96) !important;
    border-right: 1px solid rgba(0,255,180,0.15) !important;
    backdrop-filter: blur(24px) !important;
}

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-track { background: #050810; }
::-webkit-scrollbar-thumb { background: rgba(0,255,180,0.3); border-radius: 2px; }

h1, h2, h3 { font-family: 'Orbitron', monospace !important; letter-spacing: 0.08em; }

/* ── Neon Cards ── */
.glass-card {
    background: linear-gradient(135deg, rgba(14,20,38,0.9) 0%, rgba(10,14,26,0.95) 100%);
    border: 1px solid rgba(0,255,180,0.18);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(16px);
    box-shadow:
        0 4px 30px rgba(0,0,0,0.4),
        0 0 0 1px rgba(0,255,180,0.05),
        inset 0 1px 0 rgba(255,255,255,0.04);
    position: relative;
    overflow: hidden;
}
.glass-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,255,180,0.5), transparent);
}

.glass-card-red {
    border-color: rgba(255,50,100,0.28);
    box-shadow: 0 4px 30px rgba(0,0,0,0.4), 0 0 25px rgba(255,50,100,0.06);
}
.glass-card-red::before { background: linear-gradient(90deg, transparent, rgba(255,50,100,0.5), transparent); }

.glass-card-amber {
    border-color: rgba(255,180,0,0.28);
    box-shadow: 0 4px 30px rgba(0,0,0,0.4), 0 0 25px rgba(255,180,0,0.06);
}
.glass-card-amber::before { background: linear-gradient(90deg, transparent, rgba(255,180,0,0.5), transparent); }

.glass-card-teal {
    border-color: rgba(0,220,255,0.28);
    box-shadow: 0 4px 30px rgba(0,0,0,0.4), 0 0 25px rgba(0,220,255,0.06);
}
.glass-card-teal::before { background: linear-gradient(90deg, transparent, rgba(0,220,255,0.5), transparent); }

/* ── KPI ── */
.kpi-block { text-align: center; padding: 0.8rem; }
.kpi-value {
    font-family: 'Orbitron', monospace;
    font-size: 2.4rem;
    font-weight: 900;
    line-height: 1;
    color: #00ffb4;
    text-shadow: 0 0 18px rgba(0,255,180,0.55), 0 0 40px rgba(0,255,180,0.25);
    display: block;
}
.kpi-value.red {
    color: #ff3264;
    text-shadow: 0 0 18px rgba(255,50,100,0.55), 0 0 40px rgba(255,50,100,0.25);
}
.kpi-value.amber {
    color: #ffb400;
    text-shadow: 0 0 18px rgba(255,180,0,0.55), 0 0 40px rgba(255,180,0,0.25);
}
.kpi-value.teal {
    color: #00e5ff;
    text-shadow: 0 0 18px rgba(0,229,255,0.55), 0 0 40px rgba(0,229,255,0.25);
}
.kpi-label {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: rgba(200,224,240,0.45);
    margin-top: 0.3rem;
}
.kpi-sublabel {
    font-size: 0.68rem;
    color: rgba(200,224,240,0.3);
    margin-top: 0.2rem;
    letter-spacing: 0.05em;
}

/* ── Section Title ── */
.section-title {
    font-family: 'Orbitron', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    color: rgba(0,255,180,0.65);
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(0,255,180,0.25), transparent);
}

/* ── Risk Badge ── */
.risk-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-family: 'Orbitron', monospace;
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
.risk-critical { background: rgba(255,50,100,0.12); border: 1px solid rgba(255,50,100,0.45); color: #ff3264; }
.risk-high     { background: rgba(255,120,0,0.12);  border: 1px solid rgba(255,120,0,0.45);  color: #ff7800; }
.risk-moderate { background: rgba(255,180,0,0.12);  border: 1px solid rgba(255,180,0,0.45);  color: #ffb400; }
.risk-low      { background: rgba(0,255,180,0.08);  border: 1px solid rgba(0,255,180,0.35);  color: #00ffb4; }
.risk-minimal  { background: rgba(0,200,255,0.08);  border: 1px solid rgba(0,200,255,0.35);  color: #00c8ff; }

/* ── Sidebar ── */
.nav-logo {
    font-family: 'Orbitron', monospace;
    font-size: 1.05rem;
    font-weight: 900;
    color: #00ffb4;
    text-shadow: 0 0 14px rgba(0,255,180,0.5);
    letter-spacing: 0.06em;
    text-align: center;
    margin-bottom: 0.15rem;
}
.nav-sub {
    font-size: 0.58rem;
    letter-spacing: 0.32em;
    text-transform: uppercase;
    color: rgba(0,255,180,0.35);
    text-align: center;
    margin-bottom: 1.2rem;
}
.nav-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,255,180,0.2), transparent);
    margin: 0.8rem 0;
}

/* ── Inputs ── */
[data-testid="stSelectbox"] > div > div {
    background: rgba(14,20,38,0.9) !important;
    border: 1px solid rgba(0,255,180,0.2) !important;
    color: #c8e0f0 !important;
    border-radius: 8px !important;
}
[data-testid="stSlider"] > div > div > div > div {
    background: rgba(0,255,180,0.4) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: rgba(0,255,180,0.07) !important;
    border: 1px solid rgba(0,255,180,0.3) !important;
    color: #00ffb4 !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: rgba(0,255,180,0.15) !important;
    box-shadow: 0 0 20px rgba(0,255,180,0.15) !important;
    transform: translateY(-1px) !important;
}

/* ── Chat Bubbles ── */
.chat-bubble-user {
    background: linear-gradient(135deg, rgba(0,255,180,0.08), rgba(0,200,255,0.06));
    border: 1px solid rgba(0,255,180,0.18);
    border-radius: 14px 14px 4px 14px;
    padding: 0.85rem 1.1rem;
    margin: 0.6rem 0 0.6rem 2.5rem;
    font-size: 0.9rem;
    color: #c8e0f0;
    position: relative;
}
.chat-bubble-bot {
    background: linear-gradient(135deg, rgba(14,20,38,0.85), rgba(10,14,26,0.9));
    border: 1px solid rgba(0,255,180,0.12);
    border-radius: 14px 14px 14px 4px;
    padding: 0.85rem 1.1rem;
    margin: 0.6rem 2.5rem 0.6rem 0;
    font-size: 0.88rem;
    color: #a8cce0;
    line-height: 1.65;
    position: relative;
}
.chat-bubble-bot::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, rgba(0,255,180,0.3), transparent);
    border-radius: 14px;
}
.chat-avatar { font-size: 1.15rem; margin-right: 0.4rem; }
.chat-label {
    font-family: 'Orbitron', monospace;
    font-size: 0.55rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: rgba(0,255,180,0.45);
    margin-bottom: 0.35rem;
}

/* ── Recommendation Cards ── */
.rec-tier {
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    margin: 0.6rem 0;
    position: relative;
    overflow: hidden;
}
.rec-tier-critical {
    background: linear-gradient(135deg, rgba(255,50,100,0.1), rgba(200,0,50,0.05));
    border: 1px solid rgba(255,50,100,0.3);
    border-left: 3px solid #ff3264;
}
.rec-tier-high {
    background: linear-gradient(135deg, rgba(255,120,0,0.1), rgba(200,80,0,0.05));
    border: 1px solid rgba(255,120,0,0.3);
    border-left: 3px solid #ff7800;
}
.rec-tier-moderate {
    background: linear-gradient(135deg, rgba(255,180,0,0.08), rgba(200,140,0,0.04));
    border: 1px solid rgba(255,180,0,0.25);
    border-left: 3px solid #ffb400;
}
.rec-tier-low {
    background: linear-gradient(135deg, rgba(0,255,180,0.06), rgba(0,200,140,0.03));
    border: 1px solid rgba(0,255,180,0.2);
    border-left: 3px solid #00ffb4;
}
.rec-tier-minimal {
    background: linear-gradient(135deg, rgba(0,200,255,0.06), rgba(0,150,200,0.03));
    border: 1px solid rgba(0,200,255,0.2);
    border-left: 3px solid #00c8ff;
}
.rec-tier-title {
    font-family: 'Orbitron', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.rec-item {
    font-size: 0.86rem;
    color: rgba(200,224,240,0.82);
    padding: 0.25rem 0 0.25rem 1.2rem;
    position: relative;
    line-height: 1.55;
    border-bottom: 1px solid rgba(255,255,255,0.03);
}
.rec-item:last-child { border-bottom: none; }
.rec-item::before {
    content: '▸';
    position: absolute;
    left: 0;
    font-size: 0.75rem;
    line-height: 1.7;
    color: rgba(0,255,180,0.6);
}

/* ── Progress Bars ── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #00ffb4, #00d4ff) !important;
}

/* ── Tabs ── */
[data-baseweb="tab-list"] {
    background: rgba(14,20,38,0.5) !important;
    border-bottom: 1px solid rgba(0,255,180,0.15) !important;
    gap: 0 !important;
}
[data-baseweb="tab"] {
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    color: rgba(200,224,240,0.45) !important;
}
[aria-selected="true"] {
    color: #00ffb4 !important;
    border-bottom: 2px solid #00ffb4 !important;
}

/* ── Animations ── */
@keyframes pulse-red {
    0%, 100% { box-shadow: 0 4px 30px rgba(0,0,0,0.4), 0 0 20px rgba(255,50,100,0.06); }
    50%       { box-shadow: 0 4px 30px rgba(0,0,0,0.4), 0 0 40px rgba(255,50,100,0.18); }
}
.pulse-red { animation: pulse-red 2.5s ease-in-out infinite; }

@keyframes glow-green {
    0%, 100% { box-shadow: 0 4px 30px rgba(0,0,0,0.4), 0 0 15px rgba(0,255,180,0.04); }
    50%       { box-shadow: 0 4px 30px rgba(0,0,0,0.4), 0 0 30px rgba(0,255,180,0.12); }
}
.glow-green { animation: glow-green 3s ease-in-out infinite; }

/* ── Wellness Score Ring ── */
.wellness-ring-wrap { text-align: center; padding: 0.5rem; }
.wellness-label { font-family: 'Share Tech Mono', monospace; font-size: 0.72rem; color: rgba(200,224,240,0.5); letter-spacing: 0.12em; }

/* ── Health factor bar ── */
.factor-bar-wrap { margin: 0.35rem 0; }
.factor-label { font-size: 0.78rem; color: rgba(200,224,240,0.65); margin-bottom: 0.18rem; display:flex; justify-content:space-between; }
.factor-track { height: 6px; background: rgba(255,255,255,0.05); border-radius: 3px; overflow: hidden; }
.factor-fill  { height: 100%; border-radius: 3px; transition: width 0.6s ease; }

/* ── Insight tag ── */
.insight-tag {
    display: inline-block;
    background: rgba(0,255,180,0.08);
    border: 1px solid rgba(0,255,180,0.2);
    border-radius: 6px;
    padding: 0.2rem 0.6rem;
    font-size: 0.72rem;
    color: rgba(0,255,180,0.75);
    margin: 0.15rem;
    font-family: 'Share Tech Mono', monospace;
    letter-spacing: 0.06em;
}
.insight-tag.warn {
    background: rgba(255,180,0,0.08);
    border-color: rgba(255,180,0,0.2);
    color: rgba(255,180,0,0.75);
}
.insight-tag.danger {
    background: rgba(255,50,100,0.08);
    border-color: rgba(255,50,100,0.2);
    color: rgba(255,50,100,0.75);
}

/* ── Score Context bar ── */
.score-context {
    background: rgba(0,0,0,0.25);
    border-radius: 8px;
    padding: 0.65rem 0.9rem;
    margin-top: 0.5rem;
    font-size: 0.78rem;
    color: rgba(200,224,240,0.6);
    border-left: 3px solid rgba(0,255,180,0.3);
    line-height: 1.55;
}

/* ── Chat typing indicator ── */
.typing-dot {
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #00ffb4;
    margin: 0 2px;
    animation: blink 1.2s infinite;
}
.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes blink { 0%,80%,100%{opacity:0.2} 40%{opacity:1} }

/* ── Metric comparison pill ── */
.metric-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.18rem 0.6rem;
    border-radius: 12px;
    font-size: 0.7rem;
    font-family: 'Share Tech Mono', monospace;
}
.metric-pill.good { background: rgba(0,255,180,0.1); color: #00ffb4; border: 1px solid rgba(0,255,180,0.25); }
.metric-pill.warn { background: rgba(255,180,0,0.1); color: #ffb400; border: 1px solid rgba(255,180,0,0.25); }
.metric-pill.bad  { background: rgba(255,50,100,0.1); color: #ff3264; border: 1px solid rgba(255,50,100,0.25); }

</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL BUILDING — Calibrated Gradient Boosting + Random Forest Ensemble
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def build_models():
    np.random.seed(2024)
    N = 6000  # Larger cohort for better calibration

    age       = np.random.normal(42, 13, N).clip(18, 80)
    bmi       = np.random.normal(26.5, 5.5, N).clip(15, 48)
    sleep     = np.random.normal(6.4, 1.5, N).clip(3, 10)
    water     = np.random.normal(2.0, 0.9, N).clip(0.3, 4.5)
    activity  = np.random.normal(38, 30, N).clip(0, 150)
    stress    = np.random.randint(1, 11, N).astype(float)
    diet      = np.random.choice(['Poor','Average','Good','Excellent'], N, p=[0.22, 0.38, 0.28, 0.12])
    screen    = np.random.normal(6.5, 2.6, N).clip(0.5, 14)
    smoking   = np.random.choice([0, 1], N, p=[0.74, 0.26]).astype(float)
    alcohol   = np.random.choice([0, 1, 2, 3], N, p=[0.40, 0.34, 0.18, 0.08]).astype(float)
    fam_diabetes    = np.random.choice([0, 1], N, p=[0.68, 0.32]).astype(float)
    fam_hypertension= np.random.choice([0, 1], N, p=[0.63, 0.37]).astype(float)
    fam_heart       = np.random.choice([0, 1], N, p=[0.70, 0.30]).astype(float)

    diet_enc = {'Poor': 3, 'Average': 2, 'Good': 1, 'Excellent': 0}
    diet_num = np.array([diet_enc[d] for d in diet])

    def noisy(prob, noise=0.035):
        p = (prob + np.random.normal(0, noise, N)).clip(0.02, 0.94)
        return (np.random.rand(N) < p).astype(int)

    # ── Diabetes — evidence-weighted features ──
    d_prob = (
        0.05
        + 0.22 * (activity < 30)
        + 0.17 * (diet_num >= 2)
        + 0.13 * (sleep < 5.5)
        + 0.09 * (stress > 7)
        + 0.06 * (water < 1.5)
        + 0.14 * (bmi > 30)
        + 0.09 * (age > 45)
        + 0.12 * fam_diabetes
        + 0.05 * (alcohol >= 2)
        + 0.03 * (screen > 8)
    )
    diabetes = noisy(d_prob)

    # ── Hypertension — stress, sodium-proxy, activity ──
    h_prob = (
        0.05
        + 0.21 * (stress > 6)
        + 0.16 * (activity < 25)
        + 0.14 * (sleep < 5.5)
        + 0.10 * (water < 1.5)
        + 0.09 * (screen > 9)
        + 0.11 * (bmi > 28)
        + 0.08 * (age > 50)
        + 0.11 * fam_hypertension
        + 0.07 * smoking
        + 0.05 * (alcohol >= 2)
        + 0.03 * (diet_num >= 2)
    )
    hypertension = noisy(h_prob)

    # ── Heart Disease — strongest multi-factor disease ──
    hd_prob = (
        0.04
        + 0.23 * (activity < 20)
        + 0.17 * (diet_num >= 2)
        + 0.15 * (stress > 7)
        + 0.10 * (sleep < 5)
        + 0.08 * (screen > 10)
        + 0.11 * (bmi > 30)
        + 0.11 * (age > 50)
        + 0.13 * fam_heart
        + 0.12 * smoking
        + 0.05 * (alcohol >= 3)
        + 0.04 * (water < 1.2)
    )
    heart = noisy(hd_prob)

    df = pd.DataFrame({
        'Sleep_Hours':      sleep,
        'Water_Litres':     water,
        'Activity_Min':     activity,
        'Stress_Level':     stress,
        'Diet_Score':       diet_num,
        'Screen_Hours':     screen,
        'BMI':              bmi,
        'Age':              age,
        'Smoking':          smoking,
        'Alcohol_Units':    alcohol,
        'Fam_Diabetes':     fam_diabetes,
        'Fam_Hypertension': fam_hypertension,
        'Fam_Heart':        fam_heart,
    })

    FEATURES_BASE = ['Sleep_Hours','Water_Litres','Activity_Min','Stress_Level','Diet_Score','Screen_Hours']
    FEATURES_EXT  = FEATURES_BASE + ['BMI','Age','Smoking','Alcohol_Units']
    FEATURES_DIA  = FEATURES_EXT + ['Fam_Diabetes']
    FEATURES_HYP  = FEATURES_EXT + ['Fam_Hypertension']
    FEATURES_HRT  = FEATURES_EXT + ['Fam_Heart']

    def make_ensemble(features, target):
        gbm = GradientBoostingClassifier(n_estimators=220, max_depth=4,
                                          learning_rate=0.06, subsample=0.8,
                                          min_samples_leaf=15, random_state=42)
        rf  = RandomForestClassifier(n_estimators=160, max_depth=7,
                                      min_samples_leaf=10, random_state=42, n_jobs=-1)
        vc  = VotingClassifier(estimators=[('gbm', gbm), ('rf', rf)],
                                voting='soft', weights=[0.55, 0.45])
        cal = CalibratedClassifierCV(vc, cv=3, method='sigmoid')
        cal.fit(df[features], target)
        return cal, features

    models = {}
    importances = {}

    for label, target, feats in [
        ('Diabetes',     diabetes,     FEATURES_DIA),
        ('Hypertension', hypertension, FEATURES_HYP),
        ('Heart Disease',heart,        FEATURES_HRT),
    ]:
        model, feats_used = make_ensemble(feats, target)
        models[label] = (model, feats_used)

        # Extract feature importances from RF component (sklearn 1.4+ API)
        try:
            vc_fitted   = model.calibrated_classifiers_[0].estimator
            named_ests  = vc_fitted.estimators          # list of (name, est) — original order
            fitted_ests = vc_fitted.estimators_         # list of fitted estimators — same order
            rf_index    = next(i for i, (n, _) in enumerate(named_ests) if n == 'rf')
            rf_est      = fitted_ests[rf_index]
            importances[label] = dict(zip(feats_used, rf_est.feature_importances_))
        except Exception:
            importances[label] = {f: 1/len(feats_used) for f in feats_used}

    return models, importances, df


MODELS, IMPORTANCES, SAMPLE_DF = build_models()

BASE_FEATURES = ['Sleep_Hours','Water_Litres','Activity_Min','Stress_Level','Diet_Score','Screen_Hours']

IDEAL = {
    'Sleep_Hours': 8, 'Water_Litres': 3, 'Activity_Min': 60,
    'Stress_Level': 3, 'Diet_Score': 0, 'Screen_Hours': 3,
    'BMI': 22, 'Age': 30, 'Smoking': 0, 'Alcohol_Units': 0
}

FEATURE_LABELS = {
    'Sleep_Hours':       'Sleep (hrs)',
    'Water_Litres':      'Hydration (L)',
    'Activity_Min':      'Activity (min)',
    'Stress_Level':      'Stress (1-10)',
    'Diet_Score':        'Diet Quality',
    'Screen_Hours':      'Screen Time (hrs)',
    'BMI':               'BMI',
    'Age':               'Age',
    'Smoking':           'Smoking',
    'Alcohol_Units':     'Alcohol (units/day)',
    'Fam_Diabetes':      'Family: Diabetes',
    'Fam_Hypertension':  'Family: Hypertension',
    'Fam_Heart':         'Family: Heart Disease',
}


# ── Plotly theme ───────────────────────────────────────────────────────────────
def neon_layout(fig, title="", height=340):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        title=dict(
            text=title,
            font=dict(family="Orbitron", size=10, color="rgba(0,255,180,0.65)"),
            x=0.01, xanchor="left"
        ),
        margin=dict(l=20, r=20, t=40 if title else 20, b=20),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(family="Rajdhani", size=11, color="rgba(200,224,240,0.65)"),
        ),
        font=dict(family="Rajdhani", color="rgba(200,224,240,0.75)"),
    )
    fig.update_xaxes(gridcolor="rgba(0,255,180,0.06)", zerolinecolor="rgba(0,255,180,0.12)")
    fig.update_yaxes(gridcolor="rgba(0,255,180,0.06)", zerolinecolor="rgba(0,255,180,0.12)")
    return fig


# ── Risk Calculation ───────────────────────────────────────────────────────────
def predict_risks(user_input: dict):
    results = {}
    for disease, (model, feats) in MODELS.items():
        row = {f: user_input.get(f, 0) for f in feats}
        X   = pd.DataFrame([row])[feats]
        prob = model.predict_proba(X)[0][1] * 100
        # Gentle clamp — avoids extreme values while preserving meaningful range
        prob = float(np.clip(prob, 3, 85))
        results[disease] = round(prob, 1)
    return results


def risk_level(pct):
    """5-tier risk classification with calibrated thresholds."""
    if pct >= 65:   return "CRITICAL",  "#ff3264", "risk-critical"
    elif pct >= 50: return "HIGH",      "#ff7800", "risk-high"
    elif pct >= 35: return "MODERATE",  "#ffb400", "risk-moderate"
    elif pct >= 18: return "LOW",       "#00ffb4", "risk-low"
    else:           return "MINIMAL",   "#00c8ff", "risk-minimal"


def risk_context_message(pct, disease):
    """Plain-language contextual message that explains risk without panic."""
    lvl = risk_level(pct)[0]
    ctx = {
        "CRITICAL": {
            "Diabetes":     f"Your profile suggests elevated attention is needed. This score reflects multiple lifestyle factors — not a diagnosis. A check-up and dietary changes can significantly lower this.",
            "Hypertension": f"Several factors in your profile are associated with blood pressure risk. A simple BP reading from a pharmacy or clinic can clarify your actual status quickly.",
            "Heart Disease": f"Your current lifestyle patterns carry compounded risk. The good news: most of these factors are reversible with consistent changes."
        },
        "HIGH": {
            "Diabetes":     f"You're in a range where proactive steps make a big difference. Early intervention is highly effective — small changes compound quickly.",
            "Hypertension": f"Your stress and activity levels are the key drivers here. Targeted lifestyle adjustments can meaningfully reduce this score.",
            "Heart Disease": f"This range calls for attention, but it's very manageable. Most people in this range see significant improvement within 3–6 months of lifestyle change."
        },
        "MODERATE": {
            "Diabetes":     f"You have a moderate elevation — no immediate concern, but worth addressing gradually. Standard prevention habits will keep this stable.",
            "Hypertension": f"You're in the watchful range. Consistent sleep, stress management, and activity maintenance will keep this score in check.",
            "Heart Disease": f"Moderate risk means your current habits are mostly sound. A few targeted improvements will move you into the low-risk zone."
        },
        "LOW": {
            "Diabetes":     f"Your risk is well-managed. Continue your current lifestyle habits and stay consistent.",
            "Hypertension": f"Low risk — your blood pressure risk factors are well-controlled. Maintain your current routine.",
            "Heart Disease": f"You're in a good position. Sustaining your current habits keeps you protected long-term."
        },
        "MINIMAL": {
            "Diabetes":     f"Excellent metabolic health indicators. Your lifestyle is actively protecting you.",
            "Hypertension": f"Very low risk. Your body is well-regulated — keep doing what you're doing.",
            "Heart Disease": f"Outstanding cardiac risk profile. You're in the top tier for preventive health."
        },
    }
    return ctx.get(lvl, {}).get(disease, "")


def overall_risk(risks):
    avg = np.mean(list(risks.values()))
    return risk_level(avg)


def wellness_score(user_data, risks):
    """Compute 0-100 wellness score (higher = healthier)."""
    avg_risk = np.mean(list(risks.values()))
    sleep_s  = min(100, (user_data['Sleep_Hours'] / 8) * 100)
    water_s  = min(100, (user_data['Water_Litres'] / 3) * 100)
    act_s    = min(100, (user_data['Activity_Min'] / 60) * 100)
    stress_s = 100 - (user_data['Stress_Level'] / 10) * 100
    diet_s   = 100 - (user_data['Diet_Score'] / 3) * 100
    screen_s = 100 - min(100, (user_data['Screen_Hours'] / 14) * 100)
    base     = np.mean([sleep_s, water_s, act_s, stress_s, diet_s, screen_s])
    # Penalise clinical factors
    penalty  = (user_data['Smoking'] * 8) + (user_data['Alcohol_Units'] * 3) + max(0, (user_data['BMI'] - 25) * 1.5)
    score    = max(5, round(0.5 * base + 0.5 * (100 - avg_risk) - penalty))
    return min(98, score)


def percentile_rank(value, population, higher_better=True):
    """Return percentile rank vs population."""
    if higher_better:
        return round(np.mean(population <= value) * 100)
    else:
        return round(np.mean(population >= value) * 100)


def delta_badge(val, key):
    thresholds = {
        'Sleep_Hours':  (7, 8.5,  True),
        'Water_Litres': (2, 3,    True),
        'Activity_Min': (30, 60,  True),
        'Stress_Level': (5, 7,    False),
        'Diet_Score':   (1, 2,    False),
        'Screen_Hours': (6, 9,    False),
    }
    if key not in thresholds:
        return "teal"
    lo, hi, higher_better = thresholds[key]
    if higher_better:
        return "teal" if val >= hi else ("amber" if val >= lo else "red")
    else:
        return "teal" if val <= lo else ("amber" if val <= hi else "red")


# ════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class='nav-logo'>⚡ VitalSense</div>
    <div class='nav-sub'>Ultra 2.0 · Health Intelligence</div>
    <div class='nav-divider'></div>
    """, unsafe_allow_html=True)

    page = st.radio("", [
        "🧬  Risk Dashboard",
        "💊  Prevention Protocols",
        "🤖  VitalBot AI",
        "📊  Data Explorer",
    ], label_visibility="collapsed")

    st.markdown("<div class='nav-divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>LIFESTYLE METRICS</div>", unsafe_allow_html=True)

    sleep_h  = st.slider("😴 Sleep (hrs/night)", 3.0, 10.0, 6.5, 0.5)
    water_l  = st.slider("💧 Hydration (L/day)", 0.3, 4.5,  1.8, 0.1)
    activity = st.slider("🏃 Activity (min/day)", 0, 150, 35, 5)
    stress   = st.slider("🔥 Stress Level (1-10)", 1, 10, 6, 1)
    diet_opt = st.selectbox("🥗 Diet Quality", ["Excellent","Good","Average","Poor"])
    screen   = st.slider("📱 Screen Time (hrs)", 0.5, 14.0, 7.0, 0.5)

    st.markdown("<div class='nav-divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>CLINICAL FACTORS</div>", unsafe_allow_html=True)

    st.markdown("<div style='font-size:0.72rem;color:rgba(200,224,240,0.5);margin-bottom:0.3rem'>⚖️ Height & Weight → BMI auto-calculated</div>", unsafe_allow_html=True)
    _ht_unit = st.radio("Height unit", ["cm", "ft / in"], horizontal=True, label_visibility="collapsed", key="ht_unit")
    if _ht_unit == "cm":
        _ht_cm = float(st.slider("📏 Height (cm)", 140, 210, 170, 1))
    else:
        _ft = st.slider("📏 Feet", 4, 7, 5, 1)
        _in_ = st.slider("📏 Inches", 0, 11, 7, 1)
        _ht_cm = float(_ft * 30.48 + _in_ * 2.54)
    _wt_kg = st.slider("⚖️ Weight (kg)", 30, 180, 70, 1)
    bmi_val = round(_wt_kg / ((_ht_cm / 100) ** 2), 1)
    bmi_cat_disp = "Underweight" if bmi_val < 18.5 else "Healthy" if bmi_val < 25 else "Overweight" if bmi_val < 30 else "Obese"
    bmi_col = "#00ffb4" if bmi_val < 25 else ("#ffb400" if bmi_val < 30 else "#ff3264")
    st.markdown(f"<div style='font-family:Share Tech Mono;font-size:0.8rem;padding:0.4rem 0.6rem;background:rgba(0,0,0,0.3);border-radius:6px;border:1px solid rgba(0,255,180,0.15);margin-bottom:0.5rem'>BMI: <span style='color:{bmi_col};font-weight:700'>{bmi_val}</span> <span style='color:rgba(200,224,240,0.4);font-size:0.7rem'>({bmi_cat_disp})</span></div>", unsafe_allow_html=True)
    age_val  = st.slider("🗓️ Age", 18, 80, 35, 1)
    smoking  = st.selectbox("🚬 Smoking", ["No","Yes"])
    alcohol  = st.selectbox("🍺 Alcohol (units/day)", ["0","1","2","3+"])
    fam_dia  = st.checkbox("🧬 Family history: Diabetes")
    fam_hyp  = st.checkbox("🧬 Family history: Hypertension")
    fam_hrt  = st.checkbox("🧬 Family history: Heart Disease")

    alc_map  = {"0": 0, "1": 1, "2": 2, "3+": 3}
    diet_map = {"Excellent": 0, "Good": 1, "Average": 2, "Poor": 3}

    user_data = {
        "Sleep_Hours":      sleep_h,
        "Water_Litres":     water_l,
        "Activity_Min":     float(activity),
        "Stress_Level":     float(stress),
        "Diet_Score":       float(diet_map[diet_opt]),
        "Screen_Hours":     screen,
        "BMI":              bmi_val,
        "Age":              float(age_val),
        "Smoking":          float(1 if smoking == "Yes" else 0),
        "Alcohol_Units":    float(alc_map[alcohol]),
        "Fam_Diabetes":     float(1 if fam_dia else 0),
        "Fam_Hypertension": float(1 if fam_hyp else 0),
        "Fam_Heart":        float(1 if fam_hrt else 0),
    }

    st.markdown("<div class='nav-divider'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.58rem;color:rgba(200,224,240,0.25);text-align:center;
                letter-spacing:0.12em;line-height:1.8'>
    VITALSENSE ULTRA 2.0<br>CALIBRATED GB+RF ENSEMBLE<br>
    N=6,000 SYNTHETIC COHORT<br>5-TIER RISK CLASSIFICATION
    </div>
    """, unsafe_allow_html=True)


# ── Compute ────────────────────────────────────────────────────────────────────
risks    = predict_risks(user_data)
ov_level, ov_color, ov_cls = overall_risk(risks)
w_score  = wellness_score(user_data, risks)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1  —  RISK DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if "Risk Dashboard" in page:

    st.markdown("""
    <h1 style='font-family:Orbitron,monospace;font-size:1.5rem;font-weight:900;
               letter-spacing:0.1em;color:#00ffb4;text-shadow:0 0 20px rgba(0,255,180,0.5);
               margin-bottom:0.2rem'>
    RISK INTELLIGENCE DASHBOARD
    </h1>
    <p style='color:rgba(200,224,240,0.35);font-size:0.75rem;letter-spacing:0.18em;
              text-transform:uppercase;margin-bottom:1.5rem'>
    Multi-model health risk analysis · Adjust all inputs in the sidebar
    </p>
    """, unsafe_allow_html=True)

    # ── Row 1: KPI Cards ──
    k0, k1, k2, k3, k4 = st.columns(5)

    with k0:
        wcolor   = "red" if w_score < 40 else ("amber" if w_score < 60 else "teal")
        card_cls = "glass-card-red" if wcolor=="red" else ("glass-card-amber" if wcolor=="amber" else "glass-card-teal glow-green")
        st.markdown(f"""
        <div class='glass-card {card_cls}'>
            <div class='kpi-block'>
                <div class='kpi-value {wcolor}'>{w_score}</div>
                <div class='kpi-label'>Wellness Score</div>
                <div class='kpi-sublabel'>/100</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    overall_avg = round(np.mean(list(risks.values())), 1)
    for col, (disease, pct) in zip([k1, k2, k3, k4],
        [("Overall", overall_avg)] + list(risks.items())[:3]):
        lvl, clr, cls = risk_level(pct)
        c = "red" if lvl in ("CRITICAL","HIGH") else ("amber" if lvl == "MODERATE" else "teal")
        card_mod = "glass-card-red pulse-red" if c == "red" else ("glass-card-amber" if c == "amber" else "")
        with col:
            st.markdown(f"""
            <div class='glass-card {card_mod}'>
                <div class='kpi-block'>
                    <div class='kpi-value {c}'>{pct}%</div>
                    <div class='kpi-label'>{disease}</div>
                    <div style='margin-top:0.45rem'>
                        <span class='risk-badge {cls}'>● {lvl}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Contextual risk note ──
    if ov_level in ("CRITICAL","HIGH"):
        st.markdown(f"""
        <div class='score-context'>
        <b>📌 Understanding your scores:</b> These are <i>relative lifestyle-based risk indicators</i>,
        not medical diagnoses. They reflect how your current habits compare to healthier benchmarks.
        They update in real-time as you change the sidebar values.
        A {ov_level.lower()} score means your lifestyle profile is associated with higher risk —
        but it is fully actionable. See the <b>Prevention Protocols</b> tab for your personalised action plan.
        </div>
        """, unsafe_allow_html=True)

    # ── Row 2: Donut + Radar ──
    col_a, col_b = st.columns(2, gap="medium")

    with col_a:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>DISEASE RISK BREAKDOWN</div>", unsafe_allow_html=True)

        safe   = max(0, 300 - sum(risks.values()))
        labels = list(risks.keys()) + ["Safe Zone"]
        values = list(risks.values()) + [safe]
        colors = ["#ff3264","#ffb400","#00d4ff","rgba(0,255,180,0.06)"]

        fig_d = go.Figure(go.Pie(
            labels=labels, values=values, hole=0.65,
            marker=dict(colors=colors, line=dict(color="rgba(5,8,16,0.8)", width=3)),
            textfont=dict(family="Rajdhani", size=11),
            hovertemplate="<b>%{label}</b><br>Score: %{value:.1f}%<extra></extra>",
        ))
        fig_d.add_annotation(
            text=f"<b>{ov_level}</b>",
            font=dict(family="Orbitron", size=14, color=ov_color),
            showarrow=False, x=0.5, y=0.55,
        )
        fig_d.add_annotation(
            text=f"{overall_avg}% avg",
            font=dict(family="Share Tech Mono", size=10, color="rgba(200,224,240,0.45)"),
            showarrow=False, x=0.5, y=0.38,
        )
        neon_layout(fig_d, height=300)
        st.plotly_chart(fig_d, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>LIFESTYLE vs OPTIMAL BENCHMARK</div>", unsafe_allow_html=True)

        RADAR_MAX = {'Sleep_Hours': 10, 'Water_Litres': 4.5, 'Activity_Min': 150,
                     'Stress_Level': 10, 'Diet_Score': 3, 'Screen_Hours': 14}
        cats   = [FEATURE_LABELS[k] for k in BASE_FEATURES]
        invert = ['Stress_Level','Diet_Score','Screen_Hours']

        def norm_radar(d, invert_keys):
            return [100 - (d[k]/RADAR_MAX[k]*100) if k in invert_keys
                    else d[k]/RADAR_MAX[k]*100 for k in BASE_FEATURES]

        user_n  = norm_radar(user_data, invert)
        ideal_n = norm_radar(IDEAL, invert)

        fig_r = go.Figure()
        fig_r.add_trace(go.Scatterpolar(
            r=ideal_n + [ideal_n[0]], theta=cats + [cats[0]],
            name="Optimal", line=dict(color="#00ffb4", width=1.5, dash="dot"),
            fill="toself", fillcolor="rgba(0,255,180,0.04)",
        ))
        fig_r.add_trace(go.Scatterpolar(
            r=user_n + [user_n[0]], theta=cats + [cats[0]],
            name="You", line=dict(color="#ff3264", width=2.5),
            fill="toself", fillcolor="rgba(255,50,100,0.07)",
        ))
        fig_r.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0,100],
                    gridcolor="rgba(0,255,180,0.08)", linecolor="rgba(0,255,180,0.12)",
                    showticklabels=False),
                angularaxis=dict(
                    gridcolor="rgba(0,255,180,0.08)", linecolor="rgba(0,255,180,0.12)",
                    tickfont=dict(family="Rajdhani", size=10, color="rgba(200,224,240,0.55)")),
            ),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=300, margin=dict(l=40, r=40, t=15, b=15),
            legend=dict(bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Rajdhani", size=11, color="rgba(200,224,240,0.6)"),
                        orientation="h", y=-0.05),
        )
        st.plotly_chart(fig_r, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Row 3: Lifestyle Factor Status ──
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>LIFESTYLE FACTOR STATUS</div>", unsafe_allow_html=True)

    factor_data = [
        ("Sleep Quality",    sleep_h,           3,   10,   True,  "Sleep_Hours"),
        ("Hydration",        water_l,            0.3, 4.5,  True,  "Water_Litres"),
        ("Physical Activity",activity,           0,   150,  True,  "Activity_Min"),
        ("Stress Control",   10-stress,          0,   9,    True,  "Stress_Level"),
        ("Diet Quality",     3-user_data['Diet_Score'], 0, 3, True,"Diet_Score"),
        ("Screen Discipline",14-screen,          0.5, 13.5, True,  "Screen_Hours"),
    ]

    fc1, fc2 = st.columns(2)
    factor_cols = [fc1, fc2, fc1, fc2, fc1, fc2]

    for (fcol, (label, val, lo, hi, higher, key)) in zip(factor_cols, factor_data):
        with fcol:
            pct_fill  = max(0, min(100, (val - lo) / (hi - lo) * 100))
            color_key = delta_badge(user_data[key], key)
            bar_color = "#00ffb4" if color_key == "teal" else ("#ffb400" if color_key == "amber" else "#ff3264")
            status_text = "✓ Good" if color_key == "teal" else ("⚠ Fair" if color_key == "amber" else "✗ Needs Work")
            st.markdown(f"""
            <div class='factor-bar-wrap'>
                <div class='factor-label'>
                    <span>{label}</span>
                    <span style='color:{bar_color};font-family:Share Tech Mono;font-size:0.72rem'>{status_text}</span>
                </div>
                <div class='factor-track'>
                    <div class='factor-fill' style='width:{pct_fill}%;background:linear-gradient(90deg,{bar_color}aa,{bar_color});'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Row 4: Feature Importance Tabs ──
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>RISK DRIVER ANALYSIS</div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["  🔴 Diabetes  ", "  🟡 Hypertension  ", "  🔵 Heart Disease  "])

    for tab, disease in zip([tab1, tab2, tab3], ["Diabetes","Hypertension","Heart Disease"]):
        with tab:
            if disease in IMPORTANCES and IMPORTANCES[disease]:
                imp  = IMPORTANCES[disease]
                disp = {FEATURE_LABELS.get(k,k): v * 100 for k,v in imp.items() if k in FEATURE_LABELS}
                sorted_imp = sorted(disp.items(), key=lambda x: x[1], reverse=True)[:9]
                names = [k for k, _ in sorted_imp]
                vals  = [v for _, v in sorted_imp]

                colors_i = ["#ff3264" if v >= 22 else "#ffb400" if v >= 14 else "#00ffb4" for v in vals]
                fig_b = go.Figure(go.Bar(
                    x=vals, y=names, orientation="h",
                    marker=dict(color=colors_i, opacity=0.82, line=dict(width=0)),
                    text=[f"{v:.1f}%" for v in vals],
                    textposition="outside",
                    textfont=dict(family="Share Tech Mono", size=9, color="rgba(200,224,240,0.65)"),
                    hovertemplate="<b>%{y}</b><br>Importance: %{x:.2f}%<extra></extra>",
                ))
                neon_layout(fig_b, height=260)
                fig_b.update_xaxes(range=[0, max(vals)*1.3], showticklabels=False)
                st.plotly_chart(fig_b, use_container_width=True, config={"displayModeBar": False})

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Row 5: Individual Risk Gauges ──
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>INDIVIDUAL RISK GAUGES</div>", unsafe_allow_html=True)

    g1, g2, g3 = st.columns(3)
    gauge_colors = {"Diabetes": "#ff3264", "Hypertension": "#ffb400", "Heart Disease": "#00c8ff"}

    for gcol, (disease, pct) in zip([g1, g2, g3], risks.items()):
        gc = gauge_colors[disease]
        with gcol:
            lvl, _, _ = risk_level(pct)
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pct,
                number=dict(suffix="%", font=dict(family="Orbitron", size=22, color=gc)),
                title=dict(
                    text=f"{disease}<br><span style='font-size:0.6em;color:rgba(200,224,240,0.4)'>{lvl}</span>",
                    font=dict(family="Orbitron", size=9, color="rgba(200,224,240,0.5)")
                ),
                gauge=dict(
                    axis=dict(range=[0,100], tickcolor=gc,
                              tickfont=dict(color="rgba(200,224,240,0.3)", size=7)),
                    bar=dict(color=gc, thickness=0.2),
                    bgcolor="rgba(0,0,0,0)", borderwidth=0,
                    steps=[
                        dict(range=[0,18],   color="rgba(0,200,255,0.04)"),
                        dict(range=[18,35],  color="rgba(0,255,180,0.04)"),
                        dict(range=[35,50],  color="rgba(255,180,0,0.06)"),
                        dict(range=[50,65],  color="rgba(255,120,0,0.06)"),
                        dict(range=[65,100], color="rgba(255,50,100,0.08)"),
                    ],
                    threshold=dict(line=dict(color=gc, width=2), thickness=0.8, value=pct),
                ),
            ))
            fig_g.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=200, margin=dict(l=20, r=20, t=35, b=10),
                font=dict(family="Rajdhani"),
            )
            st.plotly_chart(fig_g, use_container_width=True, config={"displayModeBar": False})
            # Context message under each gauge
            ctx_msg = risk_context_message(pct, disease)
            if ctx_msg:
                st.markdown(f"<div style='font-size:0.75rem;color:rgba(200,224,240,0.5);text-align:center;padding:0 0.5rem;line-height:1.5'>{ctx_msg}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Row 6: Habit Gap Chart ──
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>HABIT GAP ANALYSIS — YOUR LEVELS vs TARGETS</div>",
                unsafe_allow_html=True)

    gap_labels = [FEATURE_LABELS[k] for k in BASE_FEATURES]
    your_vals  = [user_data[k] for k in BASE_FEATURES]
    ideal_vals = [IDEAL[k] for k in BASE_FEATURES]

    fig_gap = go.Figure()
    fig_gap.add_trace(go.Bar(name="Your Level", x=gap_labels, y=your_vals,
        marker=dict(color="rgba(255,50,100,0.72)", line=dict(width=0))))
    fig_gap.add_trace(go.Bar(name="Optimal Target", x=gap_labels, y=ideal_vals,
        marker=dict(color="rgba(0,255,180,0.5)", line=dict(width=0))))
    neon_layout(fig_gap, title="Lifestyle Gap: Where You Are vs Where You Should Be", height=300)
    fig_gap.update_layout(barmode="group")
    st.plotly_chart(fig_gap, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2  —  PREVENTION PROTOCOLS
# ══════════════════════════════════════════════════════════════════════════════
elif "Prevention Protocols" in page:

    st.markdown("""
    <h1 style='font-family:Orbitron,monospace;font-size:1.5rem;font-weight:900;
               letter-spacing:0.1em;color:#00ffb4;margin-bottom:0.2rem'>
    PREVENTION PROTOCOLS
    </h1>
    <p style='color:rgba(200,224,240,0.35);font-size:0.75rem;letter-spacing:0.18em;
              text-transform:uppercase;margin-bottom:1.5rem'>
    Risk-tiered interventions · Personalised to your health profile
    </p>
    """, unsafe_allow_html=True)

    # ── Risk Banner ──
    overall_avg = round(np.mean(list(risks.values())), 1)
    st.markdown(f"""
    <div class='glass-card {"glass-card-red" if ov_level in ("CRITICAL","HIGH") else "glass-card-amber" if ov_level=="MODERATE" else "glass-card-teal"}'>
    <div style='display:flex;align-items:center;gap:2rem;flex-wrap:wrap'>
        <div>
            <div style='font-family:Orbitron,monospace;font-size:0.6rem;letter-spacing:0.22em;
                        color:rgba(200,224,240,0.4);text-transform:uppercase;margin-bottom:0.3rem'>
                Composite Risk · Wellness Score
            </div>
            <div style='display:flex;align-items:baseline;gap:0.8rem'>
                <span style='font-family:Orbitron,monospace;font-size:2rem;font-weight:900;
                             color:{ov_color};filter:drop-shadow(0 0 12px {ov_color})'>{ov_level}</span>
                <span style='font-family:Share Tech Mono;font-size:1rem;color:rgba(200,224,240,0.5)'>{overall_avg}% risk · {w_score}/100 wellness</span>
            </div>
        </div>
        {"".join([
            f"<div style='text-align:center'>"
            f"<div style='font-family:Orbitron,monospace;font-size:1.1rem;font-weight:700;"
            f"color:{risk_level(pct)[1]}'>{pct}%</div>"
            f"<div style='font-size:0.68rem;letter-spacing:0.12em;color:rgba(200,224,240,0.4);text-transform:uppercase'>{d}</div>"
            f"<div style='margin-top:0.2rem'><span class='risk-badge {risk_level(pct)[2]}'>{risk_level(pct)[0]}</span></div>"
            f"</div>"
            for d, pct in risks.items()
        ])}
    </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tiered Recommendations — 5 levels × 3 diseases ──
    REC_SYSTEM = {
        "Diabetes": {
            "icon": "🔴", "color": "#ff3264",
            "CRITICAL": {
                "⚡ Immediate Steps": [
                    "Book a fasting glucose test with your doctor this week — proactive, not reactive",
                    "Stop all sugary beverages immediately: sodas, juices, sweet tea, energy drinks",
                    "Begin a food diary this week — identifying hidden carbs is step one",
                    "Take a 15-min walk after each main meal, starting today — simple yet highly effective",
                ],
                "🥗 Dietary Reset": [
                    "Reduce refined carbohydrates significantly — swap white rice/bread for cauliflower rice, whole grains",
                    "Aim for 50–100g carbs/day — a low-carb approach is most evidence-backed at this level",
                    "Eat protein first at every meal — this flattens the glucose curve by up to 30%",
                    "Increase fibre intake to 30–35g/day — legumes, vegetables, psyllium husk supplement",
                    "Avoid all ultra-processed food for at least 30 days to reset insulin sensitivity",
                ],
                "🏋️ Movement Protocol": [
                    "Brisk walk for 20–30 min after each main meal — post-meal movement is the most effective glucose control tool",
                    "Add resistance training 3–4×/week — muscle tissue is your primary glucose disposal organ",
                    "Set a timer to stand and move every 45 minutes during desk work",
                    "Work up to 8,000–10,000 steps per day over the next 4 weeks",
                ],
            },
            "HIGH": {
                "📋 Priority Actions": [
                    "Schedule a glucose tolerance test within the next 2–4 weeks",
                    "Reduce added sugar to below 15g/day — read every nutrition label",
                    "Replace one refined carb serving daily with a vegetable or legume alternative",
                    "Set a daily step target of 7,000–8,000 and track it",
                ],
                "🥗 Diet Improvements": [
                    "Increase fibre-rich legumes: lentils, chickpeas, black beans — 3+ servings/week",
                    "Choose whole grains (oats, barley, quinoa) over refined carbs consistently",
                    "Add 1 tablespoon apple cider vinegar before carb-heavy meals — blunts glucose spike",
                    "Cinnamon (½ tsp/day) and fenugreek seeds show measurable clinical glucose benefits",
                ],
                "🏋️ Movement": [
                    "Target 150 min/week of moderate aerobic activity minimum",
                    "Include 2× strength training sessions per week — muscle mass is protective",
                    "Use a continuous glucose monitor (CGM) if possible — it shows exactly how your food affects you",
                ],
            },
            "MODERATE": {
                "✅ Preventive Steps": [
                    "Annual fasting glucose check — catching pre-diabetes early transforms outcomes",
                    "Gradually reduce processed sugar over the next month — no need to be drastic",
                    "Aim for 6,000–8,000 steps daily as a sustainable starting point",
                ],
                "🥗 Diet Tweaks": [
                    "Swap one refined carb meal per day for a whole-food alternative",
                    "Increase vegetables to 5 servings/day — especially leafy greens and broccoli",
                    "Eat whole fruit instead of fruit juice — fibre intact, glucose impact is very different",
                ],
                "🏃 Activity": [
                    "30 min of moderate exercise 5 days/week — consistency beats intensity",
                    "Short 10-min walks after meals are highly effective for blood glucose balance",
                    "Build towards 2× resistance training per week over the next 4 weeks",
                ],
            },
            "LOW": {
                "💚 Maintenance": [
                    "Your current lifestyle is working — keep up the habits that got you here",
                    "Schedule a glucose check every 2 years as a simple checkpoint",
                    "Maintain sleep at 7+ hrs — sleep deprivation alone raises insulin resistance",
                ],
                "🥗 Sustain Habits": [
                    "Continue eating balanced whole foods with a good fibre intake",
                    "Keep fibre intake at 25–35g/day — gut health is strongly linked to glucose metabolism",
                    "Stay well hydrated — dehydration concentrates blood glucose",
                ],
                "🏃 Keep Moving": [
                    "Maintain your current activity level — regularity is the key variable",
                    "Blend cardio and strength training for best long-term metabolic outcomes",
                ],
            },
            "MINIMAL": {
                "🌿 Optimise Further": [
                    "Your risk is very low — focus on long-term consistency and prevention",
                    "Annual wellness checks are sufficient at this risk level",
                ],
                "🥗 Fine-tuning": [
                    "Consider time-restricted eating (12–16 hr fast) for additional metabolic benefit",
                    "Diversify your plant intake — a wide variety of vegetables feeds a healthy gut microbiome",
                ],
                "🏃 Performance": [
                    "Add variety: HIIT, swimming, or yoga to maintain metabolic flexibility",
                    "VO2 max is the single best predictor of longevity — zone 2 training builds it",
                ],
            },
        },
        "Hypertension": {
            "icon": "🟡", "color": "#ffb400",
            "CRITICAL": {
                "⚡ Urgent Steps": [
                    "Get a blood pressure reading today — a home monitor, pharmacy, or clinic all work",
                    "If your BP reads above 140/90 mmHg, see your doctor within the week",
                    "Eliminate all added salt for the next 14 days — read labels on everything",
                    "Stop all stimulant intake (caffeine, energy drinks) this week",
                ],
                "🍱 DASH Diet Protocol": [
                    "Follow the DASH protocol: aim for 9 servings of fruit and vegetables daily",
                    "Reduce sodium to below 1,500 mg/day — avoid all processed and restaurant food",
                    "Eat potassium-rich foods daily: banana, sweet potato, spinach, avocado — they counteract sodium",
                    "Eliminate alcohol for a minimum of 30 days",
                ],
                "🧘 Stress & Nervous System": [
                    "Practice 4-7-8 breathing 3× daily — demonstrated to reduce BP within sessions",
                    "Reduce screen time to under 4 hrs/day — blue light elevates cortisol at night",
                    "Progressive muscle relaxation before bed: 15-min protocol, highly effective",
                    "Cold face immersion: 30 seconds of cold water on face activates the vagal nerve",
                ],
            },
            "HIGH": {
                "📋 Action Plan": [
                    "Monitor blood pressure weekly at home and log the readings — data is valuable",
                    "Reduce sodium intake to below 2,000 mg/day",
                    "Identify your top 3 chronic stressors and address one this week",
                    "Target a minimum of 7 hours sleep — sleep deprivation raises BP by 5–10 mmHg",
                ],
                "🍱 Diet Upgrades": [
                    "Increase magnesium intake: dark chocolate (70%+), almonds, pumpkin seeds",
                    "Eat omega-3-rich fish (salmon, mackerel, sardines) 3×/week",
                    "Reduce caffeine to a maximum of 1 cup of coffee per day",
                    "Hibiscus tea (1–2 cups/day) has shown an average 7 mmHg BP reduction in clinical trials",
                ],
                "🏊 Exercise": [
                    "150 min/week aerobic exercise — swimming, cycling, and brisk walking are ideal",
                    "Yoga 3×/week — shown to reduce systolic BP by 5–8 mmHg independently",
                    "Avoid unsupported heavy isometric lifting — consult a fitness professional",
                ],
            },
            "MODERATE": {
                "✅ Prevention": [
                    "Check blood pressure annually — easy to do at any pharmacy",
                    "Reduce sodium gradually — start by simply not adding salt at the table",
                    "Build a consistent sleep schedule — same time every day is more important than total hours",
                ],
                "🍱 Food Changes": [
                    "Add one extra serving of vegetables daily — start with leafy greens",
                    "Swap processed snacks for unsalted nuts and whole fruit",
                    "Reduce alcohol to a maximum of 1 unit per day",
                ],
                "🏃 Activity": [
                    "30–45 min of aerobic activity on 5 days/week — your strongest lever",
                    "Include flexibility work — yoga and stretching reduce arterial stiffness over time",
                ],
            },
            "LOW": {
                "💚 Good Standing": [
                    "You're managing stress and lifestyle well — maintain this as your baseline",
                    "Annual BP checks are sufficient at this risk level",
                ],
                "🍱 Sustain": [
                    "Maintain your low-sodium habits — they compound over years",
                    "Keep alcohol in moderation as a consistent practice",
                ],
                "🧘 Wellbeing": [
                    "Continue any stress management practices you have in place",
                    "Prioritise 7–9 hours of quality sleep — it's foundational for vascular health",
                ],
            },
            "MINIMAL": {
                "🌿 Excellent State": [
                    "Very low hypertension risk — your vascular health is well-protected",
                    "Continue monitoring annually — prevention is always easier than treatment",
                ],
                "🍱 Optimise": [
                    "Experiment with beet juice — its nitrate compounds measurably support vascular health",
                    "Stay well-hydrated — even mild dehydration stresses blood pressure regulation",
                ],
                "🏃 Train Smart": [
                    "Zone 2 cardio 3–4×/week is the strongest long-term vascular health investment",
                    "Include breathing exercises and meditation to maintain parasympathetic tone",
                ],
            },
        },
        "Heart Disease": {
            "icon": "🔵", "color": "#00c8ff",
            "CRITICAL": {
                "⚡ Priority Actions": [
                    "Book a cardiac risk assessment with your GP this week — it is important, not urgent",
                    "If you smoke, quitting is the single highest-ROI health action you can take for your heart",
                    "Eliminate all trans fats, fried food, and processed meat immediately",
                    "Request a lipid panel and ECG if you haven't had one in the past 12 months",
                ],
                "❤️ Cardiac Diet": [
                    "Begin the Mediterranean diet immediately — it reduces major cardiac events by up to 30%",
                    "Replace saturated fats with omega-3: fatty fish, walnuts, and flaxseed at every opportunity",
                    "Increase soluble fibre: oats, psyllium, beans — this measurably reduces LDL cholesterol",
                    "Eliminate trans fats completely — check every label for 'partially hydrogenated' oils",
                    "Limit red meat to a maximum of 1 serving per week",
                ],
                "🏃 Exercise (Start Gentle)": [
                    "Begin with 20-min low-intensity walks daily — never overexert at this stage",
                    "Zone 2 cardio is safest: walk at a conversational pace",
                    "Avoid competitive high-intensity exercise until a doctor has assessed your risk",
                    "Stand and move every 60 minutes — prolonged sitting is an independent cardiac risk factor",
                ],
            },
            "HIGH": {
                "📋 Priority": [
                    "Request a cholesterol panel and cardiac risk score from your doctor",
                    "If you smoke, reduce immediately — the benefit to your heart begins within 24 hours of stopping",
                    "Begin a daily stress reduction practice — chronic stress directly damages arterial walls",
                    "Aim for 45-min walks daily at a moderate, comfortable pace",
                ],
                "❤️ Heart-Smart Diet": [
                    "Eat salmon, sardines, or mackerel at least 3×/week for omega-3s",
                    "Add 30g of walnuts daily — omega-3 and polyphenols are directly cardioprotective",
                    "Dark chocolate (70%+, 1–2 squares/day) — flavonoids improve endothelial function",
                    "Berries daily — their anthocyanins reduce oxidative stress and vascular inflammation",
                ],
                "🏋️ Cardio Protocol": [
                    "45 min Zone 2 cardio (60–70% max heart rate) 4×/week — the strongest cardiac longevity investment",
                    "VO2 max is the most powerful predictor of cardiac survival — train it progressively",
                    "Add 2 resistance sessions/week — muscle mass is strongly protective of the heart",
                ],
            },
            "MODERATE": {
                "✅ Prevention": [
                    "Annual lipid panel is recommended — catching elevated LDL early is very manageable",
                    "Reduce chronic stress progressively — identify your top stressors and address one at a time",
                    "Build consistent sleep habits — 7–9 hours nightly is foundational for heart health",
                ],
                "❤️ Diet": [
                    "Increase plant-based meals to 4+ per week — start with one new recipe",
                    "Choose olive oil over butter and seed oils as your default cooking fat",
                    "Limit ultra-processed food to below 10% of your meals — read labels",
                ],
                "🏃 Activity": [
                    "30–45 min moderate cardio on 5 days/week — the most consistent protective habit",
                    "Include stretching and mobility work — it reduces arterial stiffness over time",
                ],
            },
            "LOW": {
                "💚 Good Status": [
                    "Your cardiac risk is low — the habits you have are working, keep them",
                    "Cholesterol check every 3–5 years at this level",
                ],
                "❤️ Maintain": [
                    "Continue heart-friendly eating patterns consistently",
                    "Stay smoke-free — this is your most powerful long-term cardiac protection",
                ],
                "🏃 Sustain": [
                    "Maintain regular aerobic exercise as a non-negotiable habit",
                    "Keep stress well-managed — your cardiovascular system will thank you long-term",
                ],
            },
            "MINIMAL": {
                "🌿 Excellent Cardiac Health": [
                    "Your cardiac risk profile is excellent — you're investing in your future health effectively",
                    "Continue annual health reviews to stay on track",
                ],
                "❤️ Elite Optimisation": [
                    "Consider HRV (heart rate variability) monitoring — it gives precise training and recovery feedback",
                    "Regular sauna use is associated with significantly lower cardiac mortality in long-term studies",
                ],
                "🏃 Performance Zone": [
                    "Focus on progressively improving your VO2 max through zone 2 and zone 5 training",
                    "Strength training 3×/week remains essential — muscle mass is cardioprotective at any age",
                ],
            },
        },
    }

    tier_css = {
        "CRITICAL": "rec-tier-critical", "HIGH": "rec-tier-high",
        "MODERATE": "rec-tier-moderate", "LOW": "rec-tier-low", "MINIMAL": "rec-tier-minimal"
    }
    tier_colors = {
        "CRITICAL": "#ff3264", "HIGH": "#ff7800",
        "MODERATE": "#ffb400", "LOW": "#00ffb4", "MINIMAL": "#00c8ff"
    }

    for disease, data in REC_SYSTEM.items():
        lvl, clr, cls = risk_level(risks[disease])
        with st.expander(
            f"{data['icon']}  {disease.upper()}   ·   {risks[disease]}% Risk  ·  {lvl}",
            expanded=(lvl in ["CRITICAL","HIGH","MODERATE"])
        ):
            # Context message at the top
            ctx_msg = risk_context_message(risks[disease], disease)
            if ctx_msg:
                st.markdown(f"""
                <div class='score-context' style='margin-bottom:1rem;'>
                💡 {ctx_msg}
                </div>
                """, unsafe_allow_html=True)

            tier_recs   = data.get(lvl, data.get("LOW", {}))
            tier_c      = tier_colors.get(lvl, "#00ffb4")
            tier_css_cls= tier_css.get(lvl, "rec-tier-low")

            cats = list(tier_recs.keys())
            cols = st.columns(len(cats)) if cats else []

            for col, cat in zip(cols, cats):
                items      = tier_recs[cat]
                items_html = "".join([f"<div class='rec-item'>{item}</div>" for item in items])
                with col:
                    st.markdown(f"""
                    <div class='rec-tier {tier_css_cls}'>
                        <div class='rec-tier-title' style='color:{tier_c}'>{cat}</div>
                        {items_html}
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style='margin-top:0.8rem;padding:0.6rem 0.8rem;
                        background:rgba(0,0,0,0.2);border-radius:8px;
                        font-size:0.72rem;color:rgba(200,224,240,0.4);font-style:italic'>
            ⚕ These recommendations are based on your lifestyle inputs and are for informational
            purposes only. Please consult a qualified healthcare professional before making
            significant changes to your health routine.
            </div>
            """, unsafe_allow_html=True)

    # ── Habit Gap Chart ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>HABIT GAP ANALYSIS — YOUR LEVELS vs TARGETS</div>",
                unsafe_allow_html=True)

    gap_labels = [FEATURE_LABELS[k] for k in BASE_FEATURES]
    your_vals  = [user_data[k] for k in BASE_FEATURES]
    ideal_vals = [IDEAL[k] for k in BASE_FEATURES]

    fig_gap = go.Figure()
    fig_gap.add_trace(go.Bar(name="Your Level", x=gap_labels, y=your_vals,
        marker=dict(color="rgba(255,50,100,0.72)", line=dict(width=0))))
    fig_gap.add_trace(go.Bar(name="Optimal Target", x=gap_labels, y=ideal_vals,
        marker=dict(color="rgba(0,255,180,0.5)", line=dict(width=0))))
    neon_layout(fig_gap, title="Lifestyle Gap: Where You Are vs Where You Should Be", height=300)
    fig_gap.update_layout(barmode="group")
    st.plotly_chart(fig_gap, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3  —  VITALBOT AI
# ══════════════════════════════════════════════════════════════════════════════
elif "VitalBot AI" in page:

    st.markdown("""
    <h1 style='font-family:Orbitron,monospace;font-size:1.5rem;font-weight:900;
               letter-spacing:0.1em;color:#00ffb4;margin-bottom:0.2rem'>
    VITALBOT AI ASSISTANT
    </h1>
    <p style='color:rgba(200,224,240,0.35);font-size:0.75rem;letter-spacing:0.18em;
              text-transform:uppercase;margin-bottom:1.5rem'>
    Context-aware health intelligence · Responses tailored to your real-time risk profile
    </p>
    """, unsafe_allow_html=True)

    ov_avg = round(np.mean(list(risks.values())), 1)

    # ── Initialise chat ──
    if "chat_history" not in st.session_state:
        intro = (
            f"👋 Hello! I'm **VitalBot**, your personal AI health companion.\n\n"
            f"I've just analysed your lifestyle profile. Here's your current snapshot:\n\n"
            f"| Metric | Value | Status |\n"
            f"|--------|-------|--------|\n"
            f"| 🌟 Wellness Score | **{w_score}/100** | {'Good' if w_score >= 60 else 'Needs attention'} |\n"
            f"| 🔴 Diabetes Risk | **{risks['Diabetes']}%** | {risk_level(risks['Diabetes'])[0]} |\n"
            f"| 🟡 Hypertension Risk | **{risks['Hypertension']}%** | {risk_level(risks['Hypertension'])[0]} |\n"
            f"| 🔵 Heart Disease Risk | **{risks['Heart Disease']}%** | {risk_level(risks['Heart Disease'])[0]} |\n\n"
            f"**Remember:** These are *relative lifestyle risk indicators*, not diagnoses. "
            f"They update live as you adjust the sidebar. Higher scores mean your current "
            f"habits carry more risk compared to healthy benchmarks — and they are fully improvable.\n\n"
            f"**Ask me anything** about your health, risks, diet, sleep, stress, or fitness. What would you like to explore?"
        )
        st.session_state.chat_history = [{"role": "bot", "text": intro}]

    # ── Response Engine ──
    def get_bot_response(query: str) -> str:
        q = query.lower().strip()

        # ── Stress ──
        if any(w in q for w in ["stress","anxiety","nervous","relax","calm","tense","overwhelm","burnout"]):
            lvl_s = f"⚠️ Your current stress level of **{stress}/10** is above the manageable threshold." \
                    if stress > 6 else f"✅ Your stress level of **{stress}/10** is currently in a manageable range."
            return (
                f"## 🧠 Stress Management Guide\n\n{lvl_s}\n\n"
                f"**Why this matters for your specific risks:**\n"
                f"Chronic stress elevates cortisol, which directly raises blood glucose, blood pressure, "
                f"and systemic inflammation — meaning it drives all three of your risk scores simultaneously. "
                f"Your Hypertension risk is currently **{risks['Hypertension']}%** and Diabetes risk **{risks['Diabetes']}%** "
                f"— both are sensitive to stress.\n\n"
                f"**Immediate relief (use right now):**\n"
                f"- **4-7-8 Breathing:** Inhale 4 sec → Hold 7 sec → Exhale 8 sec, × 4 rounds. "
                f"This activates the parasympathetic nervous system within minutes.\n"
                f"- **Cold face immersion:** 30 seconds of cold water on your face triggers the dive reflex, "
                f"dropping heart rate and cortisol rapidly.\n"
                f"- **5-minute body scan:** Close your eyes, progressively relax each body part from feet upward.\n\n"
                f"**Daily habits to build this week:**\n"
                f"- 10 min morning meditation — reduces cortisol by ~15% after 8 weeks of consistency\n"
                f"- Reduce screen time 1 hr before bed — blue light suppresses melatonin and spikes cortisol at night\n"
                f"- 20 min walk in nature — measurably lowers cortisol versus equivalent urban walks\n"
                f"- Journalling before bed — externalising worry reduces cognitive load and nighttime cortisol\n\n"
                f"**Evidence-backed supplements** *(always consult your doctor first):*\n"
                f"- Ashwagandha KSM-66 (600mg/day) — reduces cortisol by up to 28% in randomised trials\n"
                f"- L-theanine (200mg) — promotes calm alertness without drowsiness\n"
                f"- Magnesium glycinate (400mg at night) — supports nervous system regulation and sleep quality\n\n"
                f"**Long-term approach:** Stress management is a skill, not a one-time fix. "
                f"Start with one technique above and practise it for 2 weeks before adding another."
            )

        # ── Sleep ──
        elif any(w in q for w in ["sleep","insomnia","tired","rest","awake","fatigue","night","circadian","melatonin"]):
            sleep_status = f"⚠️ You're averaging **{sleep_h} hrs** — below the 7–9 hr optimal window." \
                           if sleep_h < 7 else f"✅ You're averaging **{sleep_h} hrs** — within the healthy range."
            return (
                f"## 😴 Sleep Optimisation Protocol\n\n{sleep_status}\n\n"
                f"**How sleep directly affects your risk scores:**\n"
                f"- Below 6 hrs of sleep raises diabetes risk by approximately 30% through impaired insulin sensitivity\n"
                f"- Sleep deprivation raises blood pressure 5–10 mmHg within 24 hours of a bad night\n"
                f"- Poor sleep is one of the strongest drivers of elevated cortisol — affecting all three of your risk areas\n"
                f"- Your Hypertension risk (**{risks['Hypertension']}%**) is particularly sensitive to sleep quality\n\n"
                f"**The non-negotiable foundations:**\n"
                f"- **Fixed wake time:** Same time every morning, including weekends — this is the most powerful circadian anchor\n"
                f"- **Temperature:** A cool room at 18–20°C is necessary for the core body temperature drop that initiates sleep\n"
                f"- **Morning light:** Get 10–15 min of direct sunlight within 1 hour of waking to set your circadian clock\n"
                f"- **Caffeine cutoff:** No caffeine after 1pm — caffeine has a 5–7 hour half-life\n\n"
                f"**The sleep-optimisation stack** *(evidence-based):*\n"
                f"- Magnesium glycinate 400mg, 30 minutes before bed\n"
                f"- L-theanine 200mg — reduces time to sleep onset\n"
                f"- Dim lights and no bright screens 45 min before bed — this is your melatonin trigger\n"
                f"- Keep your bedroom dark and cool — it's for sleep, not content consumption\n\n"
                f"**Advanced insight:** Research shows poor sleepers experience a fasting glucose spike "
                f"of 10–20 mg/dL the following morning — directly relevant to your Diabetes risk of **{risks['Diabetes']}%**."
            )

        # ── Diet / Nutrition ──
        elif any(w in q for w in ["diet","food","eat","nutrition","sugar","carb","meal","vegetable","fruit","cook","recipe"]):
            diet_status = f"⚠️ Your diet is rated **{diet_opt}** — there is meaningful room to improve." \
                          if diet_opt in ["Poor","Average"] \
                          else f"✅ Your diet quality is **{diet_opt}** — a solid foundation."
            return (
                f"## 🥗 Nutritional Strategy\n\n{diet_status}\n\n"
                f"**Tailored to your three disease risks:**\n\n"
                f"**For Diabetes ({risks['Diabetes']}% risk):**\n"
                f"- Reduce refined carbohydrates; choose whole grains, legumes, and vegetables instead\n"
                f"- Eat protein first at every meal — this flattens the glucose curve by 20–30% at that meal\n"
                f"- Fibre target: 30–35g/day (most people consume under 15g — this matters enormously)\n"
                f"- Avoid eating large carb portions at night when insulin sensitivity is at its lowest\n\n"
                f"**For Heart Disease ({risks['Heart Disease']}% risk):**\n"
                f"- Replace saturated fats with omega-3 sources: fatty fish 3×/week, walnuts and flaxseed daily\n"
                f"- The Mediterranean diet reduces cardiovascular events by up to 30% in large randomised trials\n"
                f"- Berries, dark leafy greens, and olive oil are your most protective daily foods\n\n"
                f"**For Hypertension ({risks['Hypertension']}% risk):**\n"
                f"- Follow DASH principles: 9 servings of fruit and vegetables daily\n"
                f"- Reduce sodium to below 2,000 mg/day — most people consume 3,400+ mg daily\n"
                f"- Potassium directly counteracts sodium: bananas, sweet potato, spinach, and avocado\n\n"
                f"**The single rule that helps all three:** Eat 80% whole, unprocessed food. "
                f"This one change drives more improvement across all your risk scores than any supplement or hack."
            )

        # ── Exercise / Fitness ──
        elif any(w in q for w in ["exercise","workout","activity","cardio","gym","walk","run","fitness","train","sport","strength","weights"]):
            act_status = f"⚠️ You're doing **{activity} min/day** — below the WHO recommended 60 min." \
                         if activity < 60 else f"✅ You're achieving **{activity} min/day** — meeting daily targets."
            return (
                f"## 🏃 Movement Protocol\n\n{act_status}\n\n"
                f"**Why activity is your highest-leverage variable:**\n"
                f"- Low activity is a primary driver of your Diabetes risk (**{risks['Diabetes']}%**)\n"
                f"- Just 30 min/day of exercise independently reduces hypertension risk by ~35%\n"
                f"- Zone 2 cardio is the strongest long-term cardiac protector available to you\n\n"
                f"**Optimal prescription for your risk profile:**\n\n"
                f"**Zone 2 Cardio (3–4 sessions/week):**\n"
                f"- 40–50 min at 60–70% of your maximum heart rate — you should be able to hold a conversation\n"
                f"- Builds mitochondrial density and fat-burning capacity over time\n"
                f"- Best options: brisk walking, cycling, swimming, rowing, elliptical\n\n"
                f"**Resistance Training (2–3 sessions/week):**\n"
                f"- Muscle tissue is your primary glucose disposal system — each kilogram of muscle improves insulin sensitivity\n"
                f"- Full-body compound movements: squat, deadlift, row, press — these give the best systemic benefit\n"
                f"- Even 2× bodyweight sessions/week reduces diabetes risk by approximately 34% independently\n\n"
                f"**NEAT — Non-Exercise Activity (every day):**\n"
                f"- Stand and move for 5 minutes every hour during sedentary work — set an alarm\n"
                f"- Walk phone calls, take stairs, park at a distance — these add up to 300–500 calories/day\n\n"
                f"**If you're currently sedentary:** Start with a 20-min daily walk. "
                f"Research shows this single habit reduces all-cause mortality by 30%. Build from there."
            )

        # ── Hydration ──
        elif any(w in q for w in ["water","hydrat","drink","fluid","thirst","dehydrat"]):
            water_status = f"⚠️ You're drinking **{water_l}L/day** — below the optimal 2.5–3L." \
                           if water_l < 2.5 else f"✅ You're well hydrated at **{water_l}L/day**."
            return (
                f"## 💧 Hydration Protocol\n\n{water_status}\n\n"
                f"**Why hydration directly affects your risk scores:**\n"
                f"- Dehydration raises blood viscosity, which elevates blood pressure directly\n"
                f"- Concentrated blood glucose when dehydrated — worsens insulin sensitivity\n"
                f"- Cortisol rises with mild dehydration — cascading into stress and hypertension risk\n"
                f"- Your Hypertension risk is currently **{risks['Hypertension']}%** — hydration is a controllable lever\n\n"
                f"**Your personalised target:** 35ml × your weight in kg per day as a baseline\n\n"
                f"**The optimal hydration routine:**\n"
                f"- **On waking:** 500ml of water immediately — rehydrate the overnight deficit before anything else\n"
                f"- **Before meals:** 250ml water 20 minutes before each meal — reduces appetite by ~20%\n"
                f"- **During exercise:** 500ml per hour of moderate activity, plus electrolytes for sessions over 60 min\n"
                f"- **Evening:** Taper off 2 hours before bed to avoid disrupting sleep\n\n"
                f"**Electrolyte balance:**\n"
                f"- Add a pinch of sea salt and a squeeze of lemon to your morning water\n"
                f"- Coconut water provides natural potassium and sodium without the sugar of sports drinks\n"
                f"- Avoid sports drinks unless exercising for over 90 continuous minutes\n\n"
                f"**Check your colour:** Pale straw yellow = well hydrated · Dark yellow = drink more · Clear = over-hydrated"
            )

        # ── BMI / Weight ──
        elif any(w in q for w in ["bmi","weight","overweight","obese","fat","body mass","kg","pounds","body composition"]):
            bmi_cat = "underweight" if bmi_val < 18.5 else "healthy weight" if bmi_val < 25 \
                      else "overweight" if bmi_val < 30 else "obese range"
            bmi_color = "✅" if bmi_val < 25 else ("⚠️" if bmi_val < 30 else "🔴")
            return (
                f"## ⚖️ BMI & Body Composition\n\n"
                f"{bmi_color} Your BMI of **{bmi_val:.1f}** falls in the **{bmi_cat}**.\n\n"
                f"**Important context:** BMI is a population screening tool, not a precise individual measure. "
                f"Body composition — the ratio of muscle to fat, and where fat is stored — matters more. "
                f"Someone with high muscle mass may show a high BMI but have low metabolic risk.\n\n"
                f"**How BMI affects your specific risks:**\n"
                f"- Each 5-unit BMI increase raises Diabetes risk by approximately 40%\n"
                f"- Abdominal fat (visceral fat) is the most metabolically dangerous type — waist circumference is a better predictor\n"
                f"- A 5–10% weight reduction in overweight individuals reduces blood pressure by an average of 5 mmHg\n"
                f"- Current Diabetes risk: **{risks['Diabetes']}%** · Heart Disease risk: **{risks['Heart Disease']}%**\n\n"
                f"**Healthy body composition approach:**\n"
                f"- Prioritise building muscle alongside losing fat — this improves your metabolic rate and insulin sensitivity\n"
                f"- Progressive resistance training (2–3×/week) is the single most effective tool for body composition\n"
                f"- A caloric deficit of 300–500 kcal/day produces sustainable fat loss of ~0.5kg/week\n"
                f"- Protein at 1.6–2.2g/kg of bodyweight preserves muscle tissue during a caloric deficit\n\n"
                f"**Better metric:** Keep your waist circumference below half your height in centimetres. "
                f"This waist-to-height ratio is a stronger predictor of metabolic risk than BMI."
            )

        # ── Risk Score Explanation ──
        elif any(w in q for w in ["risk","score","diabetes","hypertension","heart","predict","probability","chance","mean","explain"]):
            return (
                f"## 📊 Your Risk Profile — Explained Clearly\n\n"
                f"| Disease | Score | Level | Primary Drivers |\n"
                f"|---------|-------|-------|-----------------|\n"
                f"| 🔴 Diabetes | **{risks['Diabetes']}%** | {risk_level(risks['Diabetes'])[0]} | "
                f"{'Activity & Diet' if user_data['Activity_Min'] < 40 or user_data['Diet_Score'] > 1 else 'Well-managed'} |\n"
                f"| 🟡 Hypertension | **{risks['Hypertension']}%** | {risk_level(risks['Hypertension'])[0]} | "
                f"{'Stress & Sleep' if stress > 6 or sleep_h < 6.5 else 'Well-managed'} |\n"
                f"| 🔵 Heart Disease | **{risks['Heart Disease']}%** | {risk_level(risks['Heart Disease'])[0]} | "
                f"{'Activity & Stress' if user_data['Activity_Min'] < 40 or stress > 6 else 'Well-managed'} |\n\n"
                f"**Wellness Score: {w_score}/100**\n\n"
                f"**What these scores actually mean:**\n"
                f"- They are **relative lifestyle risk indicators**, not predictions of whether you will get a disease\n"
                f"- They reflect how your current habits compare to evidence-based healthy benchmarks\n"
                f"- A score of 40% means your lifestyle profile is associated with elevated risk relative to a healthy baseline — "
                f"it does not mean you have a 40% chance of developing the disease\n"
                f"- Scores update in real-time as you adjust your inputs in the sidebar\n\n"
                f"**Your top 3 improvement levers based on your current data:**\n"
                f"1. {'↑ Increase daily activity — your most impactful lever' if user_data['Activity_Min'] < 40 else '↑ Improve diet quality — compound daily benefits' if user_data['Diet_Score'] > 1 else '✓ Maintain current activity level'}\n"
                f"2. {'↓ Reduce stress levels — it cascades into all risk areas' if stress > 6 else '↑ Improve sleep duration' if sleep_h < 7 else '✓ Keep up your stress management'}\n"
                f"3. {'↑ Improve diet quality' if diet_opt in ['Poor','Average'] else '↑ Increase hydration' if water_l < 2 else '↓ Reduce screen time before bed'}"
            )

        # ── Smoking ──
        elif any(w in q for w in ["smok","cigarette","tobacco","quit","nicotine","vape"]):
            smoking_msg = "⚠️ You indicated **you currently smoke** — this is one of the highest modifiable risk factors for all three diseases." \
                         if user_data['Smoking'] == 1 \
                         else "✅ You are **smoke-free** — this is one of the most significant things you can do for your heart and lungs."
            return (
                f"## 🚬 Smoking & Disease Risk\n\n{smoking_msg}\n\n"
                f"**The specific impact on your risk scores:**\n"
                f"- Smoking alone increases Heart Disease risk by 2–4× — it is the single highest modifiable cardiac risk factor\n"
                f"- It directly damages arterial walls, elevates blood pressure, and reduces oxygen-carrying capacity\n"
                f"- Doubles Hypertension risk and measurably raises insulin resistance\n"
                f"- Your Heart Disease risk is currently **{risks['Heart Disease']}%**\n\n"
                f"**If you smoke — quitting is the highest return action you can take:**\n"
                f"- Heart attack risk drops by 50% within just 1 year of quitting\n"
                f"- Lung function starts improving within 1–9 months\n"
                f"- By year 10, your heart disease risk approaches that of a non-smoker\n\n"
                f"**Proven cessation strategies (ranked by effectiveness):**\n"
                f"1. Combination NRT (patch + gum or lozenge) — doubles quit success rate vs. going cold turkey\n"
                f"2. Varenicline (Champix/Chantix) — the most effective pharmacological aid — consult your GP\n"
                f"3. Behavioural support combined with NRT is the gold-standard approach\n"
                f"4. Identify your top 3 smoking triggers and plan specific replacements for each\n"
                f"5. Set a quit date 2 weeks out and prepare your environment in advance"
            )

        # ── Alcohol ──
        elif any(w in q for w in ["alcohol","drink","wine","beer","spirits","units"]):
            alc_msg = f"Your current alcohol intake is **{alcohol} unit(s)/day**."
            alc_context = "✅ This is within safe limits." if alc_map.get(alcohol, 0) <= 1 \
                          else "⚠️ This is above recommended levels and is impacting your risk scores."
            return (
                f"## 🍺 Alcohol & Your Health\n\n{alc_msg} {alc_context}\n\n"
                f"**Alcohol's specific impact on your risk profile:**\n"
                f"- More than 2 units/day raises both blood pressure and Heart Disease risk significantly\n"
                f"- Alcohol is calorie-dense and disrupts blood glucose regulation — relevant to your Diabetes risk (**{risks['Diabetes']}%**)\n"
                f"- Even one drink measurably reduces sleep architecture quality — suppressing deep sleep\n"
                f"- Alcohol disrupts cortisol patterns, worsening stress-related health risks\n\n"
                f"**Evidence-based guidelines:**\n"
                f"- WHO position: no level of alcohol is completely safe for health\n"
                f"- If you choose to drink: ≤ 1 unit/day for women, ≤ 2 units/day for men as a general limit\n"
                f"- At least 2 alcohol-free days per week — liver regeneration occurs during abstinence\n\n"
                f"**Reducing harm if you choose to drink:**\n"
                f"- Never drink on an empty stomach — food significantly slows glucose spikes\n"
                f"- Match each alcoholic drink with a glass of water — reduces total consumption and dehydration\n"
                f"- Avoid cocktails and mixers with added sugar\n"
                f"- Do not use alcohol as stress relief — it disrupts the stress hormone cycle within days"
            )

        # ── Wellness Score ──
        elif any(w in q for w in ["wellness","health score","overall","how healthy","how am i doing","score"]):
            return (
                f"## 🌿 Your Wellness Score: {w_score}/100\n\n"
                f"{'🔴 Your score indicates significant room for improvement across multiple lifestyle areas.' if w_score < 40 else '🟡 Your score is moderate — targeted improvements will make a meaningful difference.' if w_score < 60 else '✅ Your wellness score is good — you are actively managing your health well.'}\n\n"
                f"**Score breakdown:**\n"
                f"| Factor | Your Value | Status |\n"
                f"|--------|-----------|--------|\n"
                f"| Sleep | {sleep_h} hrs/night | {'✓ Good' if sleep_h >= 7 else '↑ Improve'} |\n"
                f"| Hydration | {water_l}L/day | {'✓ Good' if water_l >= 2.5 else '↑ Increase'} |\n"
                f"| Activity | {activity} min/day | {'✓ Good' if activity >= 60 else '↑ Increase'} |\n"
                f"| Stress | {stress}/10 | {'✓ Good' if stress <= 5 else '↓ Reduce'} |\n"
                f"| Diet | {diet_opt} | {'✓ Good' if diet_opt in ['Excellent','Good'] else '↑ Improve'} |\n"
                f"| Screen time | {screen} hrs | {'✓ Good' if screen <= 5 else '↓ Reduce'} |\n\n"
                f"**Your highest-impact improvements:**\n"
                f"1. {'Reduce stress — it cascades into every other metric' if stress > 6 else 'Increase daily movement — single highest impact change' if activity < 40 else 'Improve diet quality — compounding daily benefit'}\n"
                f"2. {'Improve sleep duration to reach 7+ hours' if sleep_h < 7 else 'Increase hydration to 2.5–3L/day' if water_l < 2 else 'Reduce screen time, especially in the evening'}\n"
                f"3. {'Cut screen time before bed — it disrupts sleep and cortisol cycles' if screen > 6 else 'Add resistance training 2×/week for metabolic health'}"
            )

        # ── Heart-specific ──
        elif any(w in q for w in ["heart","cardiac","cardiovascular","cholesterol","chest","artery","arteries"]):
            return (
                f"## ❤️ Cardiac Health Profile\n\n"
                f"Your Heart Disease risk score is currently **{risks['Heart Disease']}%** — level: **{risk_level(risks['Heart Disease'])[0]}**\n\n"
                f"{risk_context_message(risks['Heart Disease'], 'Heart Disease')}\n\n"
                f"**Key drivers in your profile:**\n"
                f"- Activity level ({activity} min/day) — {'⚠️ below the cardiac-protective threshold' if activity < 40 else '✅ at a protective level'}\n"
                f"- Stress ({stress}/10) — {'⚠️ elevated stress directly harms arterial walls' if stress > 6 else '✅ well-managed'}\n"
                f"- Diet ({diet_opt}) — {'⚠️ poor diet quality increases LDL and inflammation' if diet_opt in ['Poor','Average'] else '✅ good foundation'}\n"
                f"- Smoking: {'⚠️ active smoker — single highest modifiable risk' if user_data['Smoking'] else '✅ non-smoker'}\n\n"
                f"**Top cardiac-protective actions for your profile:**\n"
                f"1. Zone 2 cardio 4×/week — the most evidence-backed cardiac longevity intervention\n"
                f"2. Mediterranean diet — reduces cardiac events by up to 30% in randomised trials\n"
                f"3. Stress reduction practice daily — cortisol directly damages endothelial cells\n"
                f"4. Sleep optimisation — poor sleep is an independent cardiac risk factor\n\n"
                f"See the **Prevention Protocols** tab for your complete tiered cardiac action plan."
            )

        # ── Greeting / Help ──
        elif any(w in q for w in ["hello","hi","hey","good morning","good evening","what can you","help","start","begin"]):
            return (
                f"## 👋 Welcome to VitalBot!\n\n"
                f"I'm your AI health companion. I've analysed your current lifestyle profile and I'm ready to give you "
                f"specific, actionable guidance.\n\n"
                f"**Your snapshot right now:**\n"
                f"Wellness **{w_score}/100** · Overall Risk **{ov_level}** ({ov_avg}% average)\n\n"
                f"**Topics I can help you with:**\n"
                f"- 🧠 **Stress & mental health** — evidence-based management strategies\n"
                f"- 😴 **Sleep optimisation** — science-backed protocols for better sleep\n"
                f"- 🥗 **Nutrition** — personalised to all three of your disease risk areas\n"
                f"- 🏃 **Exercise** — a complete movement prescription for your profile\n"
                f"- 💧 **Hydration** — specific targets and daily routines\n"
                f"- ⚖️ **BMI & body composition** — what the numbers actually mean\n"
                f"- 📊 **Risk scores** — what they mean and exactly how to lower them\n"
                f"- 🚬 **Smoking & alcohol** — impact on each disease risk and cessation strategies\n"
                f"- ❤️ **Heart health** — cardiac-specific guidance\n\n"
                f"Just ask me anything — the more specific you are, the more personalised my answer will be!"
            )

        # ── Default ──
        else:
            return (
                f"## 🤖 VitalBot Health Intelligence\n\n"
                f"Your current profile: Wellness **{w_score}/100** · Overall risk **{ov_level}** · "
                f"Diabetes **{risks['Diabetes']}%** · HTN **{risks['Hypertension']}%** · Heart **{risks['Heart Disease']}%**\n\n"
                f"I specialise in lifestyle medicine and your specific risk profile. "
                f"Here are some example questions to get started:\n\n"
                f"- *'How do I manage my stress?'*\n"
                f"- *'Give me a sleep improvement plan'*\n"
                f"- *'What should I eat for my heart?'*\n"
                f"- *'Why is my diabetes risk high?'*\n"
                f"- *'Explain my risk scores'*\n"
                f"- *'How can I improve my wellness score?'*\n"
                f"- *'How does smoking affect my risks?'*\n"
                f"- *'Build me an exercise plan'*\n\n"
                f"The more specific your question, the more personalised and useful my answer will be."
            )

    # ── Render Chat ──
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class='chat-bubble-user'>
                <div class='chat-label'>YOU</div>
                <span class='chat-avatar'>👤</span>{msg['text']}
            </div>
            """, unsafe_allow_html=True)
        else:
            # Render markdown in bot messages properly
            with st.container():
                st.markdown(f"""
                <div class='chat-bubble-bot' style='position:relative'>
                    <div class='chat-label'>⚡ VITALBOT</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(msg['text'])

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Quick Action Buttons ──
    st.markdown("<div class='section-title'>QUICK TOPICS</div>", unsafe_allow_html=True)
    qa_row1 = st.columns(5)
    qa_row2 = st.columns(5)

    def quick_action(label, query, col):
        with col:
            if st.button(label, key=f"qa_{label}"):
                st.session_state.chat_history.append({"role": "user",  "text": query})
                st.session_state.chat_history.append({"role": "bot",   "text": get_bot_response(query)})
                st.rerun()

    quick_action("😰 Stress",       "How do I manage my stress?",          qa_row1[0])
    quick_action("😴 Sleep",        "Give me a sleep improvement plan",    qa_row1[1])
    quick_action("🥗 Nutrition",    "What should I eat for my health?",    qa_row1[2])
    quick_action("🏃 Exercise",     "Build me an exercise plan",           qa_row1[3])
    quick_action("💧 Hydration",    "How much water should I drink?",      qa_row1[4])
    quick_action("📊 Risk Scores",  "Explain my risk scores",              qa_row2[0])
    quick_action("⚖️ BMI",          "What does my BMI mean?",              qa_row2[1])
    quick_action("🚬 Smoking",      "How does smoking affect my risks?",   qa_row2[2])
    quick_action("🍺 Alcohol",      "How does alcohol affect my health?",  qa_row2[3])
    quick_action("🌿 Wellness",     "How can I improve my wellness score?",qa_row2[4])

    # ── Chat Input ──
    st.markdown("<br>", unsafe_allow_html=True)
    user_msg = st.text_input(
        "",
        placeholder="⚡ Ask VitalBot anything about your health or risk factors...",
        label_visibility="collapsed",
        key="chat_input_v2",
    )
    if user_msg and user_msg.strip():
        st.session_state.chat_history.append({"role": "user", "text": user_msg})
        st.session_state.chat_history.append({"role": "bot",  "text": get_bot_response(user_msg)})
        st.rerun()

    col_clear, col_export = st.columns([1, 4])
    with col_clear:
        if st.button("🗑  Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4  —  DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif "Data Explorer" in page:

    st.markdown("""
    <h1 style='font-family:Orbitron,monospace;font-size:1.5rem;font-weight:900;
               letter-spacing:0.1em;color:#00ffb4;margin-bottom:0.2rem'>
    COHORT DATA EXPLORER
    </h1>
    <p style='color:rgba(200,224,240,0.35);font-size:0.75rem;letter-spacing:0.18em;
              text-transform:uppercase;margin-bottom:1.5rem'>
    6,000-sample synthetic cohort · Calibrated ensemble model analysis
    </p>
    """, unsafe_allow_html=True)

    @st.cache_data
    def get_enriched_df():
        df = SAMPLE_DF.copy()
        for disease, (model, feats) in MODELS.items():
            df[f"{disease}_Risk%"] = (model.predict_proba(df[feats])[:, 1] * 100).round(1)
        diet_map_rev = {0:"Excellent", 1:"Good", 2:"Average", 3:"Poor"}
        df["Diet_Quality"] = df["Diet_Score"].map(diet_map_rev)
        return df

    rich_df = get_enriched_df()

    # ── Row 1: Sleep histogram + Scatter ──
    ca, cb = st.columns(2, gap="medium")

    with ca:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>SLEEP DISTRIBUTION</div>", unsafe_allow_html=True)
        fig_h = go.Figure(go.Histogram(
            x=rich_df["Sleep_Hours"], nbinsx=35,
            marker=dict(color="#00ffb4", opacity=0.65, line=dict(width=0)),
            hovertemplate="Sleep: %{x:.1f}h · Count: %{y}<extra></extra>",
        ))
        fig_h.add_vline(x=sleep_h, line=dict(color="#ff3264", width=2, dash="dash"),
                        annotation_text="YOU", annotation_font_color="#ff3264",
                        annotation_font_family="Orbitron", annotation_font_size=9)
        neon_layout(fig_h, height=250)
        st.plotly_chart(fig_h, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with cb:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>ACTIVITY vs DIABETES RISK</div>", unsafe_allow_html=True)
        samp = rich_df.sample(600, random_state=42)
        fig_sc = go.Figure(go.Scatter(
            x=samp["Activity_Min"], y=samp["Diabetes_Risk%"],
            mode="markers",
            marker=dict(
                color=samp["Diabetes_Risk%"],
                colorscale=[[0,"#00ffb4"],[0.5,"#ffb400"],[1,"#ff3264"]],
                size=4, opacity=0.55, showscale=False,
            ),
            hovertemplate="Activity: %{x} min · Risk: %{y:.1f}%<extra></extra>",
        ))
        fig_sc.add_trace(go.Scatter(
            x=[activity], y=[risks["Diabetes"]],
            mode="markers",
            marker=dict(color="#ffffff", size=13, symbol="star",
                        line=dict(color="#ff3264", width=2)),
            name="You",
        ))
        neon_layout(fig_sc, height=250)
        st.plotly_chart(fig_sc, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Row 2: Correlation Matrix ──
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>FEATURE CORRELATION MATRIX</div>", unsafe_allow_html=True)

    display_cols = BASE_FEATURES + ["Diabetes_Risk%","Hypertension_Risk%","Heart Disease_Risk%"]
    corr         = rich_df[display_cols].corr()
    label_map    = {**FEATURE_LABELS, "Diabetes_Risk%": "Diabetes Risk",
                    "Hypertension_Risk%": "HTN Risk", "Heart Disease_Risk%": "Heart Risk"}

    fig_heat = go.Figure(go.Heatmap(
        z=corr.values,
        x=[label_map.get(c,c) for c in corr.columns],
        y=[label_map.get(c,c) for c in corr.index],
        colorscale=[[0,"#ff3264"],[0.5,"rgba(14,20,38,0.9)"],[1,"#00ffb4"]],
        zmin=-1, zmax=1,
        text=corr.values.round(2),
        texttemplate="%{text}",
        textfont=dict(size=8, family="Share Tech Mono", color="rgba(200,224,240,0.75)"),
        hovertemplate="%{y} × %{x}<br>Correlation: %{z:.3f}<extra></extra>",
    ))
    neon_layout(fig_heat, height=400)
    st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Row 3: Box Plots — Risk by Diet Quality ──
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>RISK BY DIET QUALITY</div>", unsafe_allow_html=True)

    bc1, bc2, bc3 = st.columns(3)
    diet_order  = ["Excellent","Good","Average","Poor"]
    diet_colors = ["#00ffb4","#00c8ff","#ffb400","#ff3264"]

    for col, disease in zip([bc1, bc2, bc3], ["Diabetes","Hypertension","Heart Disease"]):
        with col:
            fig_box = go.Figure()
            for i, dq in enumerate(diet_order):
                sub = rich_df[rich_df["Diet_Quality"] == dq][f"{disease}_Risk%"]
                if len(sub) > 0:
                    fig_box.add_trace(go.Box(
                        y=sub, name=dq,
                        marker=dict(color=diet_colors[i], opacity=0.7),
                        line=dict(color=diet_colors[i]), boxmean=True,
                    ))
            neon_layout(fig_box, title=disease, height=280)
            st.plotly_chart(fig_box, use_container_width=True, config={"displayModeBar": False})

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Row 4: Risk Score Distributions ──
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>RISK SCORE DISTRIBUTIONS — POPULATION VIEW</div>",
                unsafe_allow_html=True)

    d1, d2, d3 = st.columns(3)
    dist_colors = {"Diabetes": "#ff3264", "Hypertension": "#ffb400", "Heart Disease": "#00c8ff"}

    for col, disease in zip([d1, d2, d3], ["Diabetes","Hypertension","Heart Disease"]):
        with col:
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=rich_df[f"{disease}_Risk%"], nbinsx=30,
                marker=dict(color=dist_colors[disease], opacity=0.65, line=dict(width=0)),
                name="Population",
            ))
            fig_dist.add_vline(x=risks[disease], line=dict(color="#ffffff", width=2, dash="dot"),
                               annotation_text="You", annotation_font_color="#ffffff",
                               annotation_font_family="Orbitron", annotation_font_size=8)
            neon_layout(fig_dist, title=disease, height=220)
            fig_dist.update_xaxes(title_text="Risk %", title_font=dict(size=9))
            st.plotly_chart(fig_dist, use_container_width=True, config={"displayModeBar": False})

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Row 5: Stress vs Hypertension scatter ──
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>STRESS LEVEL vs HYPERTENSION RISK</div>", unsafe_allow_html=True)

    samp2 = rich_df.sample(500, random_state=99)
    fig_s2 = go.Figure(go.Scatter(
        x=samp2["Stress_Level"], y=samp2["Hypertension_Risk%"],
        mode="markers",
        marker=dict(
            color=samp2["Hypertension_Risk%"],
            colorscale=[[0,"#00ffb4"],[0.5,"#ffb400"],[1,"#ff3264"]],
            size=5, opacity=0.5, showscale=True,
            colorbar=dict(title="HTN Risk %", title_font=dict(size=9),
                          tickfont=dict(size=8), thickness=10)
        ),
        hovertemplate="Stress: %{x} · HTN Risk: %{y:.1f}%<extra></extra>",
    ))
    fig_s2.add_trace(go.Scatter(
        x=[stress], y=[risks["Hypertension"]],
        mode="markers",
        marker=dict(color="#ffffff", size=14, symbol="star",
                    line=dict(color="#ffb400", width=2)),
        name="You",
    ))
    neon_layout(fig_s2, height=280)
    fig_s2.update_xaxes(title_text="Stress Level (1-10)", title_font=dict(size=9))
    fig_s2.update_yaxes(title_text="Hypertension Risk %", title_font=dict(size=9))
    st.plotly_chart(fig_s2, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Raw Data Sample ──
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>SAMPLE COHORT — 200 ROWS</div>", unsafe_allow_html=True)
    show_df = rich_df[
        BASE_FEATURES + ["Diabetes_Risk%","Hypertension_Risk%","Heart Disease_Risk%","Diet_Quality"]
    ].sample(200, random_state=42).reset_index(drop=True)
    def color_risk(val):
        """Color cells by risk value without requiring matplotlib."""
        try:
            v = float(val)
        except Exception:
            return ""
        if v >= 65:
            return "background-color: rgba(255,50,100,0.25); color: #ff3264; font-weight:600"
        elif v >= 50:
            return "background-color: rgba(255,120,0,0.2); color: #ff7800; font-weight:600"
        elif v >= 35:
            return "background-color: rgba(255,180,0,0.15); color: #ffb400; font-weight:600"
        elif v >= 18:
            return "background-color: rgba(0,255,180,0.1); color: #00ffb4"
        else:
            return "background-color: rgba(0,200,255,0.1); color: #00c8ff"

    risk_cols = ["Diabetes_Risk%", "Hypertension_Risk%", "Heart Disease_Risk%"]
    styled = show_df.style.applymap(color_risk, subset=risk_cols)
    st.dataframe(styled, height=320, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
