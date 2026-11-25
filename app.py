from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="NetInsight | Network Security",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_FILE = BASE_DIR / "dataset" / "cleaned_data" / "netinsight_cleaned.csv"


# --------------------------------------------------
# CUSTOM DESIGN
# --------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --bg: #07111f;
        --panel: #101d2e;
        --panel-soft: #14243a;
        --line: rgba(148, 163, 184, 0.16);
        --text: #edf4ff;
        --muted: #9aabc2;
        --cyan: #42d6e8;
        --green: #39d98a;
        --red: #ff647c;
    }

    .stApp {
        background:
            radial-gradient(circle at 82% 0%, rgba(25, 95, 125, 0.20), transparent 28%),
            linear-gradient(135deg, #07111f 0%, #0a1626 55%, #081321 100%);
        color: var(--text);
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stHeader"] { background: rgba(7, 17, 31, 0.88); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1a2b 0%, #091321 100%);
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] * { color: #dce9f8; }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2.5rem;
        max-width: 1500px;
    }

    h1, h2, h3, h4 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: #f4f8ff !important;
        letter-spacing: -0.025em;
    }

    p, label, .stMarkdown { color: #c5d2e3; }

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 8px 0 22px 0;
    }
    .brand-icon {
        width: 42px;
        height: 42px;
        display: grid;
        place-items: center;
        border-radius: 13px;
        background: linear-gradient(135deg, #123d58, #126b78);
        border: 1px solid rgba(66, 214, 232, .35);
        font-size: 22px;
    }
    .brand-name {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 21px;
        font-weight: 700;
        color: #f5f9ff;
        line-height: 1.1;
    }
    .brand-sub {
        color: #8fa4bc;
        font-size: 11px;
        margin-top: 4px;
    }

    .eyebrow {
        color: var(--cyan);
        text-transform: uppercase;
        letter-spacing: .14em;
        font-size: 11px;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(28px, 3vw, 38px);
        font-weight: 700;
        color: #f5f9ff;
        line-height: 1.15;
        margin: 0;
    }
    .hero-subtitle {
        color: #9aabc2;
        margin-top: 9px;
        font-size: 14px;
    }

    .metric-card {
        background: linear-gradient(145deg, rgba(20, 36, 58, .98), rgba(13, 27, 44, .98));
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 18px 19px;
        min-height: 112px;
        box-shadow: 0 10px 28px rgba(0,0,0,.12);
    }
    .metric-label {
        color: #9aabc2;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 12px;
    }
    .metric-value {
        color: #f4f8ff;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 29px;
        font-weight: 700;
        line-height: 1;
    }
    .metric-note {
        color: #8297b0;
        font-size: 11px;
        margin-top: 9px;
    }
    .metric-accent { color: var(--cyan); }
    .metric-green { color: var(--green); }
    .metric-red { color: var(--red); }

    .section-card {
        background: rgba(16, 29, 46, .78);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 18px 20px;
    }
    .section-kicker {
        color: #8fa4bc;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: .12em;
        font-weight: 700;
    }
    .status-pill {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        color: #62e6a5;
        background: rgba(57, 217, 138, .10);
        border: 1px solid rgba(57, 217, 138, .25);
        font-size: 11px;
        font-weight: 700;
    }
    .small-muted { color: #8fa4bc; font-size: 12px; }
    .stButton > button, .stDownloadButton > button {
        border-radius: 10px;
        border: 1px solid rgba(66,214,232,.35);
        background: linear-gradient(135deg, #126477, #167d85);
        color: white;
        font-weight: 700;
        min-height: 42px;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: #42d6e8;
        color: white;
        background: #176d7d;
    }
    [data-testid="stFileUploader"] {
        background: rgba(16, 29, 46, .65);
        border: 1px dashed rgba(66,214,232,.35);
        border-radius: 14px;
        padding: 12px;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 12px;
        overflow: hidden;
    }
    hr { border-color: var(--line) !important; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# DATA AND MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_DIR / "netinsight_model.pkl")
    features = joblib.load(MODEL_DIR / "feature_names.pkl")
    return model, features


@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)


try:
    model, feature_names = load_model()
    df = load_data()
except Exception as exc:
    st.error(f"Unable to load project files: {exc}")
    st.info(
        "Check that the models folder contains netinsight_model.pkl and "
        "feature_names.pkl, and that the cleaned dataset exists at "
        "dataset/cleaned_data/netinsight_cleaned.csv."
    )
    st.stop()

if "Target" not in df.columns:
    st.error("The cleaned dataset does not contain the required 'Target' column.")
    st.stop()


# --------------------------------------------------
# SHARED HELPERS
# --------------------------------------------------
def page_heading(kicker, title, subtitle):
    st.markdown(
        f"""
        <div class="eyebrow">{kicker}</div>
        <div class="hero-title">{title}</div>
        <div class="hero-subtitle">{subtitle}</div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label, value, note, accent=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value {accent}">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_chart(fig, height=360):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color="#c5d2e3", size=12),
        title_font=dict(family="Space Grotesk, sans-serif", color="#f4f8ff", size=16),
        margin=dict(l=18, r=18, t=55, b=18),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="rgba(148,163,184,.10)", zerolinecolor="rgba(148,163,184,.10)"),
        yaxis=dict(gridcolor="rgba(148,163,184,.10)", zerolinecolor="rgba(148,163,184,.10)"),
    )
    return fig


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">🛡️</div>
            <div>
                <div class="brand-name">NetInsight</div>
                <div class="brand-sub">AI NETWORK SECURITY</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-kicker">WORKSPACE</div>', unsafe_allow_html=True)
    page = st.radio(
        "Navigation",
        ["Dashboard", "Attack Detection", "Traffic Analysis", "Model Performance"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown('<span class="status-pill">● MODEL ONLINE</span>', unsafe_allow_html=True)
    st.markdown(
        '<p class="small-muted">Random Forest · CIC-IDS2017</p>',
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        '<p class="small-muted">NetInsight<br>Network Intrusion Detection Research Project</p>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------
total = int(len(df))
benign = int((df["Target"] == "BENIGN").sum())
attacks = int((df["Target"] == "ATTACK").sum())
benign_pct = (benign / total * 100) if total else 0
attack_pct = (attacks / total * 100) if total else 0

if page == "Dashboard":
    page_heading(
        "SECURITY OVERVIEW",
        "Network Security Dashboard",
        "A consolidated view of the prepared CIC-IDS2017 network traffic dataset.",
    )
    st.write("")

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        metric_card("TOTAL NETWORK RECORDS", f"{total:,}", "Prepared dataset records", "metric-accent")
    with c2:
        metric_card("BENIGN TRAFFIC", f"{benign:,}", f"{benign_pct:.2f}% of dataset", "metric-green")
    with c3:
        metric_card("ATTACK TRAFFIC", f"{attacks:,}", f"{attack_pct:.2f}% of dataset", "metric-red")

    st.write("")
    left, right = st.columns([1.35, 1], gap="large")

    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("**Traffic distribution**")
        traffic_counts = pd.DataFrame(
            {"Traffic type": ["Benign", "Attack"], "Records": [benign, attacks]}
        )
        fig = px.bar(
            traffic_counts,
            x="Traffic type",
            y="Records",
            color="Traffic type",
            color_discrete_map={"Benign": "#39d98a", "Attack": "#ff647c"},
            text="Records",
        )
        fig.update_traces(texttemplate="%{text:,}", textposition="outside", marker_line_width=0)
        fig.update_yaxes(title="Number of records", rangemode="tozero")
        fig.update_xaxes(title="")
        style_chart(fig, 350)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("**Dataset composition**")
        pie_data = pd.DataFrame(
            {"Traffic type": ["Benign", "Attack"], "Records": [benign, attacks]}
        )
        fig = px.pie(
            pie_data,
            names="Traffic type",
            values="Records",
            hole=0.68,
            color="Traffic type",
            color_discrete_map={"Benign": "#39d98a", "Attack": "#ff647c"},
        )
        fig.update_traces(textposition="outside", textinfo="percent+label")
        style_chart(fig, 350)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown("#### Dataset preview")
    st.caption("First 8 rows of the prepared dataset.")
    st.dataframe(df.head(8), width="stretch", height=270)


# --------------------------------------------------
# ATTACK DETECTION
# --------------------------------------------------
elif page == "Attack Detection":
    page_heading(
        "TRAFFIC INSPECTION",
        "Network Attack Detection",
        "Upload a CSV containing the 78 model input features to classify records.",
    )
    st.write("")
    st.markdown(
        '<div class="section-card"><b>How it works</b><br>'
        '<span class="small-muted">NetInsight checks the uploaded column names, prepares numeric values, '
        'and uses the trained Random Forest model to predict BENIGN or ATTACK.</span></div>',
        unsafe_allow_html=True,
    )
    st.write("")
    uploaded_file = st.file_uploader(
        "Choose a network traffic CSV file",
        type=["csv"],
        help="The CSV must contain all 78 feature columns used during model training.",
    )

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            uploaded_df.columns = uploaded_df.columns.astype(str).str.strip()

            missing = [col for col in feature_names if col not in uploaded_df.columns]
            if missing:
                st.error(
                    f"This file is missing {len(missing)} required model feature(s). "
                    f"Examples: {', '.join(missing[:8])}"
                )
                st.caption("Please upload a compatible CIC-IDS2017 feature CSV.")
            else:
                X = uploaded_df[feature_names].copy()
                for col in feature_names:
                    X[col] = pd.to_numeric(X[col], errors="coerce")
                X = X.replace([np.inf, -np.inf], np.nan)
                X = X.fillna(X.median()).fillna(0)

                with st.spinner("Analysing network traffic..."):
                    predictions = model.predict(X)

                results = uploaded_df.copy()
                results["Prediction"] = predictions

                if hasattr(model, "predict_proba") and "ATTACK" in list(model.classes_):
                    probabilities = model.predict_proba(X)
                    attack_index = list(model.classes_).index("ATTACK")
                    results["Attack Probability (%)"] = (
                        probabilities[:, attack_index] * 100
                    ).round(2)

                attack_count = int((predictions == "ATTACK").sum())
                benign_count = int((predictions == "BENIGN").sum())
                total_predictions = len(predictions)

                st.success(f"Analysis completed for {total_predictions:,} network record(s).")
                a, b, c = st.columns(3, gap="medium")
                with a:
                    metric_card("RECORDS ANALYSED", f"{total_predictions:,}", "Uploaded CSV rows", "metric-accent")
                with b:
                    metric_card("ATTACK PREDICTIONS", f"{attack_count:,}", "Classified as ATTACK", "metric-red")
                with c:
                    metric_card("BENIGN PREDICTIONS", f"{benign_count:,}", "Classified as BENIGN", "metric-green")

                st.write("")
                st.markdown("#### Detection results")
                st.dataframe(results.head(500), width="stretch", height=360)
                if len(results) > 500:
                    st.caption("Showing the first 500 rows in the preview. The download contains all rows.")

                csv_data = results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download complete detection results",
                    data=csv_data,
                    file_name="netinsight_detection_results.csv",
                    mime="text/csv",
                )
        except Exception as exc:
            st.error(f"Detection could not be completed: {exc}")


# --------------------------------------------------
# TRAFFIC ANALYSIS
# --------------------------------------------------
elif page == "Traffic Analysis":
    page_heading(
        "DATA EXPLORATION",
        "Network Traffic Analysis",
        "Explore the balance between benign traffic and attack records in the prepared dataset.",
    )
    st.write("")
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("#### Benign vs attack traffic")
        traffic_data = pd.DataFrame(
            {"Traffic type": ["Benign", "Attack"], "Records": [benign, attacks]}
        )
        fig = px.pie(
            traffic_data,
            names="Traffic type",
            values="Records",
            hole=0.55,
            color="Traffic type",
            color_discrete_map={"Benign": "#39d98a", "Attack": "#ff647c"},
        )
        fig.update_traces(textinfo="label+percent", textposition="outside")
        style_chart(fig, 390)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    with right:
        st.markdown("#### Dataset balance")
        balance = pd.DataFrame(
            {"Traffic type": ["Benign", "Attack"], "Records": [benign, attacks]}
        )
        fig = px.bar(
            balance,
            x="Traffic type",
            y="Records",
            color="Traffic type",
            color_discrete_map={"Benign": "#39d98a", "Attack": "#ff647c"},
            text="Records",
        )
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig.update_yaxes(title="Records", rangemode="tozero")
        fig.update_xaxes(title="")
        style_chart(fig, 390)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    st.divider()
    st.markdown("#### Attack category analysis")
    if "Label" in df.columns:
        attack_labels = df[df["Label"].astype(str).str.strip().str.upper() != "BENIGN"]["Label"]
        label_counts = attack_labels.value_counts().head(10)
        if not label_counts.empty:
            category_df = label_counts.rename_axis("Attack category").reset_index(name="Records")
            fig = px.bar(
                category_df.sort_values("Records"),
                x="Records",
                y="Attack category",
                orientation="h",
                text="Records",
                color_discrete_sequence=["#42d6e8"],
            )
            fig.update_traces(texttemplate="%{text:,}", textposition="outside")
            style_chart(fig, 430)
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            st.dataframe(category_df, width="stretch")
        else:
            st.info("No non-BENIGN attack labels are present in this prepared dataset.")
    else:
        st.info("The dataset does not contain the original Label column, so category analysis is unavailable.")


# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------
elif page == "Model Performance":
    page_heading(
        "MODEL DETAILS",
        "Machine Learning Performance",
        "Training configuration and measured evaluation results for the saved model.",
    )
    st.write("")
    p1, p2, p3 = st.columns(3, gap="medium")
    with p1:
        metric_card("TEST ACCURACY", "99.90%", "Reported held-out test split", "metric-green")
    with p2:
        metric_card("MODEL ALGORITHM", "Random Forest", "Supervised classification", "metric-accent")
    with p3:
        metric_card("INPUT FEATURES", f"{len(feature_names)}", "Numeric network traffic features", "metric-accent")

    st.write("")
    st.markdown("#### Evaluation summary")
    st.info(
        "The reported 99.90% accuracy was measured on the held-out test portion "
        "of the prepared CIC-IDS2017 dataset. It is not a measurement of live network traffic."
    )

    st.markdown("#### Confusion matrix")
    # Matrix from the actual training output, with class order ATTACK, BENIGN.
    cm = np.array([[9975, 10], [10, 9977]])
    cm_fig = px.imshow(
        cm,
        text_auto=True,
        x=["Predicted ATTACK", "Predicted BENIGN"],
        y=["Actual ATTACK", "Actual BENIGN"],
        color_continuous_scale=["#102238", "#167d85", "#42d6e8"],
        aspect="auto",
    )
    cm_fig.update_layout(coloraxis_showscale=False)
    style_chart(cm_fig, 360)
    st.plotly_chart(cm_fig, width="stretch", config={"displayModeBar": False})

    st.markdown("#### Model input features")
    st.caption(f"The model expects exactly {len(feature_names)} numeric input columns.")
    with st.expander("View all feature names"):
        st.write(feature_names)

    st.markdown("#### Dataset and method")
    details = pd.DataFrame(
        {
            "Item": [
                "Dataset",
                "Classifier",
                "Prepared records",
                "Target classes",
                "Training/test split",
                "Random Forest estimators",
            ],
            "Value": [
                "CIC-IDS2017",
                "Random Forest Classifier",
                f"{total:,}",
                "BENIGN and ATTACK",
                "80% training / 20% testing",
                "100",
            ],
        }
    )
    st.dataframe(details, width="stretch", hide_index=True)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown(
    """
    <div style="text-align:center; color:#71869f; font-size:11px; padding:28px 0 4px 0;">
        NETINSIGHT &nbsp;•&nbsp; CIC-IDS2017 RESEARCH PROJECT
        <br>AI-based network intrusion detection prototype
    </div>
    """,
    unsafe_allow_html=True,
)
