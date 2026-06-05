"""
app.py — Teammate Matcher (instructor-facing web app)
=====================================================
A Streamlit UI over the Teammate Matcher ML pipeline. Instructors upload a
Google Forms CSV of anonymous student survey responses (identified only by UNCC
student ID), pick a matching model and team size, and download balanced team
assignments.

Data handling: uploaded files are processed in memory for the session only.
Nothing is written to disk or logged. The student-ID column is used purely as an
output label — it never enters the feature matrix or influences clustering.

Run locally:  streamlit run app.py
"""

import base64
import io
import os
import zipfile

import altair as alt
import pandas as pd
import streamlit as st

from src import preprocess, models, evaluate

# ── Brand palette (intentionally NOT UNC Charlotte's trademarked colors) ───────
NAVY = "#1E3A5F"      # primary
BLUE = "#2563EB"      # accent / active
EMERALD = "#059669"   # positive
TEXT = "#0F172A"
MUTED = "#64748B"
BORDER = "#E2E8F0"
CARD = "#FFFFFF"
BG = "#F4F7FB"

# Distinct, accessible hues cycled across team cards / chart bars.
TEAM_PALETTE = [
    "#2563EB", "#059669", "#7C3AED", "#DB2777", "#EA580C", "#0891B2",
    "#CA8A04", "#DC2626", "#4F46E5", "#16A34A", "#9333EA", "#0D9488",
]

APP_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(APP_DIR, "assets", "logo.png")

# ── Configuration ─────────────────────────────────────────────────────────────
# Force-copy link: clicking it prompts the teacher to save their OWN copy of the
# form to their Drive (their own questions + responses), never touching the original.
FORM_URL = "https://docs.google.com/forms/d/1n3zI__UIbtY5OZGqRnxfEafs6Zjb2Weq3V-NF7qthiA/copy"

MODEL_CHOICES = {
    "Hungarian — even sizes + schedule fit (recommended)": ("hungarian", "compatibility"),
    "K-Means — schedule & work-style similarity": ("kmeans", "compatibility"),
    "Agglomerative — schedule & work-style similarity": ("agglomerative", "compatibility"),
    "Gaussian Mixture — complementary skills": ("gmm", "complementarity"),
}

# Metrics surfaced up front, in plain instructor language (order matters).
FRIENDLY_METRICS = ("schedule_overlap", "skill_coverage", "skill_variance")

# Clustering diagnostics — accurate but technical, tucked into a collapsed panel.
# key -> (label, direction arrow, one-line meaning)
TECHNICAL_METRICS = {
    "silhouette": ("Silhouette", "↑", "Cluster cohesion vs. separation (−1 to 1)."),
    "davies_bouldin": ("Davies–Bouldin", "↓", "How much clusters overlap (0 is best)."),
    "calinski_harabasz": ("Calinski–Harabasz", "↑", "Between- vs. within-team spread."),
}

# Browser favicon: prefer the generated mark, fall back to an emoji.
try:
    from PIL import Image

    _PAGE_ICON = Image.open(LOGO_PATH)
except Exception:  # noqa: BLE001
    _PAGE_ICON = "🧩"

st.set_page_config(page_title="Teammate Matcher", page_icon=_PAGE_ICON, layout="centered")


# ── Styling ────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');

/* Hide default Streamlit chrome for a cleaner product feel */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
[data-testid="stHeader"] { background: transparent; height: 0; }

.stApp { background: #F4F7FB; }
.block-container { padding-top: 1.6rem; max-width: 820px; }

html, body, [class*="css"], .stApp, button, input, textarea, select {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    color: #0F172A;
}
h1, h2, h3, h4, h5 { font-family: 'Plus Jakarta Sans', sans-serif !important; color: #1E3A5F; }

/* Hero */
.tm-hero { display: flex; align-items: center; gap: 16px; padding: 6px 0 2px; }
.tm-hero img { width: 56px; height: 56px; border-radius: 14px; box-shadow: 0 4px 14px rgba(30,58,95,.18); }
.tm-hero .tm-title { font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 800; font-size: 1.85rem; line-height: 1.1; color: #1E3A5F; margin: 0; }
.tm-hero .tm-tagline { color: #64748B; font-size: .95rem; margin: 3px 0 0; }
.tm-rule { height: 4px; width: 64px; background: linear-gradient(90deg,#1E3A5F,#2563EB); border-radius: 99px; margin: 14px 0 22px; }

/* Step indicator */
.tm-steps { display: flex; align-items: flex-start; margin: 0 0 26px; }
.tm-step { display: flex; flex-direction: column; align-items: center; gap: 7px; }
.tm-num { width: 34px; height: 34px; border-radius: 50%; display: flex; align-items: center;
          justify-content: center; font-weight: 600; font-size: .9rem; background: #E2E8F0; color: #64748B; }
.tm-lbl { font-size: .78rem; color: #64748B; font-weight: 500; }
.tm-step.active .tm-num { background: #2563EB; color: #fff; box-shadow: 0 0 0 4px rgba(37,99,235,.15); }
.tm-step.active .tm-lbl { color: #1E3A5F; font-weight: 600; }
.tm-step.done .tm-num { background: #1E3A5F; color: #fff; }
.tm-step.done .tm-lbl { color: #1E3A5F; }
.tm-line { flex: 1; height: 2px; background: #E2E8F0; margin: 17px 8px 0; border-radius: 2px; }
.tm-line.done { background: #1E3A5F; }

/* Banner */
.tm-banner { background: linear-gradient(90deg, rgba(30,58,95,.06), rgba(37,99,235,.06));
             border: 1px solid #E2E8F0; border-left: 4px solid #059669; border-radius: 12px;
             padding: 14px 18px; margin: 4px 0 18px; color: #0F172A; }

/* Metric grid */
.tm-metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px,1fr)); gap: 12px; margin: 6px 0; }
.tm-metric { background: #fff; border: 1px solid #E2E8F0; border-radius: 12px; padding: 14px 16px; }
.tm-metric-label { font-size: .72rem; text-transform: uppercase; letter-spacing: .04em; color: #64748B; font-weight: 600; }
.tm-metric-value { font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.5rem; font-weight: 700; color: #1E3A5F; margin: 4px 0 2px; }
.tm-metric-dir { font-size: .72rem; color: #64748B; }
.tm-metric-sub { font-size: .8rem; color: #475569; margin-top: 3px; line-height: 1.4; }
.tm-metric-scale { font-size: .7rem; color: #94A3B8; margin-top: 5px; }
.tm-section { font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700; color: #1E3A5F; font-size: 1.05rem; margin: 22px 0 8px; }

/* Team cards */
.tm-team-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px,1fr)); gap: 14px; margin: 8px 0 6px; }
.tm-team-card { background: #fff; border: 1px solid #E2E8F0; border-radius: 14px; padding: 14px 16px;
                box-shadow: 0 2px 8px rgba(15,23,42,.04); }
.tm-team-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }
.tm-team-badge { color: #fff; font-weight: 700; font-size: .8rem; padding: 3px 12px; border-radius: 99px;
                 font-family: 'Plus Jakarta Sans', sans-serif; }
.tm-team-count { font-size: .76rem; color: #64748B; }
.tm-team-members { font-size: .9rem; color: #0F172A; line-height: 1.55; word-break: break-word; }

/* Buttons */
.stButton > button, .stDownloadButton > button {
    border-radius: 10px; font-weight: 600; padding: .5rem 1.1rem; border: 1px solid #E2E8F0; transition: all .15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(30,58,95,.12); }
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] { background: #2563EB; border-color: #2563EB; }

/* File uploader */
[data-testid="stFileUploaderDropzone"] { border: 2px dashed #CBD5E1; border-radius: 14px; background: #fff; }

/* Footer */
.tm-footer { margin-top: 34px; padding-top: 16px; border-top: 1px solid #E2E8F0; color: #94A3B8; font-size: .78rem; text-align: center; line-height: 1.6; }
</style>
"""


@st.cache_data
def _logo_b64() -> str:
    try:
        with open(LOGO_PATH, "rb") as fh:
            return base64.b64encode(fh.read()).decode()
    except Exception:  # noqa: BLE001
        return ""


def render_hero():
    b64 = _logo_b64()
    img = f'<img src="data:image/png;base64,{b64}" alt="logo"/>' if b64 else ""
    st.markdown(
        f"""
        <div class="tm-hero">
            {img}
            <div>
                <p class="tm-title">Teammate Matcher</p>
                <p class="tm-tagline">Form balanced student teams from a quick survey — built for UNC Charlotte instructors.</p>
            </div>
        </div>
        <div class="tm-rule"></div>
        """,
        unsafe_allow_html=True,
    )


def render_steps(active: int):
    """active: 0=Upload, 1=Configure, 2=Results."""
    labels = ["Upload", "Configure", "Results"]
    parts = ['<div class="tm-steps">']
    for i, lbl in enumerate(labels):
        state = "done" if i < active else ("active" if i == active else "")
        num = "✓" if i < active else str(i + 1)
        parts.append(
            f'<div class="tm-step {state}"><div class="tm-num">{num}</div>'
            f'<div class="tm-lbl">{lbl}</div></div>'
        )
        if i < len(labels) - 1:
            parts.append(f'<div class="tm-line {"done" if i < active else ""}"></div>')
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def render_footer():
    st.markdown(
        """
        <div class="tm-footer">
            Teammate Matcher · A student data-science project, built for UNC Charlotte instructors.<br/>
            Not affiliated with, endorsed by, or an official tool of UNC Charlotte. Files are processed in memory only — nothing is stored.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _is_missing(v) -> bool:
    return v is None or (isinstance(v, float) and pd.isna(v))


def _friendly_card(key: str, value: float, n_skills: int):
    """Return (label, big value, plain-language line, scale hint) for a metric."""
    if key == "schedule_overlap":
        return ("Schedule overlap", f"{value * 100:.0f}%",
                "Shared free-time slots for the average pair.",
                "0% = no common time · 100% = identical availability")
    if key == "skill_coverage":
        return ("Skill coverage", f"{value:.1f} of {n_skills}",
                "Skill areas where each team has a capable member (rated 3+ of 5).",
                f"higher is better · {n_skills} of {n_skills} = every skill covered")
    if key == "skill_variance":
        # Raw value is a within-team std of 0–1 skills, so it maxes out at 0.5.
        # Rescale to 0–100% so the card carries its own range like the others.
        pct = max(0.0, min(100.0, value / 0.5 * 100))
        return ("Skill diversity", f"{pct:.0f}%",
                "How much teammates' skill levels vary within a team.",
                "0% = identical skill levels · 100% = very mixed")
    return (key, f"{value:.2f}", "", "")


def render_friendly_metrics(metrics: dict, n_skills: int):
    cards = ['<div class="tm-metric-grid">']
    for key in FRIENDLY_METRICS:
        v = metrics.get(key)
        if _is_missing(v):
            continue
        label, value, sub, scale = _friendly_card(key, v, n_skills)
        cards.append(
            f'<div class="tm-metric"><div class="tm-metric-label">{label}</div>'
            f'<div class="tm-metric-value">{value}</div>'
            f'<div class="tm-metric-sub">{sub}</div>'
            f'<div class="tm-metric-scale">{scale}</div></div>'
        )
    cards.append("</div>")
    st.markdown("".join(cards), unsafe_allow_html=True)


def render_technical_metrics(metrics: dict):
    cards = ['<div class="tm-metric-grid">']
    for key, (label, arrow, desc) in TECHNICAL_METRICS.items():
        v = metrics.get(key)
        if _is_missing(v):
            continue
        val = f"{v:.1f}" if key == "calinski_harabasz" else f"{v:.2f}"
        better = "higher is better" if arrow == "↑" else "lower is better"
        cards.append(
            f'<div class="tm-metric"><div class="tm-metric-label">{label}</div>'
            f'<div class="tm-metric-value">{val}</div>'
            f'<div class="tm-metric-sub">{desc}</div>'
            f'<div class="tm-metric-scale">{arrow} {better}</div></div>'
        )
    cards.append("</div>")
    st.markdown("".join(cards), unsafe_allow_html=True)


def render_team_cards(roster: pd.DataFrame):
    cards = ['<div class="tm-team-grid">']
    for team_no in sorted(roster["team"].unique()):
        members = roster.loc[roster["team"] == team_no, "student_id"].tolist()
        color = TEAM_PALETTE[(int(team_no) - 1) % len(TEAM_PALETTE)]
        member_str = ", ".join(str(m) for m in members)
        cards.append(
            f'<div class="tm-team-card" style="border-top:4px solid {color};">'
            f'<div class="tm-team-head">'
            f'<span class="tm-team-badge" style="background:{color};">Team {team_no}</span>'
            f'<span class="tm-team-count">{len(members)} students</span></div>'
            f'<div class="tm-team-members">{member_str}</div></div>'
        )
    cards.append("</div>")
    st.markdown("".join(cards), unsafe_allow_html=True)


def render_team_chart(roster: pd.DataFrame):
    chart_df = (
        roster["team"].value_counts().sort_index().rename_axis("team").reset_index(name="students")
    )
    chart_df["team"] = chart_df["team"].astype(str)
    domain = chart_df["team"].tolist()
    rng = [TEAM_PALETTE[(int(t) - 1) % len(TEAM_PALETTE)] for t in domain]
    chart = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5, size=38)
        .encode(
            x=alt.X("team:N", title="Team", sort=domain, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("students:Q", title="Students"),
            color=alt.Color("team:N", scale=alt.Scale(domain=domain, range=rng), legend=None),
            tooltip=[alt.Tooltip("team:N", title="Team"), alt.Tooltip("students:Q", title="Students")],
        )
        .properties(height=240)
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False, labelColor=MUTED, titleColor=MUTED)
    )
    st.altair_chart(chart, use_container_width=True)


# ── Session helpers ───────────────────────────────────────────────────────────
def _reset():
    for key in ("df_raw", "source_name", "result", "processed", "feature_sets", "ids"):
        st.session_state.pop(key, None)


def _run_model(model_key, feature_key, k, balance):
    feats = st.session_state.feature_sets
    X = feats[feature_key]
    if model_key == "hungarian":
        result = models.hungarian_teams(X, k=k)
    elif model_key == "kmeans":
        result = models.kmeans_teams(X, k=k)
    elif model_key == "agglomerative":
        result = models.agglomerative_teams(X, k=k)
    elif model_key == "gmm":
        result = models.gmm_teams(X, k=k)
    else:
        raise ValueError(f"Unknown model: {model_key}")

    # Guardrail: force every team to the chosen size (±1) unless the instructor
    # opted into natural sizes. Keeps the model's matching, fixes the size drift.
    if balance:
        result = models.balance_team_sizes(result, X)
    return result


def _build_zip(roster_df, metrics_df):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("team_assignments.csv", roster_df.to_csv(index=False))
        z.writestr("evaluation_metrics.csv", metrics_df.to_csv(index=False))
    buf.seek(0)
    return buf.getvalue()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
render_hero()

# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — Upload
# ════════════════════════════════════════════════════════════════════════════
if "df_raw" not in st.session_state:
    render_steps(0)
    st.subheader("Upload survey responses")

    st.markdown(
        f"""
1. **[Make your own copy of the survey]({FORM_URL})** — this saves a private
   copy to your Google Drive (your own form, your own responses).
2. Share *your* copy's link with your class and give them a deadline.
3. In your copy, go to **Responses → ⋮ → Download responses (.csv)**.
4. Upload that CSV below.

Students are identified only by their **UNCC student ID** — no names. You can
join the results back to your Canvas roster using that ID.
"""
    )

    st.info(
        "**Privacy.** Your file is processed in memory for this session only — "
        "nothing is saved or logged. Student IDs are used purely as labels and "
        "never affect matching. Please do **not** upload files containing names "
        "or emails.",
        icon="🔒",
    )

    uploaded = st.file_uploader("Google Forms responses (.csv)", type="csv")

    col1, col2 = st.columns([1, 2])
    with col1:
        use_example = st.button("Try with example data", use_container_width=True)

    if uploaded is not None:
        st.session_state.df_raw = pd.read_csv(uploaded)
        st.session_state.source_name = uploaded.name
        st.rerun()
    elif use_example:
        st.session_state.df_raw = pd.read_csv("assets/example_survey.csv")
        st.session_state.source_name = "example_survey.csv (synthetic demo data)"
        st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — Configure
# ════════════════════════════════════════════════════════════════════════════
elif "result" not in st.session_state:
    render_steps(1)
    df_raw = st.session_state.df_raw
    n = len(df_raw)

    st.subheader("Configure")
    st.success(f"Loaded **{n} responses** from `{st.session_state.source_name}`.")

    with st.expander("Preview first 5 rows"):
        st.dataframe(df_raw.head(), use_container_width=True)

    for msg in models.validate_n(n):
        st.warning(msg)

    preferred_size = st.slider(
        "Preferred team size", min_value=3, max_value=6, value=4,
        help="Number of teams is derived automatically from class size and this value.",
    )
    k = models.derive_k(n, preferred_size)

    model_label = st.selectbox("Matching model", list(MODEL_CHOICES.keys()), index=0)
    model_key, feature_key = MODEL_CHOICES[model_label]

    with st.expander("Advanced options"):
        allow_natural = st.checkbox(
            "Allow natural (uneven) team sizes",
            value=False,
            help="By default every team is the same size (±1 for the remainder). "
                 "Check this to let the clustering model set sizes — teams may "
                 "then range more widely.",
        )
    balance = not allow_natural

    # Show the team sizes the instructor will actually get, up front.
    if balance:
        caps = models.balanced_capacities(n, k)
        clo, chi = min(caps), max(caps)
        cap_desc = f"{clo}" if clo == chi else f"{clo}–{chi}"
        st.caption(f"→ **{k} teams** of {cap_desc} students each.")
    else:
        st.caption(f"→ About **{k} teams**; sizes are set by the model and may be uneven.")

    if model_key == "gmm":
        tail = ("Team sizes are still kept even." if balance
                else "With natural sizes on, GMM team sizes may vary widely.")
        st.caption(f"GMM groups students by **complementary skills**. {tail}")

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Generate teams", type="primary", use_container_width=True):
            try:
                processed, feature_sets, ids = preprocess.process_dataframe(df_raw)
                st.session_state.processed = processed
                st.session_state.feature_sets = feature_sets
                st.session_state.ids = ids
                st.session_state.result = _run_model(
                    model_key, feature_key, k, balance
                )
                st.session_state.feature_key = feature_key
                st.session_state.model_label = model_label
                st.rerun()
            except KeyError as e:
                st.error(
                    f"Could not find an expected survey column: {e}. "
                    "Make sure the CSV came from the official Google Form template "
                    "without renamed questions."
                )
            except Exception as e:  # noqa: BLE001 — surface any pipeline error gracefully
                st.error(f"Something went wrong while processing the file: {e}")
    with c2:
        if st.button("Start over", use_container_width=True):
            _reset()
            st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# STEP 3 — Results
# ════════════════════════════════════════════════════════════════════════════
else:
    render_steps(2)
    result = st.session_state.result
    processed = st.session_state.processed
    ids = st.session_state.ids
    feature_key = st.session_state.feature_key

    st.subheader("Team assignments")
    st.caption(f"Model: **{st.session_state.model_label}**")

    # Roster: student_id → team (1-indexed for humans)
    roster = pd.DataFrame({
        "student_id": ids.values,
        "team": result.labels + 1,
    }).sort_values(["team", "student_id"]).reset_index(drop=True)

    sizes = result.team_sizes()
    size_vals = sorted(sizes.values())
    lo, hi = size_vals[0], size_vals[-1]
    size_desc = f"{lo}" if lo == hi else f"{lo}–{hi}"
    size_word = "even" if (hi - lo) <= 1 else "uneven"

    # Metrics (computed first so the banner can surface schedule overlap)
    X = st.session_state.feature_sets[feature_key]
    metrics = evaluate.evaluate(X, processed, result)
    n_skills = len([c for c in evaluate.SKILL_COLS if c in processed.columns]) \
        or len(evaluate.SKILL_COLS)

    overlap = metrics.get("schedule_overlap")
    overlap_txt = ""
    if not _is_missing(overlap):
        overlap_txt = f" · <strong>{overlap * 100:.0f}%</strong> avg schedule overlap"
    st.markdown(
        f'<div class="tm-banner"><strong>{result.k} teams</strong> formed · '
        f'sizes {size_desc} ({size_word}){overlap_txt}</div>',
        unsafe_allow_html=True,
    )

    render_team_cards(roster)
    render_team_chart(roster)

    st.markdown('<div class="tm-section">What this means for your teams</div>',
                unsafe_allow_html=True)
    render_friendly_metrics(metrics, n_skills)

    with st.expander("Technical clustering metrics"):
        render_technical_metrics(metrics)
        st.caption(
            "Diagnostics from the clustering step — handy for comparing models, "
            "not required reading. Forcing even team sizes can lower these slightly."
        )

    # Downloads
    download_roster = roster.copy()
    download_metrics = pd.DataFrame([metrics]).assign(
        model=result.model_name, k=result.k
    )
    st.download_button(
        "⬇ Download results (.zip)",
        data=_build_zip(download_roster, download_metrics),
        file_name="teammate_matcher_results.zip",
        mime="application/zip",
        type="primary",
    )

    if st.button("Start over"):
        _reset()
        st.rerun()

render_footer()
