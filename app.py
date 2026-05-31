"""
app.py — Teammate Matcher (instructor-facing web app)
=====================================================
A thin Streamlit UI over the Teammate Matcher ML pipeline. Instructors upload a
Google Forms CSV of anonymous student survey responses (identified only by UNCC
student ID), pick a matching model and team size, and download balanced team
assignments.

Data handling: uploaded files are processed in memory for the session only.
Nothing is written to disk or logged. The student-ID column is used purely as an
output label — it never enters the feature matrix or influences clustering.

Run locally:  streamlit run app.py
"""

import io
import zipfile

import pandas as pd
import streamlit as st

from src import preprocess, models, evaluate

# ── Configuration ─────────────────────────────────────────────────────────────
# Force-copy link: clicking it prompts the teacher to save their OWN copy of the
# form to their Drive (their own questions + responses), never touching the original.
FORM_URL = "https://docs.google.com/forms/d/1n3zI__UIbtY5OZGqRnxfEafs6Zjb2Weq3V-NF7qthiA/copy"

MODEL_CHOICES = {
    "Hungarian — balanced team sizes (recommended)": ("hungarian", "compatibility"),
    "K-Means — schedule/work-style similarity": ("kmeans", "compatibility"),
    "Agglomerative (Ward) — similarity": ("agglomerative", "compatibility"),
    "Gaussian Mixture — skill complementarity": ("gmm", "complementarity"),
}

METRIC_LABELS = {
    "silhouette": "Silhouette ↑",
    "davies_bouldin": "Davies-Bouldin ↓",
    "calinski_harabasz": "Calinski-Harabasz ↑",
    "skill_variance": "Intra-team skill variance ↕",
    "schedule_overlap": "Schedule overlap ↑",
    "skill_coverage": "Skill coverage ↑",
}

st.set_page_config(page_title="Teammate Matcher", page_icon="🧩", layout="centered")


# ── Session helpers ───────────────────────────────────────────────────────────
def _reset():
    for key in ("df_raw", "source_name", "result", "processed", "feature_sets", "ids"):
        st.session_state.pop(key, None)


def _run_model(model_key, feature_key, k, preferred_size):
    feats = st.session_state.feature_sets
    X = feats[feature_key]
    if model_key == "hungarian":
        # team_size is derived internally as N // k (tested behavior); k drives size.
        return models.hungarian_teams(X, k=k)
    if model_key == "kmeans":
        return models.kmeans_teams(X, k=k)
    if model_key == "agglomerative":
        return models.agglomerative_teams(X, k=k)
    if model_key == "gmm":
        return models.gmm_teams(X, k=k)
    raise ValueError(f"Unknown model: {model_key}")


def _build_zip(roster_df, metrics_df):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("team_assignments.csv", roster_df.to_csv(index=False))
        z.writestr("evaluation_metrics.csv", metrics_df.to_csv(index=False))
    buf.seek(0)
    return buf.getvalue()


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🧩 Teammate Matcher")
st.caption(
    "Form balanced student teams from anonymous survey responses. "
    "Built for UNC Charlotte instructors."
)

# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — Upload
# ════════════════════════════════════════════════════════════════════════════
if "df_raw" not in st.session_state:
    st.subheader("Step 1 — Upload survey responses")

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
    df_raw = st.session_state.df_raw
    n = len(df_raw)

    st.subheader("Step 2 — Configure")
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
    st.caption(f"→ This will produce **{k} teams** of about {preferred_size} students each.")

    model_label = st.selectbox("Matching model", list(MODEL_CHOICES.keys()), index=0)
    model_key, feature_key = MODEL_CHOICES[model_label]

    if model_key == "gmm":
        st.caption(
            "GMM groups by **complementary skills** (so team sizes may vary). "
            "The other models optimize for **schedule/work-style compatibility**."
        )

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Generate teams", type="primary", use_container_width=True):
            try:
                processed, feature_sets, ids = preprocess.process_dataframe(df_raw)
                st.session_state.processed = processed
                st.session_state.feature_sets = feature_sets
                st.session_state.ids = ids
                st.session_state.result = _run_model(
                    model_key, feature_key, k, preferred_size
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
    result = st.session_state.result
    processed = st.session_state.processed
    ids = st.session_state.ids
    feature_key = st.session_state.feature_key

    st.subheader("Step 3 — Team assignments")
    st.caption(f"Model: **{st.session_state.model_label}**")

    # Roster: student_id → team (1-indexed for humans)
    roster = pd.DataFrame({
        "student_id": ids.values,
        "team": result.labels + 1,
    }).sort_values(["team", "student_id"]).reset_index(drop=True)

    sizes = result.team_sizes()
    st.write(
        f"**{result.k} teams** · sizes: "
        + ", ".join(f"{v}" for v in sorted(sizes.values(), reverse=True))
    )

    # Per-team rosters
    for team_no in sorted(roster["team"].unique()):
        members = roster.loc[roster["team"] == team_no, "student_id"].tolist()
        with st.expander(f"Team {team_no}  ({len(members)} students)", expanded=True):
            st.write(", ".join(str(m) for m in members))

    st.bar_chart(roster["team"].value_counts().sort_index(), x_label="Team", y_label="Students")

    # Metrics
    X = st.session_state.feature_sets[feature_key]
    metrics = evaluate.evaluate(X, processed, result)
    metrics_df = pd.DataFrame(
        [{METRIC_LABELS.get(k_, k_): v for k_, v in metrics.items()}]
    )
    with st.expander("Quality metrics"):
        st.dataframe(metrics_df.T.rename(columns={0: "value"}), use_container_width=True)
        st.caption("↑ higher is better · ↓ lower is better · ↕ depends on objective")

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
