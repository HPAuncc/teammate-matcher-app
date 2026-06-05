"""
src/preprocess.py
-----------------
Full preprocessing pipeline for the Teamora survey data.
Input:  data/raw_survey_responses.csv  (Google Forms export)
Output: data/processed_survey_data.csv

Pipeline steps:
  1. Column cleaning & renaming
  2. Course code normalization → drop
  3. Timestamp drop
  4. Availability encoding (days + time slots → binary)
  5. Ordinal encoding
  6. One-hot encoding
  7. Contribution multi-select → binary columns
  8. Missing value handling (GPA)
  9. Min-Max normalization
  10. Feature set construction
"""

import re

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ── Skill schema (data-driven) ────────────────────────────────────────────────
# Skills are discovered by the ``skill_`` column prefix rather than a fixed list,
# so Teamora works for any course's skill set (or none at all). Display labels are
# derived from the slug, with overrides for acronyms.
SKILL_PREFIX = "skill_"
SKILL_LABEL_OVERRIDES = {"skill_ml": "ML"}


def skill_columns(df):
    """Skill feature columns present in ``df`` (anything with the ``skill_`` prefix)."""
    return [c for c in df.columns if c.startswith(SKILL_PREFIX)]


def skill_label(slug):
    """Human label for a skill slug, e.g. 'skill_data_analysis' -> 'Data analysis'."""
    if slug in SKILL_LABEL_OVERRIDES:
        return SKILL_LABEL_OVERRIDES[slug]
    if slug.startswith(SKILL_PREFIX):
        return slug[len(SKILL_PREFIX):].replace("_", " ").strip().capitalize()
    return slug


# Skills arrive as one Google Forms "grid" question: each row exports as a column
# "<grid title> [<skill name>]". We detect those by the bracketed suffix (the only
# grid in the template) and turn each into a skill_<slug> column, preserving the
# instructor's exact row label for display.
_SKILL_GRID_RE = re.compile(r"\[(.+?)\]\s*$")


def _slugify(label):
    slug = re.sub(r"[^0-9a-z]+", "_", str(label).strip().lower()).strip("_")
    return slug or "skill"


def _grid_skill_map(columns):
    """From raw column names, return (rename: raw->slug, labels: slug->row label)."""
    rename, labels = {}, {}
    for col in columns:
        s = str(col)
        if s.startswith(SKILL_PREFIX):
            continue
        m = _SKILL_GRID_RE.search(s)
        if m:
            label = m.group(1).strip()
            slug = SKILL_PREFIX + _slugify(label)
            rename[col] = slug
            labels[slug] = label
    return rename, labels


# ── Raw column name → clean internal name ─────────────────────────────────────
COLUMN_MAP = {
    "Timestamp"                                                        : "timestamp",
    "What is your UNCC student ID?"                                     : "student_id",
    "What class is this team assignment for?"                          : "course_code",
    "What is your year in school?"                                     : "year",
    "Which days are you generally available to meet with your team?"   : "_days_raw",
    "What time(s) of day are you available?"                           : "_times_raw",
    "How many hours per week can you realistically dedicate to group project work?": "weekly_hours",
    "Do you prefer to meet in person or remotely?"                     : "meeting_mode",
    # Skills are NOT mapped here — they come from one editable grid question and
    # are detected dynamically (see _grid_skill_map / clean).
    "What role do you naturally gravitate toward in a group?"          : "role_pref",
    "How do you typically approach deadlines?"                         : "deadline_style",
    "How do you prefer to communicate with your team? (pick primary)"  : "comm_pref",
    "How often do you prefer to check in with teammates?"              : "checkin_freq",
    "When working on a group project, I prefer to..."                  : "collab_style",
    "I tend to focus on..."                                            : "detail_orientation",
    "My approach to conflict in a team is..."                          : "conflict_style",
    "What is your GPA range? (Optional)"                               : "gpa_band",
    "What do you most contribute to a team? (Select up to 2)"         : "_contrib_raw",
    "What is your biggest challenge in group projects?"                : "pain_point",
}

# ── Ordinal maps ──────────────────────────────────────────────────────────────
YEAR_MAP = {
    "Freshman" : 1, "Sophomore": 2, "Junior": 3, "Senior": 4, "Graduate": 5,
}

HOURS_MAP = {
    "Less than 3 hours": 1,
    "3–5 hours"        : 2,
    "6–9 hours"        : 3,
    "10+ hours"        : 4,
}

ROLE_MAP = {
    "I prefer to follow clear instructions"   : 1,
    "I prefer to contribute as a specialist"  : 2,
    "I'm comfortable in either role"          : 3,
    "I prefer to lead and coordinate"         : 4,
}

DEADLINE_MAP = {
    "I work best under pressure, closer to the deadline": 1,
    "I work steadily throughout"                        : 2,
    "I finish well before the deadline"                 : 3,
}

CHECKIN_MAP = {
    "Only when necessary" : 1,
    "Once a week"         : 2,
    "A few times a week"  : 3,
    "Daily"               : 4,
}

COLLAB_MAP = {
    "Work independently and combine at the end"    : 1,
    "A mix — divide tasks but check in regularly"  : 2,
    "Collaborate closely throughout"               : 3,
}

GPA_MAP = {
    "Below 2.5" : 1,
    "2.5 – 3.0" : 2,
    "3.0 – 3.5" : 3,
    "3.5 – 4.0" : 4,
}

# ── Checkbox expansion helpers ────────────────────────────────────────────────
DAYS = {
    "avail_mon": "Monday",
    "avail_tue": "Tuesday",
    "avail_wed": "Wednesday",
    "avail_thu": "Thursday",
    "avail_fri": "Friday",
    "avail_sat": "Saturday",
    "avail_sun": "Sunday",
}

TIMES = {
    "avail_morning"  : "Morning",
    "avail_afternoon": "Afternoon",
    "avail_evening"  : "Evening",
    "avail_latenight": "Late night",
}

CONTRIBS = {
    "contrib_technical"   : "Technical execution",
    "contrib_creative"    : "Creative ideas",
    "contrib_organization": "Organization and planning",
    "contrib_writing"     : "Research and writing",
    "contrib_morale"      : "Keeping team morale up",
    "contrib_qa"          : "Quality checking / editing",
}

MEETING_DUMMIES  = ["meeting_inperson", "meeting_remote", "meeting_nopref"]
COMM_DUMMIES     = ["comm_text", "comm_email", "comm_discord", "comm_video", "comm_inperson"]
CONFLICT_DUMMIES = ["conflict_direct", "conflict_private", "conflict_natural", "conflict_defer"]
PAIN_DUMMIES     = ["pain_schedule", "pain_workload", "pain_conflict",
                    "pain_communication", "pain_motivation"]


def _expand_checkbox(series, col_map):
    """Turn a comma-separated multi-select column into binary indicator columns."""
    result = {}
    for col_name, keyword in col_map.items():
        result[col_name] = series.fillna("").apply(
            lambda x: int(keyword.lower() in x.lower())
        )
    return pd.DataFrame(result)


def _onehot(series, mapping):
    """Turn a single-select categorical into one-hot columns."""
    dummies = pd.get_dummies(series, prefix=None)
    return dummies


def load_raw(path="data/raw_survey_responses.csv"):
    """Load the raw Google Forms CSV."""
    df = pd.read_csv(path)
    return df


def clean(df):
    """
    Step 1 — Column renaming, artifact removal, course code normalization.
    Returns a working copy with clean column names.
    """
    df = df.copy()

    # Rename known columns; silently drop unknown extra columns (e.g., Column 25)
    df = df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})

    # Skills: one grid question → one skill_<slug> column per row label.
    grid_rename, _ = _grid_skill_map(df.columns)
    df = df.rename(columns=grid_rename)

    # Drop artifact columns from Google Forms
    drop_cols = [c for c in df.columns if c.startswith("Column") or c == "timestamp"]
    df = df.drop(columns=drop_cols, errors="ignore")

    # course_code is kept as-is if present (label only, never used in clustering).
    # The deployed app serves any course, so we do not normalize it to a fixed value.

    return df


def encode_availability(df):
    """Step 2 — Expand day + time checkboxes into binary columns."""
    df = df.copy()
    day_df  = _expand_checkbox(df["_days_raw"],  DAYS)
    time_df = _expand_checkbox(df["_times_raw"], TIMES)
    df = pd.concat([df, day_df, time_df], axis=1)
    df = df.drop(columns=["_days_raw", "_times_raw"])
    return df


def encode_ordinals(df):
    """Step 3 — Map ordered categoricals to integers."""
    df = df.copy()
    df["year"]         = df["year"].map(YEAR_MAP)
    df["weekly_hours"] = df["weekly_hours"].map(HOURS_MAP)
    df["role_pref"]    = df["role_pref"].map(ROLE_MAP)
    df["deadline_style"] = df["deadline_style"].map(DEADLINE_MAP)
    df["checkin_freq"] = df["checkin_freq"].map(CHECKIN_MAP)
    df["collab_style"] = df["collab_style"].map(COLLAB_MAP)

    # GPA: "Prefer not to say" → NaN
    df["gpa_band"] = df["gpa_band"].replace("Prefer not to say", np.nan)
    df["gpa_band"] = df["gpa_band"].map(GPA_MAP)

    return df


def encode_onehot(df):
    """Step 4 — One-hot encode nominal variables."""
    df = df.copy()

    # Meeting mode
    meeting_map = {
        "In person only": "meeting_inperson",
        "Remote only"   : "meeting_remote",
        "No preference" : "meeting_nopref",
    }
    for col, key in meeting_map.items():
        df[key] = (df["meeting_mode"] == col).astype(int)
    df = df.drop(columns=["meeting_mode"])

    # Communication preference
    comm_map = {
        "Text/iMessage"        : "comm_text",
        "Email"                : "comm_email",
        "Discord/Slack"        : "comm_discord",
        "Video call (Zoom, Teams)": "comm_video",
        "In person"            : "comm_inperson",
    }
    for col, key in comm_map.items():
        df[key] = (df["comm_pref"] == col).astype(int)
    df = df.drop(columns=["comm_pref"])

    # Conflict style
    conflict_map = {
        "Address it directly and immediately": "conflict_direct",
        "Talk it out privately first"        : "conflict_private",
        "Let it resolve naturally"           : "conflict_natural",
        "Defer to whoever is leading"        : "conflict_defer",
    }
    for col, key in conflict_map.items():
        df[key] = (df["conflict_style"] == col).astype(int)
    df = df.drop(columns=["conflict_style"])

    # Pain point (kept for ethics analysis — not used in clustering)
    pain_map = {
        "Coordinating schedules"          : "pain_schedule",
        "Unequal workload distribution"   : "pain_workload",
        "Disagreements on direction"      : "pain_conflict",
        "Communication breakdowns"        : "pain_communication",
        "Staying motivated"               : "pain_motivation",
    }
    for col, key in pain_map.items():
        df[key] = (df["pain_point"] == col).astype(int)
    df = df.drop(columns=["pain_point"])

    return df


def encode_contributions(df):
    """Step 5 — Expand multi-select contributions into binary columns."""
    df = df.copy()
    contrib_df = _expand_checkbox(df["_contrib_raw"], CONTRIBS)
    df = pd.concat([df, contrib_df], axis=1)
    df = df.drop(columns=["_contrib_raw"])
    return df


def handle_missing(df):
    """
    Step 6 — Impute missing values.
    GPA: imputed with median ordinal band.
    Any row with >20% features missing: dropped.
    """
    df = df.copy()

    # GPA median imputation
    gpa_median = df["gpa_band"].median()
    missing_gpa = df["gpa_band"].isna().sum()
    df["gpa_band"] = df["gpa_band"].fillna(gpa_median)
    if missing_gpa > 0:
        print(f"  [impute] gpa_band: {missing_gpa} missing → filled with median ({gpa_median})")

    # Drop rows with >20% missing across numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    threshold = 0.20
    missing_pct = df[numeric_cols].isnull().mean(axis=1)
    dropped = (missing_pct > threshold).sum()
    df = df[missing_pct <= threshold].reset_index(drop=True)
    if dropped > 0:
        print(f"  [drop] {dropped} rows with >{threshold*100:.0f}% missing values removed")

    return df


def normalize(df, exclude=None):
    """
    Step 7 — Min-Max normalize all numeric features to [0, 1].
    exclude: list of columns to skip (e.g., already-binary columns).
    """
    df = df.copy()
    if exclude is None:
        exclude = []

    # Skills use a FIXED 1–5 → [0,1] mapping ((x-1)/4), not data-relative scaling,
    # so thresholds like "capable ≥ 3 of 5" mean the same in every class.
    skill_cols = skill_columns(df)
    for c in skill_cols:
        df[c] = ((pd.to_numeric(df[c], errors="coerce") - 1) / 4).clip(0, 1)

    # Identify columns that are already binary (only 0/1)
    binary_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if df[c].dropna().isin([0, 1]).all()
    ]
    skip = set(exclude) | set(binary_cols) | set(skill_cols)

    scale_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in skip
    ]

    scaler = MinMaxScaler()
    if scale_cols:
        df[scale_cols] = scaler.fit_transform(df[scale_cols])

    print(f"  [normalize] {len(scale_cols)} cols MinMax-scaled + "
          f"{len(skill_cols)} skill cols fixed 1–5")
    return df


def build_feature_sets(df):
    """
    Step 8 — Construct the two feature sets used by different model families.

    compatibility_features:
        Schedule overlap + work style → used for similarity-based clustering
        (K-Means, Agglomerative, Hungarian Algorithm)

    complementarity_features:
        Technical skill dimensions only → used for diversity-based clustering (GMM)
    """
    availability_cols = (
        [f"avail_{d}" for d in ["mon","tue","wed","thu","fri","sat","sun"]]
        + [f"avail_{t}" for t in ["morning","afternoon","evening","latenight"]]
    )
    workstyle_cols = [
        "weekly_hours", "role_pref", "deadline_style", "checkin_freq",
        "collab_style", "detail_orientation",
        "meeting_inperson", "meeting_remote", "meeting_nopref",
        "comm_text", "comm_email", "comm_discord", "comm_video", "comm_inperson",
        "conflict_direct", "conflict_private", "conflict_natural", "conflict_defer",
    ]
    skill_cols = skill_columns(df)

    # Only keep columns that actually exist in df
    compat_cols = [c for c in availability_cols + workstyle_cols if c in df.columns]
    compl_cols  = [c for c in skill_cols if c in df.columns]

    return {
        "compatibility"    : df[compat_cols].copy(),
        "complementarity"  : df[compl_cols].copy(),
        "all_features"     : df[[c for c in compat_cols + compl_cols if c in df.columns]].copy(),
    }


# Fixed (non-skill) questions the pipeline needs. Skills are optional.
REQUIRED_COLS = [
    "student_id", "_days_raw", "_times_raw", "year", "weekly_hours",
    "meeting_mode", "role_pref", "deadline_style", "comm_pref", "checkin_freq",
    "collab_style", "conflict_style", "gpa_band", "_contrib_raw", "pain_point",
]


def validate_raw(df_raw):
    """
    Return a list of human-readable problems with an uploaded CSV (empty = OK).

    Checks that the fixed template questions are present. Skills are optional —
    a file with no skill grid simply runs in schedule-only mode.
    """
    try:
        cleaned = clean(df_raw)
    except Exception as e:  # noqa: BLE001
        return [f"The file could not be read ({e})."]
    rev = {v: k for k, v in COLUMN_MAP.items()}
    missing = [rev.get(c, c) for c in REQUIRED_COLS if c not in cleaned.columns]
    problems = []
    if missing:
        problems.append(
            "These expected questions are missing or were renamed: "
            + "; ".join(f'"{m}"' for m in missing)
        )
    return problems


def process_dataframe(df_raw, id_col="student_id"):
    """
    In-memory preprocessing for the web app.

    Runs the same encoding/normalization pipeline as run_pipeline(), but on a
    DataFrame already in memory (an uploaded Google Forms CSV) — no disk reads,
    no disk writes, no row shuffling.

    The identifier column (default 'student_id') is preserved as a *label only*:
    it is excluded from Min-Max scaling and never enters any feature set, so it
    cannot influence clustering. It is returned separately so the app can map
    team assignments back to real students.

    Parameters
    ----------
    df_raw : pd.DataFrame — raw Google Forms export
    id_col : name of the identifier column after renaming (default 'student_id')

    Returns
    -------
    (processed_df, feature_sets, ids)
        processed_df : fully encoded + normalized DataFrame (index reset)
        feature_sets : dict from build_feature_sets()
        ids          : pd.Series of identifiers aligned row-for-row with the
                       feature sets and with any model's labels
    """
    df = clean(df_raw)
    df = encode_availability(df)
    df = encode_ordinals(df)
    df = encode_onehot(df)
    df = encode_contributions(df)
    df = handle_missing(df)              # may drop rows; resets index
    df = normalize(df, exclude=[id_col]) # keep the identifier out of scaling
    df = df.drop(columns=["course_code"], errors="ignore")
    df = df.reset_index(drop=True)

    if id_col in df.columns:
        ids = df[id_col].reset_index(drop=True)
    else:
        # No identifier column found — fall back to row labels so the app still works
        ids = pd.Series([f"row_{i + 1}" for i in range(len(df))], name=id_col)

    # Carry the instructor's own skill labels (from the grid row names) so the UI
    # can show them verbatim. Stored on the final df so it survives to the app.
    _, skill_labels = _grid_skill_map(df_raw.columns)
    df.attrs["skill_labels"] = skill_labels

    feature_sets = build_feature_sets(df)
    return df, feature_sets, ids


def run_pipeline(raw_path="data/raw_survey_responses.csv",
                 out_path="data/processed_survey_data.csv"):
    """
    Full end-to-end preprocessing pipeline.
    Returns (processed_df, feature_sets_dict).
    """
    print("=" * 55)
    print("Teamora — Preprocessing Pipeline")
    print("=" * 55)

    print("\n[1] Loading raw data...")
    df = load_raw(raw_path)
    print(f"    {df.shape[0]} responses, {df.shape[1]} columns")

    print("\n[2] Cleaning & renaming columns...")
    df = clean(df)

    print("\n[3] Encoding availability (days + time slots)...")
    df = encode_availability(df)

    print("\n[4] Encoding ordinal variables...")
    df = encode_ordinals(df)

    print("\n[5] One-hot encoding nominal variables...")
    df = encode_onehot(df)

    print("\n[6] Encoding contribution multi-select...")
    df = encode_contributions(df)

    print("\n[7] Handling missing values...")
    df = handle_missing(df)

    print("\n[8] Normalizing numeric features...")
    df = normalize(df)

    print(f"\n[9] Saving processed data → {out_path}")
    # Drop non-feature metadata before saving
    save_df = df.drop(columns=["course_code"], errors="ignore")

    # Shuffle rows before saving.
    # The raw CSV is ordered by submission timestamp, which is a quasi-identifier
    # in a small class (31 students who know each other could correlate row position
    # to submission time). Shuffling with a fixed seed breaks this link while
    # keeping the pipeline fully reproducible.
    save_df = save_df.sample(frac=1, random_state=42).reset_index(drop=True)

    save_df.to_csv(out_path, index=False)
    print(f"    {save_df.shape[0]} rows × {save_df.shape[1]} features saved")
    print(f"    (rows shuffled — submission order not preserved)")

    print("\n[10] Building feature sets...")
    feature_sets = build_feature_sets(save_df)
    for name, fset in feature_sets.items():
        print(f"    {name:20s}: {fset.shape[1]} features")

    print("\nPipeline complete.")
    print("=" * 55)

    return save_df, feature_sets


if __name__ == "__main__":
    df, fsets = run_pipeline()
