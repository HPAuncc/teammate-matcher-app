"""
src/evaluate.py
---------------
Evaluation metrics for all four Teammate Matcher models.

Metrics computed:
  1. Silhouette Score           — cluster cohesion vs. separation (higher = better)
  2. Davies-Bouldin Index       — avg similarity of each cluster to its closest (lower = better)
  3. Calinski-Harabasz Index    — between- vs. within-cluster dispersion (higher = better)
  4. Intra-team Skill Variance  — std dev of skill ratings within teams (context-dependent)
  5. Schedule Overlap Score     — mean Jaccard similarity of availability vectors (higher = better)
  6. Skill Coverage Score       — skill dimensions where ≥1 member scores ≥ threshold (higher = better)

All metrics accept a TeamingResult and the processed DataFrame.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

from src.models import TeamingResult


# ── Availability columns ──────────────────────────────────────────────────────
AVAIL_COLS = (
    ["avail_mon", "avail_tue", "avail_wed", "avail_thu",
     "avail_fri", "avail_sat", "avail_sun"]
    + ["avail_morning", "avail_afternoon", "avail_evening", "avail_latenight"]
)

SKILL_COLS = [
    "skill_python", "skill_data_analysis", "skill_statistics",
    "skill_visualization", "skill_ml", "skill_writing",
    "skill_research", "skill_presenting",
]


# ── Individual metrics ────────────────────────────────────────────────────────

def silhouette(X, result: TeamingResult) -> float:
    """
    Silhouette Score: measures how similar each point is to its own cluster
    compared to other clusters. Range [-1, 1]; higher = better separation.
    Requires ≥ 2 clusters with ≥ 1 member each.
    """
    X_arr = np.asarray(X, dtype=float)
    labels = result.labels
    if len(np.unique(labels)) < 2:
        return np.nan
    return float(silhouette_score(X_arr, labels))


def davies_bouldin(X, result: TeamingResult) -> float:
    """
    Davies-Bouldin Index: average ratio of within-cluster scatter to
    between-cluster separation. Range [0, ∞); lower = better.
    """
    X_arr = np.asarray(X, dtype=float)
    labels = result.labels
    if len(np.unique(labels)) < 2:
        return np.nan
    return float(davies_bouldin_score(X_arr, labels))


def calinski_harabasz(X, result: TeamingResult) -> float:
    """
    Calinski-Harabasz Index (Variance Ratio Criterion):
    ratio of between-cluster dispersion to within-cluster dispersion.
    Range [0, ∞); higher = better-defined clusters.
    """
    X_arr = np.asarray(X, dtype=float)
    labels = result.labels
    if len(np.unique(labels)) < 2:
        return np.nan
    return float(calinski_harabasz_score(X_arr, labels))


def intra_team_skill_variance(df: pd.DataFrame, result: TeamingResult,
                               skill_cols: Optional[List[str]] = None) -> float:
    """
    Mean standard deviation of skill ratings within each team.

    Low variance → homogeneous (each team similar in skills).
    High variance → diverse (complementary skills within team).

    Interpretation depends on objective:
      - Complementarity models (GMM): high variance preferred.
      - Similarity models: lower variance (on non-skill features) preferred.

    Returns the mean across all teams and all skill dimensions.
    """
    if skill_cols is None:
        skill_cols = [c for c in SKILL_COLS if c in df.columns]
    if not skill_cols:
        return np.nan

    variances = []
    for team_idx in range(result.k):
        mask = result.labels == team_idx
        if mask.sum() < 2:
            continue
        team_skills = df.loc[mask, skill_cols].values
        variances.append(np.std(team_skills, axis=0).mean())

    return float(np.mean(variances)) if variances else np.nan


def schedule_overlap_score(df: pd.DataFrame, result: TeamingResult,
                            avail_cols: Optional[List[str]] = None) -> float:
    """
    Mean Jaccard similarity of availability vectors within each team.

    For two binary availability vectors A and B:
        Jaccard(A, B) = |A ∩ B| / |A ∪ B|

    Range [0, 1]; higher = more scheduling overlap within teams.
    Averaged over all within-team pairs across all teams.
    """
    if avail_cols is None:
        avail_cols = [c for c in AVAIL_COLS if c in df.columns]
    if not avail_cols:
        return np.nan

    avail = df[avail_cols].values.astype(float)
    all_similarities = []

    for team_idx in range(result.k):
        row_indices = np.where(result.labels == team_idx)[0]
        if len(row_indices) < 2:
            continue
        team_avail = avail[row_indices]

        for i in range(len(row_indices)):
            for j in range(i + 1, len(row_indices)):
                a, b = team_avail[i], team_avail[j]
                intersection = np.sum(a * b)
                union = np.sum(np.clip(a + b, 0, 1))
                if union > 0:
                    all_similarities.append(intersection / union)

    return float(np.mean(all_similarities)) if all_similarities else np.nan


def skill_coverage_score(df: pd.DataFrame, result: TeamingResult,
                          skill_cols: Optional[List[str]] = None,
                          threshold: float = 0.5) -> float:
    """
    Skill Coverage Score: for each team, count the number of skill dimensions
    where at least one member scores ≥ threshold (in [0,1] normalized space).

    threshold=0.5 corresponds to a raw rating of ≥ 3 on the 1–5 scale
    after Min-Max normalization: (3-1)/(5-1) = 0.5.

    Range [0, len(skill_cols)]; higher = broader skill coverage per team.
    Returns the mean across all teams.
    """
    if skill_cols is None:
        skill_cols = [c for c in SKILL_COLS if c in df.columns]
    if not skill_cols:
        return np.nan

    coverages = []
    for team_idx in range(result.k):
        mask = result.labels == team_idx
        if mask.sum() == 0:
            continue
        team_skills = df.loc[mask, skill_cols].values
        # Count skills where at least one member meets threshold
        coverage = np.sum(team_skills.max(axis=0) >= threshold)
        coverages.append(int(coverage))

    return float(np.mean(coverages)) if coverages else np.nan


# ── Full evaluation report ────────────────────────────────────────────────────

def evaluate(X, df: pd.DataFrame, result: TeamingResult,
             skill_cols: Optional[List[str]] = None,
             avail_cols: Optional[List[str]] = None,
             coverage_threshold: float = 0.5) -> Dict[str, float]:
    """
    Run all six evaluation metrics for a single TeamingResult.

    Parameters
    ----------
    X                  : feature array used for clustering (for silhouette, DB, CH)
    df                 : full processed DataFrame (for domain metrics)
    result             : TeamingResult from any model
    skill_cols         : skill columns to use (default: SKILL_COLS)
    avail_cols         : availability columns (default: AVAIL_COLS)
    coverage_threshold : min normalized score to count as "covered" (default 0.5 → raw ≥ 3)

    Returns
    -------
    Dict of metric_name → float value
    """
    metrics = {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        metrics["silhouette"]           = silhouette(X, result)
        metrics["davies_bouldin"]       = davies_bouldin(X, result)
        metrics["calinski_harabasz"]    = calinski_harabasz(X, result)
        metrics["skill_variance"]       = intra_team_skill_variance(
                                              df, result, skill_cols)
        metrics["schedule_overlap"]     = schedule_overlap_score(
                                              df, result, avail_cols)
        metrics["skill_coverage"]       = skill_coverage_score(
                                              df, result, skill_cols, coverage_threshold)

    return metrics


def evaluate_all(X_compat, X_compl, df: pd.DataFrame,
                 results: Dict[str, TeamingResult],
                 coverage_threshold: float = 0.5) -> pd.DataFrame:
    """
    Evaluate all four models and return a single comparison DataFrame.

    Parameters
    ----------
    X_compat   : compatibility feature array (for K-Means, Agglomerative, Hungarian)
    X_compl    : complementarity feature array (for GMM)
    df         : full processed DataFrame
    results    : {'kmeans': result, 'agglomerative': result, 'hungarian': result, 'gmm': result}
    coverage_threshold: passed to skill_coverage_score

    Returns
    -------
    DataFrame with models as rows and metrics as columns.
    Better-direction noted in column comments:
        silhouette       ↑ higher better
        davies_bouldin   ↓ lower better
        calinski_harabasz↑ higher better
        skill_variance   ↕ context-dependent
        schedule_overlap ↑ higher better
        skill_coverage   ↑ higher better
    """
    # Which feature set each model uses for algorithmic metrics
    X_map = {
        "kmeans"       : X_compat,
        "agglomerative": X_compat,
        "hungarian"    : X_compat,
        "gmm"          : X_compl,
    }

    rows = []
    for name, result in results.items():
        X = X_map.get(name, X_compat)
        m = evaluate(X, df, result, coverage_threshold=coverage_threshold)
        m["model"]     = result.model_name
        m["k"]         = result.k
        m["team_sizes"]= str(result.team_sizes())
        rows.append(m)

    cols = ["model", "k", "team_sizes",
            "silhouette", "davies_bouldin", "calinski_harabasz",
            "skill_variance", "schedule_overlap", "skill_coverage"]
    return pd.DataFrame(rows)[cols].set_index("model")


def print_comparison_table(eval_df: pd.DataFrame) -> None:
    """Pretty-print the comparison table with direction indicators."""
    DIRECTION = {
        "silhouette"       : "↑",
        "davies_bouldin"   : "↓",
        "calinski_harabasz": "↑",
        "skill_variance"   : "↕",
        "schedule_overlap" : "↑",
        "skill_coverage"   : "↑",
    }

    print("\n" + "=" * 70)
    print("MODEL COMPARISON TABLE")
    print("(↑ higher=better  ↓ lower=better  ↕ context-dependent)")
    print("=" * 70)

    display_df = eval_df.copy()
    for col, arrow in DIRECTION.items():
        if col in display_df.columns:
            display_df = display_df.rename(columns={col: f"{col} {arrow}"})

    # Format numeric columns
    fmt_df = display_df.copy()
    for c in fmt_df.columns:
        if c not in ["k", "team_sizes"]:
            try:
                fmt_df[c] = fmt_df[c].apply(lambda v: f"{v:.4f}" if pd.notna(v) else "—")
            except (TypeError, ValueError):
                pass

    print(fmt_df.to_string())
    print("=" * 70)
