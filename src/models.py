"""
src/models.py
-------------
Model wrappers for the four Teammate Matcher approaches.

  Model 1 — K-Means (similarity baseline)
  Model 2 — Agglomerative Hierarchical Clustering (Ward linkage)
  Model 3 — Size-Constrained Assignment via Hungarian Algorithm ★ beyond syllabus
  Model 4 — Gaussian Mixture Model (soft assignments on skill features) ★ beyond syllabus

All models accept a numpy array or pandas DataFrame of pre-scaled features and return
a TeamingResult namedtuple containing:
    labels      : 1-D array of team assignments (0-indexed)
    k           : number of teams
    model_obj   : fitted sklearn object (for accessing attributes downstream)
    meta        : dict of model-specific metadata (centroids, proba, dendrogram, etc.)
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class TeamingResult:
    """Standardised output for every model."""
    model_name : str
    labels     : np.ndarray          # shape (N,) — team index per student (0-based)
    k          : int                 # number of teams
    model_obj  : Any                 # fitted sklearn / scipy object
    meta       : Dict[str, Any] = field(default_factory=dict)

    # ── Convenience ──────────────────────────────────────────────────────────

    def team_members(self, team_idx: int) -> np.ndarray:
        """Return row indices of students assigned to team_idx."""
        return np.where(self.labels == team_idx)[0]

    def team_sizes(self) -> Dict[int, int]:
        """Return {team_idx: size} mapping."""
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))

    def summary(self) -> str:
        sizes = self.team_sizes()
        lines = [
            f"Model     : {self.model_name}",
            f"Teams (k) : {self.k}",
            f"Team sizes: {sizes}",
        ]
        return "\n".join(lines)


# ── Helper ───────────────────────────────────────────────────────────────────

def _to_array(X) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        return X.values.astype(float)
    return np.asarray(X, dtype=float)


# ── Deployment helpers: scale team count to arbitrary class sizes ─────────────

def derive_k(n_students: int, preferred_size: int = 4) -> int:
    """
    Derive the number of teams from the class size and a preferred team size.

        k = ceil(N / preferred_size)

    Bounded to [2, N // 2] so we never request more teams than the data can
    support (every team needs at least ~2 members for clustering metrics).

    Examples
    --------
    >>> derive_k(24, 4)   # 6 teams of 4
    6
    >>> derive_k(31, 4)   # 8 teams of ~3-4 (matches the original deployment)
    8
    """
    if n_students < 2:
        return 1
    k = max(2, math.ceil(n_students / max(1, preferred_size)))
    return min(k, max(2, n_students // 2))


def validate_n(n: int) -> List[str]:
    """
    Return a list of human-readable warnings about the uploaded class size.
    Empty list means N is in the comfortable range.
    """
    warnings_list: List[str] = []
    if n < 9:
        warnings_list.append(
            f"Only {n} responses detected. Clustering is unreliable below ~9 students "
            "— consider assigning teams manually for a class this small."
        )
    if n > 150:
        warnings_list.append(
            f"{n} responses detected. This tool was validated on smaller cohorts; "
            "results should still be reasonable but may take a moment to compute."
        )
    return warnings_list


# ── Size guardrails: honor a target team size across ANY model ────────────────
# K-Means, Agglomerative, and GMM cluster purely by similarity and have no notion
# of team size, so raw cluster sizes drift (instructors reported teams of 2–6 when
# they asked for 3–4). These helpers re-distribute members so every team lands at
# the chosen size (±1 for the unavoidable remainder) while keeping each team's
# centre of gravity — i.e. the model still decides *who groups with whom*; the
# guardrail only enforces *how many*.

def balanced_capacities(n: int, k: int) -> List[int]:
    """
    Split ``n`` students into ``k`` teams as evenly as possible.

    Returns k capacities summing to n, each equal to ⌊n/k⌋ or ⌈n/k⌉, so no two
    teams differ in size by more than one. The first ``n % k`` teams get the
    extra member.

    >>> balanced_capacities(31, 8)
    [4, 4, 4, 4, 4, 4, 4, 3]
    >>> balanced_capacities(20, 4)
    [5, 5, 5, 5]
    """
    if k <= 0:
        raise ValueError("k must be a positive integer")
    base, rem = divmod(n, k)
    return [base + 1] * rem + [base] * (k - rem)


def _assign_to_capacities(X_arr: np.ndarray, centroids: np.ndarray,
                          capacities: List[int]) -> np.ndarray:
    """
    Optimally assign every row of ``X_arr`` to a team subject to fixed per-team
    capacities, via the Hungarian algorithm on a capacity-expanded cost matrix.

    Each team column is replicated ``capacities[j]`` times so the assignment
    matrix is square (N × N); ``sum(capacities)`` must equal ``len(X_arr)``.
    Returns 0-indexed team labels (one per row of X_arr).
    """
    n = X_arr.shape[0]
    if sum(capacities) != n:
        raise ValueError("capacities must sum to the number of students")
    cost = cdist(X_arr, centroids, metric="euclidean")            # (N, k)
    col_team = np.repeat(np.arange(len(capacities)), capacities)   # (N,) col → team
    expanded = cost[:, col_team]                                   # (N, N)
    row_ind, col_ind = linear_sum_assignment(expanded)
    labels = np.empty(n, dtype=int)
    labels[row_ind] = col_team[col_ind]
    return labels


def balance_team_sizes(result: "TeamingResult", X) -> "TeamingResult":
    """
    Re-assign students so every team is the same size (±1 for the remainder),
    while preserving the matching criteria the chosen model produced.

    The model has already grouped students by similarity / complementarity; this
    keeps each cluster's centroid but reshuffles members at the margins so the
    instructor's chosen team size is actually honored — the guardrail the pure
    clustering models (K-Means, Agglomerative, GMM) otherwise lack.

    Returns a NEW TeamingResult; the original is left untouched.
    """
    X_arr = _to_array(X)
    n = X_arr.shape[0]
    labels = np.asarray(result.labels)

    # Prefer the model's own centroids; fall back to GMM means, then member means.
    centroids = result.meta.get("centroids")
    if centroids is None or len(centroids) != result.k:
        centroids = result.meta.get("means")
    if centroids is None or len(centroids) != result.k:
        uniq = np.unique(labels[labels >= 0])
        centroids = np.vstack([X_arr[labels == u].mean(axis=0) for u in uniq])
    centroids = np.asarray(centroids, dtype=float)
    k = len(centroids)

    caps = balanced_capacities(n, k)
    new_labels = _assign_to_capacities(X_arr, centroids, caps)

    return TeamingResult(
        model_name=result.model_name,
        labels=new_labels,
        k=k,
        model_obj=result.model_obj,
        meta={**result.meta, "size_balanced": True, "capacities": caps},
    )


def _select_k_silhouette(X: np.ndarray, k_range=(2, 8),
                          random_state=42) -> int:
    """
    Sweep K-Means over k_range; return k that maximises Silhouette Score.
    Falls back to the middle of k_range if all scores are equal.
    """
    from sklearn.metrics import silhouette_score
    best_k, best_score = None, -1.0
    for k in range(k_range[0], k_range[1] + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        if len(np.unique(labels)) < 2:
            continue
        score = silhouette_score(X, labels)
        if score > best_score:
            best_score, best_k = score, k
    return best_k if best_k is not None else k_range[0]


# ── Model 1: K-Means ──────────────────────────────────────────────────────────

def kmeans_teams(X, k: Optional[int] = None, k_range=(2, 8),
                 random_state: int = 42) -> TeamingResult:
    """
    K-Means clustering on compatibility features.

    Parameters
    ----------
    X           : array-like (N, F) — compatibility feature set (pre-scaled)
    k           : number of clusters. If None, selected via Silhouette Score sweep.
    k_range     : range of k values to sweep when k is None
    random_state: reproducibility seed

    Returns
    -------
    TeamingResult with:
        meta['centroids']         : (k, F) centroid coordinates
        meta['inertia']           : within-cluster sum of squares
        meta['silhouette_scores'] : {k: score} from the sweep (if k was auto-selected)
    """
    X_arr = _to_array(X)
    meta: Dict[str, Any] = {}

    # ── Optional elbow / silhouette sweep ─────────────────────────────────────
    if k is None:
        from sklearn.metrics import silhouette_score
        inertias, sil_scores = {}, {}
        for kk in range(k_range[0], k_range[1] + 1):
            km_tmp = KMeans(n_clusters=kk, random_state=random_state, n_init=10)
            lbl = km_tmp.fit_predict(X_arr)
            inertias[kk] = km_tmp.inertia_
            if len(np.unique(lbl)) >= 2:
                sil_scores[kk] = silhouette_score(X_arr, lbl)
        k = max(sil_scores, key=sil_scores.get) if sil_scores else k_range[0]
        meta["inertia_sweep"]    = inertias
        meta["silhouette_sweep"] = sil_scores
        print(f"  [K-Means] Auto-selected k={k} "
              f"(silhouette={sil_scores.get(k, '—'):.3f})")

    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(X_arr)

    meta["centroids"] = km.cluster_centers_
    meta["inertia"]   = km.inertia_

    return TeamingResult(
        model_name="K-Means",
        labels=labels,
        k=k,
        model_obj=km,
        meta=meta,
    )


# ── Model 2: Agglomerative (Ward) ─────────────────────────────────────────────

def agglomerative_teams(X, k: Optional[int] = None, k_range=(2, 8),
                        linkage_method: str = "ward") -> TeamingResult:
    """
    Agglomerative hierarchical clustering with Ward linkage.

    Parameters
    ----------
    X              : array-like (N, F) — compatibility feature set (pre-scaled)
    k              : number of clusters. If None, selected via Silhouette Score sweep.
    k_range        : sweep range when k is None
    linkage_method : scipy linkage method (default 'ward')

    Returns
    -------
    TeamingResult with:
        meta['linkage_matrix'] : Z matrix for dendrogram plotting
        meta['dendrogram_data']: output of scipy dendrogram (for plotting)
    """
    X_arr = _to_array(X)
    meta: Dict[str, Any] = {}

    # ── Build full linkage matrix for dendrogram ───────────────────────────────
    Z = linkage(X_arr, method=linkage_method)
    meta["linkage_matrix"] = Z

    # ── Select k ──────────────────────────────────────────────────────────────
    if k is None:
        from sklearn.metrics import silhouette_score
        sil_scores = {}
        for kk in range(k_range[0], k_range[1] + 1):
            lbl = fcluster(Z, t=kk, criterion="maxclust") - 1  # 0-indexed
            if len(np.unique(lbl)) >= 2:
                sil_scores[kk] = silhouette_score(X_arr, lbl)
        k = max(sil_scores, key=sil_scores.get) if sil_scores else k_range[0]
        meta["silhouette_sweep"] = sil_scores
        print(f"  [Agglomerative] Auto-selected k={k} "
              f"(silhouette={sil_scores.get(k, '—'):.3f})")

    labels = fcluster(Z, t=k, criterion="maxclust") - 1  # 0-indexed

    # ── Truncated dendrogram data (for plotting) ───────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dend = dendrogram(Z, truncate_mode="level", p=5, no_plot=True)
    meta["dendrogram_data"] = dend

    # ── Also fit sklearn object for metric compatibility ───────────────────────
    agg = AgglomerativeClustering(n_clusters=k, linkage=linkage_method)
    agg.fit(X_arr)

    return TeamingResult(
        model_name="Agglomerative (Ward)",
        labels=labels,
        k=k,
        model_obj=agg,
        meta=meta,
    )


# ── Model 3: Hungarian Size-Constrained Assignment ★ ─────────────────────────

def hungarian_teams(X, k: Optional[int] = None, team_size: Optional[int] = None,
                    random_state: int = 42) -> TeamingResult:
    """
    Size-constrained team assignment via the Hungarian Algorithm.

    Algorithm
    ---------
    1. Run K-Means to obtain k cluster centroids.
    2. Build cost matrix C[i, j] = Euclidean distance from student i to centroid j.
       Duplicate centroid columns to enforce an equal team-size constraint:
       each centroid column is replicated team_size times → expanded cost matrix
       shape (N, N).
    3. Solve the linear sum assignment on the expanded matrix using
       scipy.optimize.linear_sum_assignment (O(N³) Hungarian Algorithm).
    4. Map expanded column indices back to centroid indices → team labels.

    Parameters
    ----------
    X         : array-like (N, F) — compatibility feature set (pre-scaled)
    k         : number of teams. If None, auto-selected via Silhouette Score.
    team_size : target students per team. If None, computed as ⌊N/k⌋.
                Remaining students (N mod k overflow) are assigned greedily.
    random_state: reproducibility seed

    Returns
    -------
    TeamingResult with:
        meta['centroids']    : (k, F) centroid coordinates from K-Means
        meta['cost_matrix']  : (N, k) raw Euclidean distances
        meta['km_labels']    : raw K-Means cluster labels (before rebalancing)
        meta['team_size']    : enforced team size
    """
    X_arr = _to_array(X)
    N     = X_arr.shape[0]
    meta: Dict[str, Any] = {}

    # ── Step 1: K-Means to get centroids ──────────────────────────────────────
    if k is None:
        k = _select_k_silhouette(X_arr, random_state=random_state)
        print(f"  [Hungarian] Auto-selected k={k}")

    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    km_labels = km.fit_predict(X_arr)
    centroids = km.cluster_centers_
    meta["centroids"]   = centroids
    meta["km_labels"]   = km_labels

    # ── Step 2: Cost matrix (N × k) ───────────────────────────────────────────
    cost_matrix = cdist(X_arr, centroids, metric="euclidean")  # (N, k)
    meta["cost_matrix"] = cost_matrix

    # ── Team size ─────────────────────────────────────────────────────────────
    if team_size is None:
        team_size = N // k
    meta["team_size"] = team_size
    n_full = team_size * k   # students accounted for in balanced assignment

    # ── Step 3: Expand cost matrix → (n_full, n_full) for size constraint ─────
    # Replicate each centroid column team_size times
    expanded = np.tile(cost_matrix[:n_full, :], (1, team_size))  # (n_full, k*team_size)
    # expanded is already (n_full, n_full) since k * team_size == n_full

    row_ind, col_ind = linear_sum_assignment(expanded)

    # ── Step 4: Map expanded columns back to centroid index ───────────────────
    labels = np.full(N, -1, dtype=int)
    for r, c in zip(row_ind, col_ind):
        labels[r] = c % k   # expanded col → centroid index

    # ── Handle overflow students (if N % k != 0) ──────────────────────────────
    overflow = np.where(labels == -1)[0]
    if len(overflow) > 0:
        # Assign each overflow student to their nearest centroid
        for ov_idx in overflow:
            labels[ov_idx] = np.argmin(cost_matrix[ov_idx])
        print(f"  [Hungarian] {len(overflow)} overflow student(s) assigned greedily")

    return TeamingResult(
        model_name="Hungarian Assignment",
        labels=labels,
        k=k,
        model_obj=km,  # underlying K-Means
        meta=meta,
    )


# ── Model 4: Gaussian Mixture Model ★ ────────────────────────────────────────

def gmm_teams(X, k: Optional[int] = None, k_range=(2, 8),
              random_state: int = 42) -> TeamingResult:
    """
    Gaussian Mixture Model for soft skill-profile clustering.

    Applied to the complementarity (skills-only) feature set.
    Produces a probability vector per student rather than a hard assignment,
    surfacing ambiguous students who straddle multiple skill archetypes.

    Model selection uses BIC (Bayesian Information Criterion):
        BIC = k · ln(N) - 2 · ln(L̂)
    Lower BIC = better fit penalised for complexity.

    Parameters
    ----------
    X           : array-like (N, F) — complementarity (skill) feature set
    k           : number of Gaussian components. If None, selected via BIC.
    k_range     : BIC sweep range when k is None
    random_state: reproducibility seed

    Returns
    -------
    TeamingResult with:
        meta['proba']          : (N, k) soft membership probabilities
        meta['bic_scores']     : {k: BIC} from sweep
        meta['aic_scores']     : {k: AIC} from sweep
        meta['means']          : (k, F) component means (skill profiles)
        meta['ambiguous_mask'] : boolean mask — students where max_prob < 0.6
    """
    X_arr = _to_array(X)
    meta: Dict[str, Any] = {}

    # ── BIC sweep to select k ─────────────────────────────────────────────────
    if k is None:
        bic_scores, aic_scores = {}, {}
        for kk in range(k_range[0], k_range[1] + 1):
            gmm_tmp = GaussianMixture(
                n_components=kk, covariance_type="full",
                random_state=random_state, n_init=5
            )
            gmm_tmp.fit(X_arr)
            bic_scores[kk] = gmm_tmp.bic(X_arr)
            aic_scores[kk] = gmm_tmp.aic(X_arr)
        k = min(bic_scores, key=bic_scores.get)
        meta["bic_scores"] = bic_scores
        meta["aic_scores"] = aic_scores
        print(f"  [GMM] Auto-selected k={k} via BIC "
              f"(BIC={bic_scores[k]:.1f})")

    gmm = GaussianMixture(
        n_components=k, covariance_type="full",
        random_state=random_state, n_init=5
    )
    gmm.fit(X_arr)

    proba  = gmm.predict_proba(X_arr)   # (N, k)
    labels = gmm.predict(X_arr)         # hard assignment = argmax(proba)

    meta["proba"]           = proba
    meta["means"]           = gmm.means_
    meta["covariances"]     = gmm.covariances_
    meta["weights"]         = gmm.weights_
    meta["ambiguous_mask"]  = proba.max(axis=1) < 0.6   # students with uncertain fit

    n_ambiguous = meta["ambiguous_mask"].sum()
    if n_ambiguous > 0:
        print(f"  [GMM] {n_ambiguous} students have max membership probability < 0.60 "
              f"— flagged for human review")

    return TeamingResult(
        model_name="GMM",
        labels=labels,
        k=k,
        model_obj=gmm,
        meta=meta,
    )


# ── Convenience: run all four models ─────────────────────────────────────────

def run_all_models(compat_features, compl_features,
                   k: Optional[int] = None,
                   random_state: int = 42) -> Dict[str, TeamingResult]:
    """
    Fit all four models and return results as a dict.

    Parameters
    ----------
    compat_features : DataFrame — compatibility feature set (availability + work style)
    compl_features  : DataFrame — complementarity feature set (skills only)
    k               : override number of teams for all models (None = auto-select)
    random_state    : seed

    Returns
    -------
    {
        'kmeans'      : TeamingResult,
        'agglomerative': TeamingResult,
        'hungarian'   : TeamingResult,
        'gmm'         : TeamingResult,
    }
    """
    print("Running Model 1 — K-Means...")
    km_result = kmeans_teams(compat_features, k=k, random_state=random_state)

    print("\nRunning Model 2 — Agglomerative (Ward)...")
    agg_result = agglomerative_teams(compat_features, k=km_result.k)

    print(f"\nRunning Model 3 — Hungarian Assignment (k={km_result.k})...")
    hung_result = hungarian_teams(compat_features, k=km_result.k,
                                   random_state=random_state)

    print("\nRunning Model 4 — GMM (complementarity features)...")
    gmm_result = gmm_teams(compl_features, random_state=random_state)

    return {
        "kmeans"       : km_result,
        "agglomerative": agg_result,
        "hungarian"    : hung_result,
        "gmm"          : gmm_result,
    }
