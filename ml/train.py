from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
import joblib

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "oulad"
ART = ROOT / "artifacts"
CUTOFF = 28
KEYS = ["code_module", "code_presentation", "id_student"]
CAT = ["code_module", "code_presentation"]
NUM = ["num_of_prev_attempts", "studied_credits", "clicks_day_0_28", "active_days_day_0_28"]
FEATURES = CAT + NUM


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def build_cohort() -> tuple[pd.DataFrame, pd.Series, pd.Series, dict]:
    info = pd.read_csv(RAW / "studentInfo.csv", na_values="?")
    registration = pd.read_csv(RAW / "studentRegistration.csv", na_values="?")
    info["final_result"] = info["final_result"].astype(str)
    merged = info.merge(registration, on=KEYS, how="left", validate="one_to_one")
    reg_day = pd.to_numeric(merged["date_registration"], errors="coerce")
    unreg_day = pd.to_numeric(merged["date_unregistration"], errors="coerce")
    # Operational definition: eligible at end of day 28 iff registered by then
    # and not already withdrawn on/before that day. These dates are eligibility
    # controls only and are never predictors.
    eligible = reg_day.notna() & (reg_day <= CUTOFF) & (unreg_day.isna() | (unreg_day > CUTOFF))
    cohort = merged.loc[eligible].copy()

    click_totals: dict[tuple, int] = {}
    active_dates: set[tuple] = set()
    chunks = pd.read_csv(RAW / "studentVle.csv", usecols=KEYS + ["date", "sum_click"], chunksize=250_000)
    for chunk in chunks:
        day = pd.to_numeric(chunk["date"], errors="coerce")
        part = chunk.loc[day.ge(0) & day.le(CUTOFF)].copy()
        if part.empty:
            continue
        daily = part.groupby(KEYS + ["date"], as_index=False, sort=False)["sum_click"].sum()
        for row in daily.itertuples(index=False, name=None):
            key = tuple(row[:3])
            click_totals[key] = click_totals.get(key, 0) + int(row[4])
            active_dates.add(key + (int(row[3]),))
    click_counts = pd.Series(click_totals, dtype="int64", name="clicks_day_0_28")
    click_counts.index = pd.MultiIndex.from_tuples(click_counts.index, names=KEYS)
    daily_counts = pd.Series([1] * len(active_dates), index=pd.MultiIndex.from_tuples([x[:3] for x in active_dates], names=KEYS))
    active_counts = daily_counts.groupby(level=KEYS).sum().rename("active_days_day_0_28")
    cohort = cohort.merge(click_counts.reset_index(), on=KEYS, how="left", validate="one_to_one")
    cohort = cohort.merge(active_counts.reset_index(), on=KEYS, how="left", validate="one_to_one")
    cohort[["clicks_day_0_28", "active_days_day_0_28"]] = cohort[["clicks_day_0_28", "active_days_day_0_28"]].fillna(0).astype(int)
    X = cohort[FEATURES].copy()
    for col in CAT:
        X[col] = X[col].fillna("Unknown").astype(str)
    for col in NUM:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        X[col] = X[col].fillna(X[col].median())
    y = (cohort["final_result"] == "Withdrawn").astype(int)
    groups = cohort["id_student"].astype(str)
    counts = {
        "raw_enrollments": int(len(info)), "eligible_enrollments": int(len(cohort)),
        "excluded_missing_registration": int(reg_day.isna().sum()),
        "excluded_registered_after_cutoff": int((reg_day > CUTOFF).sum()),
        "excluded_withdrawn_on_or_before_cutoff": int((unreg_day.notna() & (unreg_day <= CUTOFF)).sum()),
        "positive_withdrawn": int(y.sum()), "negative_other_final_results": int((1-y).sum()),
        "class_values": sorted(info["final_result"].dropna().unique().tolist()),
    }
    return X, y, groups, counts


def make_pipeline(alpha: float = 0.0) -> Pipeline:
    prep = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
        ("numeric", "passthrough", NUM),
    ], remainder="drop", verbose_feature_names_out=True)
    # A pre-specified depth limit makes the academic tree inspectable; balanced
    # class weights reflect the higher cost of missing the minority Withdrawn class.
    tree = DecisionTreeClassifier(criterion="gini", random_state=42, max_depth=8,
                                  ccp_alpha=float(alpha), class_weight="balanced")
    return Pipeline([("preprocess", prep), ("tree", tree)])


def scores(y_true, y_pred) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_withdrawn": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_withdrawn": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_withdrawn": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "confusion_matrix_labels": ["Not withdrawn", "Withdrawn"],
        "confusion_matrix": cm.tolist(),
    }


def train() -> None:
    X, y, groups, counts = build_cohort()
    split = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    tr, te = next(split.split(X, y, groups))
    X_train, X_test = X.iloc[tr].reset_index(drop=True), X.iloc[te].reset_index(drop=True)
    y_train, y_test = y.iloc[tr].reset_index(drop=True), y.iloc[te].reset_index(drop=True)
    g_train, g_test = groups.iloc[tr].reset_index(drop=True), groups.iloc[te].reset_index(drop=True)

    # Candidate alphas come only from training data. CV is grouped by student
    # so one student's repeated enrolments cannot cross train/validation folds.
    preliminary = make_pipeline().fit(X_train, y_train)
    Xt = preliminary.named_steps["preprocess"].transform(X_train)
    path = preliminary.named_steps["tree"].cost_complexity_pruning_path(Xt, y_train)
    positive_alphas = path.ccp_alphas[path.ccp_alphas > 0]
    # Log spacing covers the highly uneven pruning path much better than
    # evenly spaced quantiles, which can otherwise leave every candidate huge.
    alphas = np.unique(np.r_[0.0, np.geomspace(positive_alphas.min(), path.ccp_alphas.max(), 14)]).tolist()
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(make_pipeline(), {"tree__ccp_alpha": alphas}, scoring="f1", cv=cv, refit=False, n_jobs=1, return_train_score=False)
    search.fit(X_train, y_train, groups=g_train)
    candidate_complexities = []
    for alpha in alphas:
        candidate = make_pipeline(alpha).fit(X_train, y_train)
        candidate_tree = candidate.named_steps["tree"]
        candidate_complexities.append({"alpha": float(alpha), "depth": int(candidate_tree.get_depth()), "leaves": int(candidate_tree.get_n_leaves())})
    cv_means = search.cv_results_["mean_test_score"]
    cv_stds = search.cv_results_["std_test_score"]
    best_cv_index = int(np.argmax(cv_means))
    one_se_floor = float(cv_means[best_cv_index] - cv_stds[best_cv_index])
    eligible_indices = [i for i, score in enumerate(cv_means) if score >= one_se_floor]
    chosen_index = min(eligible_indices, key=lambda i: (candidate_complexities[i]["leaves"], candidate_complexities[i]["depth"], -alphas[i]))
    selected_alpha = float(alphas[chosen_index])
    model: Pipeline = make_pipeline(selected_alpha).fit(X_train, y_train)
    pred = model.predict(X_test)
    majority = int(y_train.value_counts().idxmax())
    baseline_pred = np.full(len(y_test), majority, dtype=int)
    baseline = scores(y_test, baseline_pred)
    pruned = scores(y_test, pred)

    unpruned = make_pipeline(0.0).fit(X_train, y_train)
    unpruned_pred = unpruned.predict(X_test)
    unpruned_metrics = scores(y_test, unpruned_pred)
    unpruned_tree = unpruned.named_steps["tree"]
    tree = model.named_steps["tree"]
    candidates = [
        {**complexity, "mean_cv_f1_withdrawn": float(m), "std_cv_f1": float(s),
         "within_one_standard_error": bool(float(m) >= one_se_floor), "selected": i == chosen_index}
        for i, (complexity, m, s) in enumerate(zip(candidate_complexities, cv_means, cv_stds))
    ]
    metrics = {
        "software": {"python": sys.version.split()[0], "scikit_learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
        "dataset": "Open University Learning Analytics Dataset (OULAD)", "source_doi": "10.24432/C5KK69",
        "license": "CC BY 4.0", "label": "Withdrawn versus Pass/Fail/Distinction",
        "prediction_cutoff_day": CUTOFF, "eligibility": "registered by day 28 and not unregistered on/before day 28; missing registration date excluded",
        "features": FEATURES, "categorical_features": CAT, "numeric_features": NUM,
        "tree_policy": {"criterion": "gini", "max_depth": 8, "class_weight": "balanced", "ccp_alpha_selection": "5-fold grouped CV; simplest candidate within 1 SE of best Withdrawn F1"},
        "data_counts": counts, "split": {"method": "GroupShuffleSplit", "test_size": 0.20, "random_state": 42,
            "train_enrollments": len(tr), "test_enrollments": len(te), "train_students": int(g_train.nunique()), "test_students": int(g_test.nunique()), "student_overlap": int(len(set(g_train)&set(g_test)))},
        "baseline_majority_class": majority, "baseline_test": baseline,
        "unpruned_tree_test": unpruned_metrics,
        "pruned_tree_test": pruned,
        "unpruned_tree": {"depth": int(unpruned_tree.get_depth()), "leaves": int(unpruned_tree.get_n_leaves()), "nodes": int(unpruned_tree.tree_.node_count)},
        "pruned_tree": {"ccp_alpha": selected_alpha, "depth": int(tree.get_depth()),
            "leaves": int(tree.get_n_leaves()), "nodes": int(tree.tree_.node_count)},
        "cv": {"folds": 5, "method": "StratifiedGroupKFold", "selection_metric": "F1 for Withdrawn", "best_mean_f1": float(cv_means[best_cv_index]), "one_standard_error_floor": one_se_floor, "selection_rule": "simplest tree (fewest leaves, then depth) within one standard error of best mean F1", "candidates": candidates},
        "limitations": ["OULAD records are from 2013–2014 and may not represent current students.", "This predicts administrative course withdrawal, not failure or lack of learning.", "A course withdrawal can follow many unobserved causes; this model is support information, not a decision about a student.", "The tree's leaf class fraction is not a calibrated probability.", "Group holdout is by anonymized student ID; generalization to a new institution or time period is not established."],
    }
    ART.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ART / "cart_withdrawal.joblib")
    (ART / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    X_train.to_csv(ART / "train_features.csv", index=False)
    # Keep a single known-good form example from a real training row.
    example = X_train.iloc[0].to_dict()
    meta = {
        "feature_order": FEATURES, "categorical_features": CAT, "numeric_features": NUM,
        "categories": {c: sorted(X_train[c].astype(str).unique().tolist()) for c in CAT},
        "numeric_ranges": {c: {"min": float(X_train[c].min()), "max": float(X_train[c].max()), "integer": c in ["num_of_prev_attempts", "studied_credits", "active_days_day_0_28"]} for c in NUM},
        "example": example, "selected_alpha": metrics["pruned_tree"]["ccp_alpha"],
        "dataset_zip_sha256": sha256(next(p for p in [ROOT / "data" / "raw" / "oulad_valid.zip", ROOT / "data" / "raw" / "oulad.zip"] if p.exists())), "sklearn_version": sklearn.__version__,
    }
    (ART / "model_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({"counts": counts, "split": metrics["split"], "baseline": baseline, "unpruned": unpruned_metrics, "pruned": pruned, "tree": metrics["pruned_tree"], "alpha": metrics["pruned_tree"]["ccp_alpha"]}, indent=2))

if __name__ == "__main__":
    train()
