"""
pipeline_utils.py

Shared utilities between the training notebook and the Streamlit prototype.
This module MUST be importable from both places so that joblib can correctly
pickle/unpickle the custom PositiveShifter transformer used inside the
preprocessing pipeline. Keep this file next to both the notebook and app.py.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PositiveShifter(BaseEstimator, TransformerMixin):
    """
    Shifts each column so all values become strictly positive, which
    Box-Cox requires. The shift is learned from the training data (fit)
    and reused unchanged on any other data (transform) -- no leakage,
    no re-computing per call.
    """

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        col_min = X.min(axis=0)
        self.shift_ = np.where(col_min <= 0, np.abs(col_min) + 1, 0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return X + self.shift_


# The exact feature list + order the clustering model was trained on.
# The Streamlit app must build its feature vector with these columns,
# in this order, before calling pipeline.predict().
FEATURES = [
    "accuracy_rate",
    "avg_attempts",
    "retry_rate",
    "avg_hint_dependency",
    "median_response_time",
    "max_opportunity",
    "total_interactions",
]

# Persona names mapped to the trained model's cluster labels (K-Means, k=4).
# IMPORTANT: cluster label -> persona mapping can shift if you re-run the
# grid search (K-Means label order isn't guaranteed stable across refits).
# Re-check this mapping against your notebook's cluster_profiles table
# after every retrain, using the corrected centroid table:
#   0 -> Active Strivers, 1 -> Deliberate Masters,
#   2 -> Overwhelmed & Struggling, 3 -> Efficient Practicers
PERSONA_NAMES = {
    0: "Active Strivers",
    1: "Deliberate Masters",
    2: "Overwhelmed & Struggling",
    3: "Efficient Practicers",
}

PERSONA_DESCRIPTIONS = {
    "Active Strivers": (
        "High practice volume and fast response times, but only moderate "
        "accuracy. This student persists and attempts many problems, but "
        "may benefit from more targeted practice rather than raw repetition."
    ),
    "Deliberate Masters": (
        "Very high accuracy with minimal attempts, retries, or hints, but "
        "slower, more careful responses. This student appears to succeed "
        "with little support."
    ),
    "Overwhelmed & Struggling": (
        "Low accuracy paired with heavy hint use, frequent retries, and "
        "slow response times. Highest-priority profile for additional "
        "scaffolding or step-by-step support."
    ),
    "Efficient Practicers": (
        "High accuracy, fast responses, and minimal hint/retry use after a "
        "moderate amount of practice. This student appears to have reached "
        "fluency and may be ready for more challenging material."
    ),
}


def build_feature_vector(question_log):
    """
    Convert a list of per-question interaction dicts collected during a
    Streamlit assignment session into the same student-level feature
    vector (a single-row DataFrame) that the clustering pipeline expects.

    Each item in `question_log` should be a dict with keys:
        - correct (bool): was the final submitted answer correct?
        - attempts (int): number of submit attempts on this question (>=1)
        - hints_used (int): number of hints the student requested
        - hints_available (int): number of hints available for this question
        - response_time_sec (float): time from question shown -> first submit
        - opportunity (int): how many times this student has practiced this
          skill before (including this question)

    Returns a pandas DataFrame with one row and the FEATURES columns, in
    the exact order the pipeline was trained on.
    """
    n = len(question_log)
    if n == 0:
        raise ValueError("question_log is empty -- no assignment data to score.")

    accuracy_rate = np.mean([q["correct"] for q in question_log])
    avg_attempts = np.mean([q["attempts"] for q in question_log])
    retry_rate = np.mean([1 if q["attempts"] > 1 else 0 for q in question_log])

    hint_dependencies = [
        (q["hints_used"] / q["hints_available"]) if q["hints_available"] > 0 else 0
        for q in question_log
    ]
    avg_hint_dependency = float(np.mean(hint_dependencies))

    median_response_time = float(np.median([q["response_time_sec"] for q in question_log]))
    avg_opportunity = float(np.mean([q["opportunity"] for q in question_log]))
    total_interactions = n

    row = {
        "accuracy_rate": accuracy_rate,
        "avg_attempts": avg_attempts,
        "retry_rate": retry_rate,
        "avg_hint_dependency": avg_hint_dependency,
        "median_response_time": median_response_time,
        "avg_opportunity": avg_opportunity,
        "total_interactions": total_interactions,
    }

    return pd.DataFrame([row], columns=FEATURES)
