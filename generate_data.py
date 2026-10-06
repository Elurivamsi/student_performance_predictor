"""Generates a sample student dataset (students.csv).

You can replace students.csv with a real dataset (e.g. from Kaggle) as long as
it has the same column names.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 1000

study_hours = rng.uniform(0, 10, n).round(1)          # hours studied per day
attendance = rng.uniform(40, 100, n).round(1)         # attendance %
previous_score = rng.uniform(30, 100, n).round(1)     # last exam marks
assignments = rng.integers(0, 11, n)                  # assignments done (out of 10)
sleep_hours = rng.uniform(4, 9, n).round(1)

final_score = (
    0.30 * previous_score
    + 4.0 * study_hours
    + 0.20 * attendance
    + 1.5 * assignments
    + 0.8 * sleep_hours
    + rng.normal(0, 4, n)
)
final_score = np.clip(final_score, 0, 100).round(1)

df = pd.DataFrame({
    "study_hours": study_hours,
    "attendance": attendance,
    "previous_score": previous_score,
    "assignments": assignments,
    "sleep_hours": sleep_hours,
    "final_score": final_score,
})
df.to_csv("students.csv", index=False)
print(f"Saved students.csv with {len(df)} rows")
