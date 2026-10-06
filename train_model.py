"""Trains a Random Forest model to predict a student's final score."""
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

FEATURES = ["study_hours", "attendance", "previous_score", "assignments", "sleep_hours"]

df = pd.read_csv("students.csv")
X, y = df[FEATURES], df["final_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print(f"MAE : {mean_absolute_error(y_test, pred):.2f} marks")
print(f"R2  : {r2_score(y_test, pred):.3f}")

print("\nFeature importance:")
for name, imp in sorted(zip(FEATURES, model.feature_importances_), key=lambda t: -t[1]):
    print(f"  {name:15s} {imp:.3f}")

joblib.dump(model, "model.pkl")
print("\nModel saved to model.pkl")
