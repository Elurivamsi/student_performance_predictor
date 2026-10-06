"""Flask web app: enter student details and get a predicted final score."""
import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)
model = joblib.load("model.pkl")

FEATURES = ["study_hours", "attendance", "previous_score", "assignments", "sleep_hours"]
LIMITS = {
    "study_hours": (0, 24),
    "attendance": (0, 100),
    "previous_score": (0, 100),
    "assignments": (0, 10),
    "sleep_hours": (0, 24),
}


def grade(score):
    if score >= 85:
        return "A"
    if score >= 70:
        return "B"
    if score >= 55:
        return "C"
    if score >= 40:
        return "D"
    return "F"


def advice(values):
    tips = []
    if values["study_hours"] < 3:
        tips.append("Try to study at least 3-4 hours a day.")
    if values["attendance"] < 75:
        tips.append("Improve attendance to 75% or above.")
    if values["assignments"] < 7:
        tips.append("Complete more assignments; they boost your score.")
    if values["sleep_hours"] < 6:
        tips.append("Get 6-8 hours of sleep for better focus.")
    return tips or ["Great habits! Keep it up."]


@app.route("/", methods=["GET", "POST"])
def index():
    result, error, tips = None, None, []
    if request.method == "POST":
        try:
            values = {f: float(request.form[f]) for f in FEATURES}
            for f, (lo, hi) in LIMITS.items():
                if not lo <= values[f] <= hi:
                    raise ValueError(f"{f.replace('_', ' ').title()} must be between {lo} and {hi}.")
            score = float(model.predict(pd.DataFrame([values]))[0])
            score = max(0, min(100, score))
            result = {"score": round(score, 1), "grade": grade(score)}
            tips = advice(values)
        except (ValueError, KeyError) as e:
            error = str(e) if str(e) and "could not convert" not in str(e) else "Please enter valid numbers in all fields."
    return render_template("index.html", result=result, error=error, tips=tips)


if __name__ == "__main__":
    app.run(debug=True)
