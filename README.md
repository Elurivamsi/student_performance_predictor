# Student Performance Predictor (Python, Flask, scikit-learn)

A machine learning web app that predicts a student's final exam score from
study hours, attendance, previous score, assignments completed and sleep hours.

## Setup
```
pip install -r requirements.txt
python generate_data.py     # creates students.csv (sample data)
python train_model.py       # trains the model, prints MAE / R2, saves model.pkl
python app.py               # open http://127.0.0.1:5000
```

## Files
- generate_data.py : creates the sample dataset (replace with a real dataset if you like)
- train_model.py   : trains and evaluates a Random Forest regression model
- app.py           : Flask web app with input validation
- templates/index.html : web form UI

## Tech
Python, Pandas, NumPy, scikit-learn, Flask, HTML, CSS
