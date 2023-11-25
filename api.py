from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("emoji_predictor.joblib")

@app.route('/', methods=['GET'])
def form():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    prediction = model.predict(pd.DataFrame([data]))

    return render_template("form.html", prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)