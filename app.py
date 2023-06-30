import numpy as np
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model = pickle.load(open("model.pkl","rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods = ["POST"])
def predict():
    # output = [x for x in request.form.values()]
    # print(output)
    features = [x for x in request.form.values()]
    float_features = [float(x) for x in features[:5]]
    print(float_features)
    final_features = [np.array(float_features)]
    predictions = model.predict(final_features)
    output = round(predictions[0], 2)
    print(output)
    if output==0:
        return render_template("index.html", prediction_text = "You are Diabetic")
    else:
        return render_template("index.html", prediction_text = "You are not Diabetic")

if __name__ == "__main__":
    app.run(debug = False)