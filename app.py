from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("anemia_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    gender = float(request.form["Gender"])
    hemoglobin = float(request.form["Haemoglobin"])
    mch = float(request.form["MCH"])
    mchc = float(request.form["MCHC"])
    mcv = float(request.form["MCV"])

    input_data = [[gender, hemoglobin, mch, mchc, mcv]]

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        result = "Anemia Detected"
    else:
        result = "No Anemia Detected"

    return render_template("predict.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)