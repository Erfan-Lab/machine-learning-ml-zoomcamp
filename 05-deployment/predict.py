import pickle
from flask import Flask
from flask import request
from flask import jsonify


output_file = "model_C=0.5.bin"


with open(output_file, "rb") as file_input:
    dv, model = pickle.load(file_input)


app = Flask("churn_prediction")


@app.route("/prediction", methods=["POST"])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0][1]
    churn = y_pred >= 0.5

    result = {
        "churn_probability": y_pred,
        "churn": bool(churn)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)