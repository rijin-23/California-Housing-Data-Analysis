from flask import Flask, render_template, request
import California_Housing_Prediction as chp

app = Flask(__name__)

@app.route("/", methods=["GET","POST"])
def prediction():
    data = []
    pred = None
    if request.method == "POST":
        data = [request.form["longitude"], request.form["latitude"], request.form["age"],
                request.form["rooms"], request.form["bedrooms"], request.form["population"],
                request.form["households"], request.form["income"], request.form["proximity"]]
        pred = chp.data_prep_and_pred([data])
    return render_template("index.html", info = pred)


if __name__ == "__main__":
    app.run(debug=True)