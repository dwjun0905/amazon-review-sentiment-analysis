import os

from pipeline import generate_predictions
from flask import Flask, redirect, url_for, render_template, request

app = Flask(__name__, template_folder='../app/templates')


@app.route("/", methods=['POST', 'GET'])
def home():
    global url
    if request.method == "POST":
        if "url" in request.form:
            url = request.form['url']
            print("Start!\n")
            # Pipeline
            generate_predictions(url)
            return redirect(url_for("prediction"))

    return render_template("index.html")


@app.route("/prediction", methods=["POST", "GET"])
def prediction():
    return render_template("prediction.html")


# @app.route(""))
# def results
if __name__ == "__main__":
    app.debug = True
    app.run(use_reloader=False)


# Or a dynamic way: In the same home() html, shows input box until the user
# send the input, if send the request, change it to loading symbols and run the
# pipeline.. Then at the end, return the results with rendering the page
