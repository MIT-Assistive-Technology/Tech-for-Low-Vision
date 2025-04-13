from flask import Flask

from slicer import test

app = Flask(__name__)


@app.route("/")
def index():
    return "root"


@app.route("/test")
def callTest():
    test()
    return "success"
