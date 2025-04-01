from slicer import test
from flask import Flask

app = Flask(__name__)

@app.route("/")
def index():
  return "root"

@app.route("/test")
def callTest():
  test()
  return "success"
