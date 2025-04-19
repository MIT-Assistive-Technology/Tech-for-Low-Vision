import numpy as np
from flask import Flask, request

from slicer import clean, retrieve, test

app = Flask(__name__)


fmap = lambda f: lambda l: [f(x) for x in l]


@app.route("/")
def index():
    return "root"


@app.route("/test")
def callTest():
    test()
    return "success"


@app.route("/clean")
def cleanCache():
    clean()
    return "success"


@app.route("/get")
def get():
    file = request.args.get("file")
    if file is None:
        return "error; need `file`"

    poseX = request.args.get("poseX")
    poseY = request.args.get("poseY")
    poseZ = request.args.get("poseZ")

    if poseX is None or poseY is None or poseZ is None:
        return "error; need `poseX`, `poseY`, and `poseZ`"

    dirX = request.args.get("dirX")
    dirY = request.args.get("dirY")
    dirZ = request.args.get("dirZ")

    if dirX is None or dirY is None or dirZ is None:
        return "error; need `dirX`, `dirY`, and `dirZ`"

    n = request.args.get("n")
    i = request.args.get("i")

    if n is None or i is None:
        return "error; need `n` and `i`"

    return retrieve(
        file,
        np.array([fmap(float, [poseX, poseY, poseZ])]),  # pyright: ignore
        np.array([fmap(float, [dirX, dirY, dirZ])]),  # pyright: ignore
        int(n),
        int(i),
    )
