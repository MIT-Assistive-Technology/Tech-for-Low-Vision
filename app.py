from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
import os
import numpy as np
from slicer import retrieve, clean  # your existing slicer logic

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# Serve cached slice images
@app.route("/assets/cache/<path:filename>")
def serve_cache(filename):
    cache_dir = os.path.join("serve", "assets", "cache")
    return send_from_directory(cache_dir, filename)

# ===== ROUTES =====
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/slice", methods=["GET"])
def api_slice():
    required = [
        "file", "poseX", "poseY", "poseZ",
        "dirX", "dirY", "dirZ",
        "n", "i"
    ]
    missing = [p for p in required if p not in request.args]
    if missing:
        return jsonify({"error": "Missing parameters", "missing": missing}), 400

    # Extract parameters
    file = request.args["file"]
    pose = np.array([
        float(request.args["poseX"]),
        float(request.args["poseY"]),
        float(request.args["poseZ"])
    ])
    direction = np.array([
        float(request.args["dirX"]),
        float(request.args["dirY"]),
        float(request.args["dirZ"])
    ])
    n_slices = int(request.args["n"])
    i_slice = int(request.args["i"])

    file_path = os.path.join("models", file)

    result = retrieve(file_path, pose, direction, n_slices, i_slice)

    # Make URL relative for browser
    if "path" in result and os.path.exists(result["path"]):
        # Extract relative path after 'serve/assets/'
        rel_path = os.path.relpath(result["path"], "serve/assets")
        result["url"] = f"/assets/{rel_path.replace(os.sep, '/')}"
    else:
        result["url"] = None

    return jsonify(result)

@app.route("/api/send-to-device", methods=["POST"])
def api_send_to_device():
    data = request.json
    if not data or not all(k in data for k in ("sliceData", "width", "height")):
        return jsonify({"error": "Missing sliceData, width, or height"}), 400

    print(f"Sending slice {data['width']}x{data['height']} to device")
    return jsonify({"message": "Data grid successfully sent to the pin device"})

@app.route("/api/clean", methods=["POST"])
def api_clean():
    clean()
    return jsonify({"message": "Cache cleaned"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)
