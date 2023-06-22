"""
If you are in the same directory as this file (app.py), you can run run the app using gunicorn:

    $ gunicorn --bind 0.0.0.0:<PORT> app:app

gunicorn can be installed via:

    $ pip install gunicorn

"""
import os
import logging
import torch
from resnet18 import ResNet18
from PIL import Image
import json
from re import A
from flask import Flask, jsonify, request
from prediction import split_example
from utils import labels_to_fen
from torchvision import transforms
import base64
import io
from comet_ml import API

LOG_FILE = os.environ.get("FLASK_LOG", "flask.log")

app = Flask(__name__)

def load_model(workspace, model_name, version):
    if not os.path.exists(f"models/{workspace}/{model_name}/{version}/"):
        os.makedirs(f"models/{workspace}/{model_name}/", exist_ok=True)
    api = API(os.environ["COMET_API_KEY"])
    api.download_registry_model(
        workspace,
        model_name,
        version,
        output_path=f"models/{workspace}/{model_name}/{version}",
        expand=True
    )
    app.logger.info(f"Downloaded model {workspace}/{model_name}/{version}")

    path = f"models/{workspace}/{model_name}/{version}"
    file = os.listdir(path)[0]

    with open(os.path.join(path, file), "rb") as f:
        model_obj = ResNet18(num_classes=13)
        model_obj.load_state_dict(torch.load(f ,map_location='cpu'))
        app.logger.info(f"Loaded model {workspace}/{model_name}/{version}")

    return model_obj


@app.route("/logs", methods=["GET"])
def logs():
    """Reads data from the log file and returns them as the response"""

    with open(LOG_FILE) as f:
        response = {
            "logs": f.read()
        }

    return jsonify(response)


@app.route("/download_registry_model", methods=["POST"])
def download_registry_model():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/download_registry_model

    The comet API key should be retrieved from the ${COMET_API_KEY} environment variable.

    Recommend (but not required) json with the schema:

        {
            workspace: (required),
            model: (required),
            version: (required),
            ... (other fields if needed) ...
        }

    """
    # Get POST json data
    json = request.get_json()
    app.logger.info(json)
    app.logger.info(f'API KEY {os.environ["COMET_API_KEY"]}')

    workspace = json.get("workspace")
    model_name = json.get("model")
    version = json.get("version")

    try:
        app.model = load_model(workspace, model_name, version)
        response = {"result": "success"}

    except:
        response = {"result": "failed"}

    app.logger.info(response)
    return jsonify(response)  # response must be json serializable!


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/predict

    Returns predictions
    """

    if app.model is None:
        app.logger.error("Error, no model loaded. Please use /download_registry_model first")
        return {"predictions": "Error, no model loaded. Please use /download_registry_model first"}

    # Get POST json data
    image_json = json.loads(request.get_json())

    image_data = base64.b64decode(image_json.get("img"))
    image = Image.open(io.BytesIO(image_data))

    image = image.resize((400,400), Image.Resampling.LANCZOS)
    convert_tensor = transforms.ToTensor()
    features = convert_tensor(image)[:3,:,:]

    split_features = split_example(features)

    logits = app.model(split_features)
    preds = logits.argmax(dim=1)
    fen_pred = labels_to_fen(preds.reshape(8,8))

    response = {
        "predictions": fen_pred
    }

    app.logger.info(response)
    return jsonify(response)  # response must be json serializable!


@app.errorhandler(500)
def internal_server_error(e):
    return jsonify(error=500, text=str(e)), 500


with app.app_context():
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO)
    app.model = load_model("vincentlongpre", "basic-resnet18", "1.0.0",)
