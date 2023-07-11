import json
import base64
import requests
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)


class ServingClient:
    def __init__(self, ip: str = "0.0.0.0", port: int = 5000, features=None):
        self.base_url = f"http://{ip}:{port}"
        logger.info(f"Initializing client; base URL: {self.base_url}")

    def predict(self, file_bytes) -> json:
        """
        Formats the inputs into an appropriate payload for a POST request, and queries the
        prediction service. Retrieves the response from the server, and processes it back into a
        dataframe that corresponds index-wise to the input dataframe.

        Args:
            file_bytes (bytestring): Input image bytestring to submit to the prediction service.
        """

        data = {}
        data["img"] = base64.encodebytes(file_bytes).decode("utf-8")

        return requests.post(f"{self.base_url}/predict", json=json.dumps(data)).json()

    def logs(self) -> dict:
        """Get server logs"""

        return requests.get(f"{self.base_url}/logs").json()

    def download_registry_model(self, workspace: str, model: str, version: str) -> dict:
        """
        Triggers a "model swap" in the service; the workspace, model, and model version are
        specified and the service looks for this model in the model registry and tries to
        download it.

        See more here:

            https://www.comet.ml/docs/python-sdk/API/#apidownload_registry_model

        Args:
            workspace (str): The Comet ML workspace
            model (str): The model in the Comet ML registry to download
            version (str): The model version to download
        """

        return requests.post(
            f"{self.base_url}/download_registry_model",
            json={
                "workspace": workspace,
                "model": model,
                "version": version,
            },
        ).json()


if __name__ == "__main__":
    # Tests, make sure the client is running on 127.0.0.1:8080 (your local machine)
    # If this does not work, make sure that your docker container has either --network=host or -p 8080:8080
    client = ServingClient(ip="0.0.0.0", port="8080")

    # Testing logs
    # print(client.logs())

    # test data
    test_data_path = "dataset/ood"
    file_name = "2b5-p2NBp1p-1bp1nPPr-3P4-2pRnr1P-1k1B1Ppp-1P1P1pQP-Rq1N3K.jpeg"

    with open(os.path.join(test_data_path, file_name), mode="rb") as file:
        image_bytes = file.read()

    # Testing predictions
    first_preds = client.predict(image_bytes)
    second_preds = client.predict(image_bytes)
    assert first_preds == second_preds, "Predictions changed between two API calls"
    print(first_preds)

    # Loading a different model
    print(
        client.download_registry_model("vincentlongpre", "augmented-resnet18", "1.0.0")
    )

    # Testing predictions
    third_preds = client.predict(image_bytes)
    assert first_preds != third_preds, "Predictions did not change with model change"
    print(third_preds)

    # Testing logs (with data)
    print(client.logs())
