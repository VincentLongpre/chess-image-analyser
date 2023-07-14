import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from chess_board import make_position
from serving_client import *
from pyperclip import copy

st.title("Chess Image Analysizer")

version_dict = {"basic-resnet18": ("1.0.0",), "augmented-resnet18": ("1.0.0", "1.1.0")}

lichess_editor_url = "https://lichess.org/editor/"

client = ServingClient(ip="0.0.0.0", port="8080")

if "cache" not in st.session_state:
    st.session_state.cache = {"pred_fen": None, "prev_file": None}


def load_model(workspace: str, model: str, version: str):
    """
    Make a model download request via the client.

    Args:
        workspace (str): The Comet ML workspace
        model (str): The model in the Comet ML registry to download
        version (str): The model version to download
    """

    response = client.download_registry_model(workspace, model, version)
    try:
        if response["result"] == "success":
            st.info(f"The {model} model was loaded successfully")
        else:
            st.info(
                f"There was an error during the loading of the {model} model. Please try again or change model"
            )
    except:
        st.info(f"{response['text']}")


def predict_fen(bytes_data):
    """
    Make a prediction request for the bytes of a given image via the client.

    Args:
        bytes_data (bytestring): Input image bytestring to submit to the prediction service.
    """

    response = client.predict(bytes_data)
    try:
        if not response["predictions"]:
            st.info(f"There was an error during the prediction of the FEN.")
    except:
        st.info(f"{response['text']}")
    return response["predictions"]


def make_fig(image_data):
    """
    Create figure from a PIL Image.

    Args:
        image_data (PIL Image): Input image bytestring to submit to the prediction service.
    """
    fig, ax = plt.subplots()
    ax.imshow(image_data)
    ax.set_axis_off()
    return fig


locals().update(st.session_state.cache)

with st.sidebar:
    workspace = st.sidebar.selectbox("workspace", ("vincentlongpre",))
    model = st.sidebar.selectbox("Model", ("basic-resnet18", "augmented-resnet18"))
    version = st.sidebar.selectbox("Version", version_dict[model])
    st.button("Get Model", on_click=load_model, args=(workspace, model, version))

with st.container():
    uploaded_file = st.file_uploader("Choose File")

    # If the uploaded file is modified
    if uploaded_file is not None and uploaded_file != prev_file:
        input_bytes = uploaded_file.getvalue()

        # Operations that take time are done using spinner animation
        with st.spinner("Analyzing Image..."):
            input_image = Image.open(BytesIO(input_bytes))
            pred_fen = predict_fen(input_bytes)
            pred_image = make_position(pred_fen)
            actual_fig = make_fig(input_image)
            pred_fig = make_fig(pred_image)

        st.session_state.cache.update(
            {
                "prev_file": uploaded_file,
                "pred_fen": pred_fen,
                "actual_fig": actual_fig,
                "pred_fig": pred_fig,
            }
        )

with st.container():
    if pred_fen:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                "<p style='text-align: center;'>Actual Position</p>",
                unsafe_allow_html=True,
            )
            st.pyplot(actual_fig)

        with col2:
            st.markdown(
                "<p style='text-align: center;'>Predicted Position</p>",
                unsafe_allow_html=True,
            )
            st.pyplot(pred_fig)

        col1, col2 = st.columns(2)

        with col1:
            color = st.selectbox("Your color", ("white", "black"))
        with col2:
            to_play = st.selectbox("Color to play", ("white", "black"))

        if color == "black":
            pred_fen = "-".join(pred_fen.split("-")[::-1])

        # Lichess redirection url construction
        option_ext = f"_{to_play[0]}_-_-_0_1?color={color}"
        fen_ext = "/".join(pred_fen.split("-"))
        lichess_position_url = lichess_editor_url + fen_ext + option_ext

        link = f"Link to Lichess [board editor]({lichess_position_url})"
        st.markdown(link, unsafe_allow_html=True)
