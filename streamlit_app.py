import streamlit as st

st.set_page_config(page_title="StyleGAN Image Generator", layout="wide")

import logging
import os
import pickle
import sys
from io import BytesIO
from zipfile import ZipFile

import requests
import torch

logging.basicConfig(level=logging.INFO)

MODEL_URL = "https://www.dropbox.com/s/p2fz1c8vxtkkb40/stylegan_models.zip?dl=1"
MODEL = "photo_impressionist_portrait_trained_generator.pt"
PRETRAINED = "ffhq.pkl"
OUTPUT_SIZE = 480


def download_and_extract_models():
    """Download and extract model files from zip"""
    if not (os.path.exists(MODEL) and os.path.exists(PRETRAINED)):
        try:
            st.info("Downloading model files...")
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()

            with ZipFile(BytesIO(response.content)) as zip_file:
                zip_file.extractall()

            st.success("Models downloaded and extracted successfully")
        except Exception as e:
            st.error(f"Error downloading models: {str(e)}")
            return False
    return True


if "/stylegan" not in sys.path:
    sys.path.extend(["/stylegan", "stylegan"])

# Determine device
device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if torch.mps.is_available()
    else torch.device("cpu")
)

# Custom CSS
st.markdown(
    """
    <style>
        .main {
            max-width: 800px;
            margin: 0 auto;
            padding: 0 20px;
        }
        .css-1v0mbdj.e115fcil1 {
            text-align: center;
        }
    </style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_stylegan_generator():
    try:
        with open("ffhq.pkl", "rb") as f:
            stylegan = pickle.load(f)
        generator = stylegan["G_ema"]
        generator.to(device)
        generator.eval()
        return generator
    except Exception as e:
        st.error(f"Error loading StyleGAN model: {str(e)}")
        return None


@st.cache_resource
def load_model():
    try:
        model = load_stylegan_generator()
        if model is None:
            return None

        checkpoint = torch.load(MODEL, map_location=device)
        model.load_state_dict(checkpoint["generator_state_dict"])
        return model
    except Exception as e:
        st.error(f"Error loading checkpoint: {str(e)}")
        return None


# Main app
if download_and_extract_models():
    try:
        model = load_model()

        with st.container():
            col1, col2, col3 = st.columns([1, 2, 1])

            with col2:
                st.title("StyleGAN Image Generator")

                if model is None:
                    st.warning("Model failed to load. Please check the model files.")
                else:
                    if st.button("Generate Image", key="generate_btn"):
                        with st.spinner("Generating image..."):
                            z = torch.randn(1, 512, device=device)
                            with torch.no_grad():
                                ws = model.mapping(z, None)
                                images = model.synthesis(ws, noise_mode="const")

                            images = torch.nn.functional.interpolate(
                                images,
                                size=(OUTPUT_SIZE, OUTPUT_SIZE),
                                mode="bilinear",
                                align_corners=False,
                            )
                            image = images[0].permute(1, 2, 0).add(1).div(2).cpu().numpy()

                            # Center the image display
                            st.image(
                                image,
                                clamp=True,
                                caption="Generated Image",
                                use_container_width=True,
                            )

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
