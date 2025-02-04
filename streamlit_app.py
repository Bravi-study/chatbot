import logging
import pickle
import sys

import streamlit as st
import torch

logging.basicConfig(level=logging.INFO)

OUTPUT_SIZE = 512

if "/stylegan" not in sys.path:
    sys.path.extend(["/stylegan", "stylegan"])

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

st.set_page_config(page_title="StyleGAN Image Generator", layout="wide")

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

        checkpoint = torch.load("imp_model.pt", map_location=device)
        model.load_state_dict(checkpoint["generator_state_dict"])
        return model
    except Exception as e:
        st.error(f"Error loading checkpoint: {str(e)}")
        return None


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
                            image, clamp=True, caption="Generated Image", use_container_width=True
                        )

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
