import gc
import logging
import os
import pickle
import sys
from io import BytesIO
from zipfile import ZipFile

import requests
import streamlit as st
import torch

logging.basicConfig(level=logging.INFO)

if "current_style" not in st.session_state:
    st.session_state.current_style = None
if "generator" not in st.session_state:
    st.session_state.generator = None

st.set_page_config(
    page_title="Impressionist StyleGAN",
    layout="wide",
)

# URL-адрес архива с моделями
MODEL_URL = "https://www.dropbox.com/s/mb2meuylasr23k0/stylegan_models.zip?dl=1"
PRETRAINED = "ffhq.pkl"
OUTPUT_SIZE = 650

# Словарь стилей с описанием
STYLES = {
    "Импрессионизм": {
        "file": "photo_impressionist_portrait_trained_generator.pt",
        "generated_name": "импрессионизма",
    },
    "Аниме": {
        "file": "photo_anime_trained_generator.pt",
        "generated_name": "аниме",
    },
    "Кубизм": {
        "file": "photo_cubism_trained_generator.pt",
        "generated_name": "кубизма",
    },
    "Масло": {
        "file": "photo_oil_painting_trained_generator.pt",
        "generated_name": "масляной живописи",
    },
}


def download_and_extract_models():
    """Загрузка и распаковка файлов моделей"""
    style_files = [style["file"] for style in STYLES.values()]
    files_exist = all(os.path.exists(f) for f in style_files)

    if files_exist:
        return True
    else:
        try:
            st.info("Загрузка моделей...")
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()

            # Save zip file temporarily
            zip_path = "stylegan_models.zip"
            with open(zip_path, "wb") as f:
                f.write(response.content)

            # Extract and then clean up
            with ZipFile(zip_path) as zip_file:
                zip_file.extractall()

            # Clean up temporary files
            os.remove(zip_path)

            st.success("Models downloaded and extracted successfully")
        except Exception as e:
            st.error(f"Ошибка загрузки моделей: {str(e)}")
            return False


if "/stylegan" not in sys.path:
    sys.path.extend(["/stylegan", "stylegan"])

# Определение устройства для вычислений
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

st.markdown(
    """
    <style>
        .main {
            max-width: 800px;
            margin: 0 auto;
            padding: 0 20px;
        }
        .stSelectbox {
            margin-bottom: 2rem;
        }
        .css-1v0mbdj.etr89bj1 {
            display: flex;
            justify-content: center;
        }
        .sidebar-glory {
            position: fixed;
            bottom: 16px;
            left: 0;
            width: 100%;
            max-width: 17rem;
            text-align: center;
            padding: 1rem;
        }
    </style>
""",
    unsafe_allow_html=True,
)


def load_stylegan_generator():
    """Загрузка базовой модели StyleGAN"""
    try:
        if not st.session_state.generator:
            with open("ffhq.pkl", "rb") as f:
                stylegan = pickle.load(f)
            generator = stylegan["G_ema"]
            generator.to(device)
            generator.eval()
            st.session_state.generator = generator

        return st.session_state.generator

    except Exception as e:
        st.error(f"Ошибка загрузки базовой модели: {str(e)}")
        return None


def load_model(model_file: str):
    """Загрузка обученной модели выбранного стиля"""
    try:
        generator = st.session_state.generator or load_stylegan_generator()
        checkpoint = torch.load(model_file, map_location=device)
        generator.load_state_dict(checkpoint["generator_state_dict"])
        generator.eval()

    except Exception as e:
        st.error(f"Ошибка загрузки модели стиля {model_file}: {str(e)}")
        return None


def generate_image(device):
    """Выделенная функция генерации изображения для лучшего управления памятью"""
    try:
        generator = st.session_state.generator
        z = torch.randn(1, 512, device=device)
        with torch.no_grad():
            ws = generator.mapping(z, None)
            images = generator.synthesis(ws, noise_mode="const")

        # Сразу очищаем промежуточные тензоры
        del z, ws
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        images = torch.nn.functional.interpolate(
            images,
            size=(OUTPUT_SIZE, OUTPUT_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        image = images[0].permute(1, 2, 0).add(1).div(2).cpu().numpy()

        # Очищаем оставшиеся тензоры
        del images
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return image

    except Exception as e:
        st.error(f"Ошибка генерации изображения: {str(e)}")
        return None


# Основное приложение
if download_and_extract_models():
    try:
        # Выбор стиля в боковой панели
        with st.sidebar:
            st.markdown("<br>" * 3, unsafe_allow_html=True)
            selected_style_name = st.selectbox(
                "Стиль портрета", options=list(STYLES.keys()), key="style_select"
            )
            st.markdown(
                "<div style='flex-grow: 1; min-height: 50vh'></div>", unsafe_allow_html=True
            )
            st.markdown(
                '<div class="sidebar-glory">Да славится <a href="https://dls.samcs.ru/">DLS</a>!</div>',
                unsafe_allow_html=True,
            )

        selected_style = STYLES[selected_style_name]

        if st.session_state.current_style != selected_style_name:
            st.session_state.current_style = selected_style_name
            load_model(selected_style["file"])

        st.title(f"StyleGAN – генератор {selected_style['generated_name']}!")

        if st.session_state.generator:
            if st.button("Создать портрет", key="generate_btn"):
                with st.spinner("Готовим шедевр..."):
                    if (image := generate_image(device)) is not None:
                        st.image(
                            image,
                            clamp=True,
                            caption="Ну, как вам?",
                            width=OUTPUT_SIZE,
                        )
        else:
            st.warning("Не удалось загрузить модель!")

    except Exception as e:
        st.error(f"Произошла ошибка: {str(e)}")
