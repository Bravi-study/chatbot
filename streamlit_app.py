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

# Настройка страницы
st.set_page_config(
    page_title="Impressionist StyleGAN",
    layout="wide",
)

# URL-адрес архива с моделями
MODEL_URL = (
    "https://www.dropbox.com/scl/fi/jp9n3uyuzlfj7a9hap5o2/"
    "style_models.zip?rlkey=a6hkuwe135h64lidq0t1x3oja&dl=1"
)
PRETRAINED = "ffhq.pkl"
OUTPUT_SIZE = 650

# Словарь стилей с описанием
STYLES = {
    "IMPRESSIONISM": {
        "name": "Импрессионизм",
        "file": "photo_impressionist_portrait_trained_generator.pt",
    },
    "ANIME": {
        "name": "Аниме",
        "file": "photo_anime_trained_generator.pt",
    },
    "CUBISM": {
        "name": "Кубизм",
        "file": "photo_cubism_trained_generator.pt",
    },
}


def download_and_extract_models():
    """Загрузка и распаковка файлов моделей"""
    # Проверка наличия всех необходимых файлов
    required_files = [PRETRAINED] + [style["file"] for style in STYLES.values()]
    files_exist = all(os.path.exists(f) for f in required_files)

    if not files_exist:
        try:
            st.info("Загрузка моделей...")
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()

            with ZipFile(BytesIO(response.content)) as zip_file:
                zip_file.extractall()

            st.success("Models downloaded and extracted successfully")
        except Exception as e:
            st.error(f"Ошибка загрузки моделей: {str(e)}")
            return False
    return True


# Добавление пути к StyleGAN
if "/stylegan" not in sys.path:
    sys.path.extend(["/stylegan", "stylegan"])

# Определение устройства для вычислений
device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if torch.mps.is_available()
    else torch.device("cpu")
)

# Пользовательский CSS
st.markdown(
    """
    <style>
        .main {
            max-width: 800px;
            margin: 0 auto;
            padding: 0 20px;
        }
        .glory-text {
            position: fixed;
            left: 20px;
            bottom: 20px;
            z-index: 999;
        }
        .stSelectbox {
            margin-bottom: 2rem;
        }
        .css-1v0mbdj.etr89bj1 {
            display: flex;
            justify-content: center;
        }
    </style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_stylegan_generator():
    """Загрузка базовой модели StyleGAN"""
    try:
        with open("ffhq.pkl", "rb") as f:
            stylegan = pickle.load(f)
        generator = stylegan["G_ema"]
        generator.to(device)
        generator.eval()
        return generator
    except Exception as e:
        st.error(f"Ошибка загрузки базовой модели: {str(e)}")
        return None


@st.cache_resource
def load_model(model_file):
    """Загрузка обученной модели выбранного стиля"""
    try:
        model = load_stylegan_generator()
        if model is None:
            return None

        checkpoint = torch.load(model_file, map_location=device)
        model.load_state_dict(checkpoint["generator_state_dict"])
        return model
    except Exception as e:
        st.error(f"Ошибка загрузки модели стиля: {str(e)}")
        return None


# Основное приложение
if download_and_extract_models():
    try:
        # Выбор стиля в боковой панели
        with st.sidebar:
            style_names = {style["name"]: key for key, style in STYLES.items()}
            selected_style_name = st.selectbox(
                "Стиль портрета", options=list(style_names.keys()), key="style_select"
            )

        # Основной контент
        selected_style = STYLES[style_names[selected_style_name]]
        model = load_model(selected_style["file"])

        # Адаптация окончания для названия стиля
        ending = "а" if selected_style["name"] != "Аниме" else ""
        st.title(f"StyleGAN – генератор {selected_style_name.lower()}{ending}!")

        if model is None:
            st.warning("Не удалось загрузить модель!")
        else:
            if st.button("Создать портрет", key="generate_btn"):
                with st.spinner("Готовим шедевр..."):
                    # Генерация случайного вектора
                    z = torch.randn(1, 512, device=device)
                    with torch.no_grad():
                        # Генерация изображения
                        ws = model.mapping(z, None)
                        images = model.synthesis(ws, noise_mode="const")

                    # Изменение размера изображения
                    images = torch.nn.functional.interpolate(
                        images,
                        size=(OUTPUT_SIZE, OUTPUT_SIZE),
                        mode="bilinear",
                        align_corners=False,
                    )
                    # Преобразование тензора в изображение
                    image = images[0].permute(1, 2, 0).add(1).div(2).cpu().numpy()

                    # Отображение изображения по центру
                    st.image(
                        image,
                        clamp=True,
                        caption="Ну, как вам?",
                        width=OUTPUT_SIZE,
                    )

        # Текст внизу страницы
        st.markdown(
            '<p class="glory-text">For the glory of <a href="https://dls.samcs.ru/">DLS</a>!</p>',
            unsafe_allow_html=True,
        )

    except Exception as e:
        st.error(f"Произошла ошибка: {str(e)}")
