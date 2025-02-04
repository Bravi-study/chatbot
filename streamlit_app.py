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
MODEL_URL = "https://www.dropbox.com/s/bjk832dk5h7qe4h/style_models.zip?dl=1"
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
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Обновляем CSS
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
        with open("ffhq.pkl", "rb") as f:
            stylegan = pickle.load(f)
        # Получаем базовую модель
        generator = stylegan["G_ema"]
        # Копируем состояние на новое устройство
        state_dict = generator.state_dict()
        generator = type(generator)(*generator.init_args, **generator.init_kwargs)
        generator.to(device)
        generator.load_state_dict(state_dict)
        generator.eval()
        return generator
    except Exception as e:
        st.error(f"Ошибка загрузки базовой модели: {str(e)}")
        return None


def load_model(model_file: str):
    """Загрузка обученной модели выбранного стиля"""
    try:
        # Очищаем CUDA кэш перед загрузкой
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model = load_stylegan_generator()
        if model is None:
            return None

        # Загружаем чекпойнт напрямую в модель
        checkpoint = torch.load(model_file, map_location=device)
        model.load_state_dict(checkpoint["generator_state_dict"])
        model.eval()
        return model
    except Exception as e:
        st.error(f"Ошибка загрузки модели стиля {model_file}: {str(e)}")
        return None


def get_style_by_name(style_name: str) -> dict:
    """Получение стиля по русскому названию"""
    for style_data in STYLES.values():  # Изменено с keys() на items()
        if style_data["name"] == style_name:
            return style_data
    return None


# Основное приложение
if download_and_extract_models():
    try:
        # Выбор стиля в боковой панели
        with st.sidebar:
            st.markdown("<br>" * 3, unsafe_allow_html=True)
            style_names = [style["name"] for style in STYLES.values()]
            selected_style_name = st.selectbox(
                "Стиль портрета", options=style_names, key="style_select"
            )

            # Больше пространства перед надписью
            st.markdown(
                "<div style='flex-grow: 1; min-height: 50vh'></div>", unsafe_allow_html=True
            )

            # Обновленная надпись внизу сайдбара
            st.markdown(
                '<div class="sidebar-glory">Да славится <a href="https://dls.samcs.ru/">DLS</a>!</div>',
                unsafe_allow_html=True,
            )

        # Получаем данные выбранного стиля
        selected_style = get_style_by_name(selected_style_name)

        # Загружаем модель без кэширования
        model = load_model(selected_style["file"])

        # Адаптация окончания для названия стиля
        ending = "а" if selected_style_name != "Аниме" else ""
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

    except Exception as e:
        st.error(f"Произошла ошибка: {str(e)}")
