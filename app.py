import streamlit as st
import numpy as np
import pandas as pd
import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

mapping = ["cs", "stat", "math", "q-bio", "physics", "eess", "nlin", "q-fin", "econ"]
abbr_mapping = {
    "cs": "Computer Science",
    "stat": "Statistics",
    "math": "Mathematics",
    "q-bio": "Quantitative Biology",
    "physics": "Physics",
    "eess": "Electrical Engineering and Systems Science",
    "nlin": "Nonlinear Sciences",
    "q-fin": "Quantitative Finance",
    "econ": "Economics"
}

@st.cache_resource
def pipeline_getter():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")

    config = AutoConfig.from_pretrained("./config.json")
    model = AutoModelForSequenceClassification.from_config(config)

    state_dict = torch.load(
        "./model_weights.pth",
        map_location="cpu",
        weights_only=True
    )
    model.load_state_dict(state_dict)
    model.eval()

    return tokenizer, model


tokenizer, model = pipeline_getter()


def predict_article_categories_with_confidence(
    text_data,
    abstract_text=None,
    confidence_level=0.85,
    max_categories=9
):
    tokenized_input = tokenizer(
        text=text_data,
        text_pair=abstract_text,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        model_output = model(**tokenized_input)

    logits = model_output.logits
    probs = torch.sigmoid(logits).cpu().numpy().flatten()

    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    cumulative_probs = np.cumsum(sorted_probs)

    selected_indices = []
    for i, cum_prob in enumerate(cumulative_probs):
        if cum_prob >= confidence_level or i >= max_categories - 1:
            selected_indices = sorted_indices[:i + 1]
            break

    predicted_abbrs = [mapping[idx] for idx in selected_indices]
    top_abbr = mapping[sorted_indices[0]]

    result = {
        "probabilities": probs,
        "predicted_categories": [abbr_mapping[abbr] for abbr in predicted_abbrs],
        "confidence": cumulative_probs[len(selected_indices) - 1],
        "top_category": abbr_mapping[top_abbr],
        "top_category_prob": probs[sorted_indices[0]],
        "used_categories": len(selected_indices)
    }

    return result


def is_english_title(text):
    pattern = r"^[A-Za-z0-9\s.,:;!?()\-\"'\/&[\]{}_+=*%#$@`~^<>|\\]+$"
    return bool(re.fullmatch(pattern, text))


st.set_page_config(
    page_title="Классификатор arXiv",
    page_icon="📚",
    layout="centered"
)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #f8fbff 0%, #eef4ff 55%, #e8f1ff 100%);
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .hero-box {
        background: transparent;
        padding: 0 0 1rem 0;
        margin-bottom: 0.8rem;
        border: none;
        box-shadow: none;
    }

    .hero-title {
        font-size: 34px !important;
        font-weight: 800;
        color: #16324f;
        margin-bottom: 6px;
        line-height: 1.2;
    }

    .hero-subtitle {
        font-size: 16px;
        color: #5b6b7a;
        margin-bottom: 0;
    }

    .section-box {
        background: transparent;
        padding: 0;
        border: none;
        box-shadow: none;
        margin-bottom: 1rem;
    }

    .result-box {
        background: transparent;
        padding: 20px 0 0 0;
        border: none;
        box-shadow: none;
        margin-top: 10px;
    }

    .result-label {
        color: black;
        font-size: 16px;
        font-weight: 600;
        margin-bottom: 6px;
    }

    .category-badge {
        display: inline-block;
        background: linear-gradient(135deg, #1f77b4 0%, #3f8fd1 100%);
        color: white;
        padding: 8px 14px;
        margin: 6px 6px 0 0;
        border-radius: 999px;
        font-size: 14px;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(31, 119, 180, 0.18);
    }

    .top-badge {
        display: inline-block;
        background: linear-gradient(135deg, #0f4c81 0%, #1f77b4 100%);
        color: white;
        padding: 10px 16px;
        margin-top: 8px;
        border-radius: 999px;
        font-size: 15px;
        font-weight: 700;
        box-shadow: 0 3px 10px rgba(15, 76, 129, 0.22);
    }

    div.stButton > button {
        width: 100%;
        border-radius: 14px;
        padding: 0.75rem 1rem;
        font-size: 16px;
        font-weight: 700;
        border: none;
        background: linear-gradient(135deg, #1f77b4 0%, #3f8fd1 100%);
        color: white;
        box-shadow: 0 4px 14px rgba(31, 119, 180, 0.2);
    }

    div.stButton > button:hover {
        background: linear-gradient(135deg, #17689e 0%, #357fbe 100%);
        color: white;
    }

    .metric-line {
        font-size: 15px;
        color: #334155;
        margin-top: 10px;
    }

    div[data-baseweb="input"] {
        background-color: transparent !important;
        border: 1px solid #bfd5f0 !important;
        border-radius: 14px !important;
        box-shadow: none !important;
    }

    div[data-baseweb="input"] > div {
        background-color: transparent !important;
    }

    div[data-baseweb="input"] input {
        background-color: transparent !important;
        color: black !important;
        -webkit-text-fill-color: black !important;
    }

    div[data-baseweb="input"] input::placeholder {
        color: #9ca3af !important;
        opacity: 1 !important;
        -webkit-text-fill-color: #9ca3af !important;
    }

    div[data-baseweb="textarea"] {
        background-color: transparent !important;
        border: 1px solid #bfd5f0 !important;
        border-radius: 14px !important;
        box-shadow: none !important;
    }

    div[data-baseweb="textarea"] > div {
        background-color: transparent !important;
    }

    div[data-baseweb="textarea"] textarea {
        background-color: transparent !important;
        color: black !important;
        -webkit-text-fill-color: black !important;
    }

    div[data-baseweb="textarea"] textarea::placeholder {
        color: #9ca3af !important;
        opacity: 1 !important;
        -webkit-text-fill-color: #9ca3af !important;
    }

    .stTextInput, .stTextArea {
        background: transparent !important;
    }

    label {
        color: black !important;
        font-weight: 600 !important;
    }

    h3 {
        color: #16324f !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero-box">
        <div class="hero-title">📚 Классификатор статей с arXiv</div>
        <div class="hero-subtitle">
            Введите название и аннотацию статьи, чтобы определить наиболее вероятные категории, под которые она попадает. Названия и аннотация должны быть на английском.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="section-box">', unsafe_allow_html=True)

title_input = st.text_input(
    "Введите название статьи:",
    placeholder="например: A mixture model for aggregation of multiple pre-trained weak classifiers"
)

abstract_input = st.text_area(
    "Введите аннотацию статьи с arXiv:",
    placeholder="Вставьте сюда abstract для получения более точного результата...",
    height=170
)

st.markdown("</div>", unsafe_allow_html=True)

if st.button("Классифицировать статью"):
    if len(title_input.strip()) == 0:
        st.error("Нужно ввести хотя бы название статьи.")
    elif not is_english_title(title_input.strip()):
        st.error("Заголовок статьи должен содержать только английские символы.")
    else:
        with st.spinner("Анализирую содержание статьи..."):
            result = predict_article_categories_with_confidence(
                title_input,
                abstract_input if abstract_input else None,
            )

        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.subheader("Результаты")

        st.markdown(
            '<div class="result-label">Наиболее вероятная категория статьи:</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="top-badge">{result["top_category"]} '
            f'(p={result["top_category_prob"]:.3f})</div>',
            unsafe_allow_html=True
        )

        if len(result["predicted_categories"]) > 1:
            st.markdown(
                '<div class="result-label" style="margin-top: 14px;">Другие возможные категории:</div>',
                unsafe_allow_html=True
            )
            for category in result["predicted_categories"][1:]:
                st.markdown(
                    f'<div class="category-badge">{category}</div>',
                    unsafe_allow_html=True
                )