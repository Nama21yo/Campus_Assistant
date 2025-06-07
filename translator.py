#TODO: To be Implemented

import streamlit as st
from transformers import pipeline
MODEL_NAME = "facebook/nllb-200-3.3B"

@st.cache_resource
def get_translation_pipeline():
    """
    Loads and caches the translation model pipeline.
    This is a heavy object, so caching is essential.
    """
    print("Translator: Loading NLLB translation model... (This may take a while on first run)")
    try:
        translator_pipe = pipeline("translation", model=MODEL_NAME)
        print("Translator: Model loaded successfully.")
        return translator_pipe
    except Exception as e:
        st.error(f"Failed to load the translation model. Please check your internet connection and dependencies. Error: {e}")
        return None

def translate_amharic_to_english(text: str) -> str:
    """Translates a string from Amharic to English."""
    translator_pipe = get_translation_pipeline()
    if not translator_pipe:
        return "Translation service is unavailable."
    
    result = translator_pipe(text, src_lang="amh_Ethi", tgt_lang="eng_Latn")
    return result[0]['translation_text']

def translate_english_to_amharic(text: str) -> str:
    """Translates a string from English to Amharic."""
    translator_pipe = get_translation_pipeline()
    if not translator_pipe:
        return "የትርጉም አገልግሎቱ አይገኝም።"
    
    # Swap the language codes for reverse translation
    result = translator_pipe(text, src_lang="eng_Latn", tgt_lang="amh_Ethi")
    return result[0]['translation_text']
