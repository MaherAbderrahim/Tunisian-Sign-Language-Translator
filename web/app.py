"""
Streamlit Web App for TSL Translator
Supports: drag & drop images, video upload, camera, text translation
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
from pathlib import Path
import time

from src.translation.translator import TSLTranslator, translate_text_to_tsl
from src.languages.dictionary import TSLDictionary


def init_translator():
    """Initialize translator with caching"""
    @st.cache_resource
    def load_translator():
        return TSLTranslator()
    return load_translator()


def display_translation_result(result, confidence_threshold=0.5):
    """Display translation results in a nice format"""
    if "error" in result:
        st.error(f"No hands detected in the image")
        return

    confidence = result.get('confidence', 0)

    st.markdown("---")

    # Main translation result
    col_tsl, col_tun = st.columns(2)
    with col_tsl:
        st.metric("Sign (TSL)", result.get('tsl', 'N/A'))
    with col_tun:
        st.metric("Tunisian Dialect", result.get('tunisian', 'N/A'))

    # Other languages
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("French", result.get('french', 'N/A'))
    with col2:
        st.metric("English", result.get('english', 'N/A'))
    with col3:
        st.metric("Arabic", result.get('arabic', 'N/A'))
    with col4:
        color = "green" if confidence >= confidence_threshold else "red"
        st.metric("Confidence", f"{confidence:.1%}")


def process_image(image, translator, confidence_threshold=0.5):
    """Process a single image and display results"""
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    result = translator.translate_frame(frame)
    display_translation_result(result, confidence_threshold)
    return result


def process_video_frames(video_path, translator, confidence_threshold=0.5,
                          target_lang="en", extra_langs=None):
    """Process video and assemble continuous sentences via streaming recognizer.

    Returns the list of finalized sentences with multi-language translations.
    """
    if extra_langs is None:
        extra_langs = ["en", "fr", "ar"]
    sentences = translator.translate_video_file(
        video_path,
        target_lang=target_lang,
        extra_langs=extra_langs,
        sample_every_n_frames=3,
    )
    return sentences


def main():
    """Main Streamlit app"""

    st.set_page_config(
        page_title="TSL Translator",
        page_icon="",
        layout="wide"
    )

    st.title("TSL - Tunisian Sign Language Translator")
    st.markdown("Traduction de la Langue des Signes Tunisienne vers le dialecte tunisien et autres langues")
    st.markdown("---")

    # Sidebar settings
    st.sidebar.header("Settings")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05
    )

    # Initialize
    translator = init_translator()
    dictionary = TSLDictionary()

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Drop Image",
        "Upload Video",
        "Camera",
        "Text Translation",
        "Dictionary"
    ])

    # ========== TAB 1: DROP IMAGE ==========
    with tab1:
        st.header("Drag & Drop an Image")
        st.markdown("Upload a photo of a TSL sign gesture for translation")

        uploaded_files = st.file_uploader(
            "Choose image(s)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="image_uploader"
        )

        if uploaded_files:
            st.markdown(f"**{len(uploaded_files)} image(s) uploaded**")

            for i, uploaded_file in enumerate(uploaded_files):
                st.markdown(f"### Image {i+1}: {uploaded_file.name}")

                col_img, col_result = st.columns([1, 1])

                with col_img:
                    image = Image.open(uploaded_file)
                    st.image(image, caption=uploaded_file.name, use_container_width=True)

                with col_result:
                    process_image(image, translator, confidence_threshold)

                st.markdown("---")

    # ========== TAB 2: UPLOAD VIDEO ==========
    with tab2:
        st.header("Upload a Video — Continuous Recognition")
        st.markdown("Stream-friendly: signs are recognized continuously and assembled into sentences via the streaming recognizer.")

        uploaded_video = st.file_uploader(
            "Choose video", type=["mp4", "avi", "mov", "mkv"], key="video_uploader"
        )

        col_tl, col_el = st.columns(2)
        with col_tl:
            target_lang = st.selectbox("Sentence output language",
                                        ["en", "fr", "ar", "tunisian"], index=0)
        with col_el:
            extra_langs_str = st.text_input("Also translate sentences to",
                                              "en,fr,ar")
        extra_langs = [l.strip() for l in extra_langs_str.split(",") if l.strip()]

        if uploaded_video:
            tdir = tempfile.mkdtemp()
            tpath = Path(tdir) / uploaded_video.name
            tpath.write_bytes(uploaded_video.read())
            st.video(str(tpath))

            if st.button("Process Video (continuous)", key="process_video"):
                with st.spinner("Recognizing signs and assembling sentences..."):
                    sentences = process_video_frames(
                        str(tpath), translator, confidence_threshold,
                        target_lang=target_lang, extra_langs=extra_langs,
                    )

                if sentences:
                    st.success(f"Assembled {len(sentences)} sentence(s)")
                    for i, s in enumerate(sentences, 1):
                        st.markdown(f"### Sentence {i}")
                        signs_str = ", ".join(s.get("signs", []) or ["?"])
                        st.markdown(f"**Signs:** {signs_str}")
                        st.markdown(f"**Sentence ({target_lang}):** {s['sentence']}")
                        if s.get("translations"):
                            st.markdown("**Translations:**")
                            for lang, txt in s["translations"].items():
                                st.markdown(f"- `{lang}` → {txt}")
                        st.markdown("---")
                else:
                    st.warning("No sentences detected (try lowering the confidence threshold in the sidebar)")

    # ========== TAB 3: CAMERA ==========
    with tab3:
        st.header("Camera Translation")
        st.markdown("Use your webcam to capture a TSL sign in real-time")

        camera_input = st.camera_input("Take a picture of a TSL sign")

        if camera_input:
            image = Image.open(camera_input)
            process_image(image, translator, confidence_threshold)

    # ========== TAB 4: TEXT TRANSLATION ==========
    with tab4:
        st.header("Text Translation + Multi-language")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Tunisian / Text -> TSL")
            tunisian_input = st.text_input(
                "Enter text in Tunisian dialect",
                placeholder="e.g., labes, non, ab, dar..."
            )

            if st.button("Translate to TSL"):
                if tunisian_input:
                    tsl_label = translate_text_to_tsl(tunisian_input)
                    if tsl_label:
                        result = dictionary.get_all_translations(tsl_label)
                        st.success(f"**TSL Label:** {tsl_label}")
                        st.write(f"**Tunisian:** {result.get('tunisian')}")
                        st.write(f"**French:** {result.get('french')}")
                        st.write(f"**English:** {result.get('english')}")
                        st.write(f"**Arabic (dialect):** {result.get('arabic')}")
                        st.write(f"**Arabic (MSA):** {result.get('arabic_msa')}")
                    else:
                        st.warning("No TSL sign found for this text")

        with col2:
            st.subheader("TSL Label -> All Languages (incl. MSA)")
            tsl_input = st.text_input(
                "Enter TSL label",
                placeholder="e.g., labes, non, bou, dar..."
            )

            if st.button("Translate from TSL"):
                if tsl_input:
                    result = dictionary.get_all_translations(tsl_input)
                    if result:
                        st.metric("TSL", result['tsl'])
                        st.metric("Tunisian", result['tunisian'])
                        st.metric("French", result['french'])
                        st.metric("English", result['english'])
                        st.metric("Arabic (dialect)", result.get('arabic', 'N/A'))
                        st.metric("Arabic (MSA)", result.get('arabic_msa', 'N/A'))
                    else:
                        st.warning("No translation found for this label")

        # Multi-language text translation (via offline Argos)
        st.markdown("---")
        st.subheader("Translate any text to multiple languages (offline)")
        from src.translation.translator_backend import OfflineTranslator
        argos = OfflineTranslator()

        with st.form("multi_translate_form"):
            mt_text = st.text_input("Source text", placeholder="e.g., bonjour, hello, مرحبا")
            mt_src = st.selectbox("Source language",
                                     ["french", "english", "arabic", "tunisian",
                                      "spanish", "german"])
            mt_dst = st.multiselect("Target languages",
                                     ["english", "french", "arabic", "spanish",
                                      "german"],
                                     default=["english", "arabic"])
            submitted = st.form_submit_button("Translate")
            if submitted and mt_text:
                for lang in mt_dst:
                    out = argos.translate(mt_text, mt_src, lang)
                    st.write(f"**{lang}** → {out}")

    # ========== TAB 5: DICTIONARY ==========
    with tab5:
        st.header("TSL Dictionary")

        # Category filter
        category = st.selectbox(
            "Filter by Category",
            ["All", "Demandes (24)", "Destinations (8)", "Famille (12)",
             "Jours (7)", "Transport (6)"]
        )

        # Get labels
        labels = dictionary.get_all_labels()

        st.markdown(f"### {len(labels)} Signs Available")

        # Build table
        import pandas as pd
        data = []
        for label in labels:
            translations = dictionary.get_all_translations(label)
            data.append({
                'TSL Label': label,
                'Tunisian': translations.get('tunisian', 'N/A'),
                'French': translations.get('french', 'N/A'),
                'English': translations.get('english', 'N/A'),
                'Arabic': translations.get('arabic', 'N/A')
            })

        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, height=400)

        # Statistics
        st.markdown("---")
        st.markdown("### Statistics")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total", len(labels))
        c2.metric("Demandes", 24)
        c3.metric("Destinations", 8)
        c4.metric("Famille", 12)
        c5.metric("Jours+Transport", 13)


if __name__ == "__main__":
    main()
