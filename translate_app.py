import os
import tempfile

import streamlit as st
from ebooklib import epub

from chapter_translator import ChapterTranslator

translator = ChapterTranslator(model="gpt-4.1", api_key=st.secrets['openai_key'])

st.set_page_config(page_title="EPUB Chapter Translator", layout="centered")

st.title("EPUB Chapter Translator")

uploaded = st.file_uploader("Choose an EPUB file", type=["epub"])

if uploaded is None:
    st.info("Upload an EPUB to begin.")
    st.stop()

# read uploaded bytes and write to a temp file (ebooklib reads from path)
orig_filename = uploaded.name
name_root, _ = os.path.splitext(orig_filename)

with tempfile.NamedTemporaryFile(delete=False, suffix=".epub") as tf:
    tf.write(uploaded.getvalue())
    temp_input_path = tf.name

# load book
book = epub.read_epub(temp_input_path)

# get chapter/document items
chapters = list(book.get_items_of_type(epub.ITEM_DOCUMENT))
n_chapters = len(chapters)

st.write(f"Found **{n_chapters}** chapter(s).")

# choose range / all
translate_all = st.checkbox("Translate all chapters", value=True)

start_idx = 1
end_idx = n_chapters

if not translate_all:
    col1, col2 = st.columns(2)
    with col1:
        start_idx = st.number_input(
            "Start chapter (1-based)", min_value=1, max_value=n_chapters, value=1, step=1
        )
    with col2:
        end_idx = st.number_input(
            "End chapter (1-based)", min_value=1, max_value=n_chapters, value=n_chapters, step=1
        )

    if start_idx > end_idx:
        st.error("Start must be <= End.")
        st.stop()

# button to start translation
start_button = st.button("Start translation")

# Keep a place for logs/progress
progress_bar = st.progress(0)
status = st.empty()

# We'll store translated book in a temporary file and then offer download
output_temp_path = None

if start_button:
    # convert to 0-based indices
    s = int(start_idx) - 1
    e = int(end_idx) - 1

    total_to_translate = (e - s + 1)
    count = 0

    status.info(f"Translating chapters {s + 1}–{e + 1} ...")
    for i, chapter in enumerate(chapters):
        if i < s or i > e:
            continue

        # Call translator.translate(chapter)
        # The translator may:
        #  - return a str/bytes containing the new HTML/content
        #  - modify the chapter object in place and return None
        try:
            result = translator.translate(chapter)
        except Exception as exc:
            # don't stop the whole process; record and continue
            status.error(f"Error translating chapter {i + 1}: {exc}")
            result = None

        # If a result is returned, try to set the chapter content
        if result is not None:
            # ensure bytes
            if isinstance(result, str):
                new_bytes = result.encode("utf-8")
            else:
                new_bytes = result

            # attempt to set content in a few possible ways
            if hasattr(chapter, "set_content"):
                try:
                    chapter.set_content(new_bytes)
                except Exception:
                    setattr(chapter, "content", new_bytes)
            elif hasattr(chapter, "content"):
                try:
                    chapter.content = new_bytes
                except Exception:
                    setattr(chapter, "content", new_bytes)
            else:
                setattr(chapter, "content", new_bytes)

        # else assume translation mutated chapter in place

        count += 1
        progress = int((count / total_to_translate) * 100)
        progress_bar.progress(progress)
        status.write(f"Translated chapter {i + 1} ({count}/{total_to_translate})")

    # write out translated epub
    with tempfile.NamedTemporaryFile(delete=False, suffix=".epub") as out_tf:
        output_temp_path = out_tf.name

    epub.write_epub(output_temp_path, book)
    progress_bar.progress(100)
    status.success(f"Translation finished: {count} chapter(s) translated.")

    # prepare download
    # default filename: {originalName}_de.epub
    default_out_name = f"{name_root}_de.epub"
    with open(output_temp_path, "rb") as f:
        out_bytes = f.read()

    st.download_button(
        label="Download translated EPUB",
        data=out_bytes,
        file_name=default_out_name,
        mime="application/epub+zip",
    )

    # cleanup temp input file
    try:
        os.remove(temp_input_path)
    except Exception:
        pass
