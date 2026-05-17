import os
import tempfile

import streamlit as st
from ebooklib import epub
import ebooklib
from bs4 import BeautifulSoup

from chapter_translator import ChapterTranslator
from llm_provider import create_provider

st.set_page_config(page_title="EPUB Chapter Translator", layout="centered")

st.title("EPUB Chapter Translator")

# --- Provider / model selection ---
provider_name = st.selectbox("LLM Provider", ["openai", "anthropic"])

if provider_name == "openai":
    api_key = st.secrets.get("openai_key", "")
    default_model = st.secrets.get("openai_model", "gpt-4o")
elif provider_name == "anthropic":
    api_key = st.secrets.get("anthropic_key", "")
    default_model = st.secrets.get("anthropic_model", "claude-sonnet-4-20250514")

model = st.text_input("Model", value=default_model)

if not api_key:
    st.error(f"No API key found for {provider_name}. Add it to .streamlit/secrets.toml")
    st.stop()

provider = create_provider(provider_name, api_key=api_key, model=model)
translator = ChapterTranslator(provider=provider)

# --- File upload ---
uploaded = st.file_uploader("Choose an EPUB file")

if uploaded is None:
    st.info("Upload an EPUB to begin.")
    st.stop()

# Validate that the uploaded file is actually an EPUB (EPUBs are ZIP files)
import zipfile
try:
    # Check if it's a valid ZIP file
    with zipfile.ZipFile(uploaded, 'r') as zip_ref:
        # EPUB must contain mimetype file at the start
        namelist = zip_ref.namelist()
        if 'mimetype' not in namelist:
            st.error("Invalid EPUB file: missing mimetype")
            st.stop()
        # Check mimetype content
        mimetype_content = zip_ref.read('mimetype').decode('utf-8')
        if mimetype_content != 'application/epub+zip':
            st.error("Invalid EPUB file: incorrect mimetype")
            st.stop()
except zipfile.BadZipFile:
    st.error("Invalid EPUB file: not a valid ZIP archive")
    st.stop()
except Exception as e:
    st.error(f"Error validating EPUB file: {e}")
    st.stop()

# Reset file pointer after reading
uploaded.seek(0)

# read uploaded bytes and write to a temp file (ebooklib reads from path)
orig_filename = uploaded.name
name_root, _ = os.path.splitext(orig_filename)

with tempfile.NamedTemporaryFile(delete=False, suffix=".epub") as tf:
    tf.write(uploaded.getvalue())
    temp_input_path = tf.name

# load book
book = epub.read_epub(temp_input_path)

# get chapter/document items, excluding navigation documents
all_items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
nav_item_names = set()

# Identify navigation items to skip
for item in all_items:
    if getattr(item, 'is_chapter', lambda: False)():
        continue
    # Check if item is a nav document by content
    try:
        content = item.get_content().decode("utf-8", errors="ignore")
        soup = BeautifulSoup(content, "html.parser")
        if soup.find("nav", attrs={"epub:type": "toc"}) or soup.find("nav", id="toc"):
            nav_item_names.add(item.get_name())
    except Exception:
        pass

# Also check by file name patterns common for nav/toc
for item in all_items:
    name = item.get_name().lower()
    if "nav" in name or "toc" in name:
        nav_item_names.add(item.get_name())

chapters = [item for item in all_items if item.get_name() not in nav_item_names]
n_chapters = len(chapters)

st.write(f"Found **{n_chapters}** translatable chapter(s) (skipping {len(nav_item_names)} navigation item(s)).")

# choose range / all
translate_all = st.checkbox("Translate all chapters", value=False)

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
chapter_status = st.empty()

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

        try:
            # Get raw sections before translation for glossary update
            soup = BeautifulSoup(chapter.content, "html.parser")
            blocks = soup.find_all(["p", "li", "h1", "h2", "h3", "h4", "blockquote"])
            raw_sections = [tag.get_text(strip=True) for tag in blocks]

            translator.translate(chapter)

            # Extract translated sections for glossary update
            soup_after = BeautifulSoup(chapter.content, "html.parser")
            blocks_after = soup_after.find_all(["p", "li", "h1", "h2", "h3", "h4", "blockquote"])
            # Every other block is the original (in brackets), so take even-indexed ones
            translated_texts = []
            for tag in blocks_after:
                text = tag.get_text(strip=True)
                if not text.startswith("[") or not text.endswith("]"):
                    translated_texts.append(text)

            # Update glossary for cross-chapter consistency
            translator.update_glossary(raw_sections, translated_texts)

            # Display current glossary in collapsible expander
            if translator.glossary:
                with glossary_container:
                    with st.expander(f"Glossary ({len(translator.glossary)} terms)"):
                        for term, translation in translator.glossary.items():
                            st.write(f"**{term}** → {translation}")

        except Exception as exc:
            st.error(f"Error translating chapter {i + 1}: {exc}")

        count += 1
        progress = int((count / total_to_translate) * 100)
        progress_bar.progress(progress)
        chapter_status.info(f"**Translated chapter {i + 1} of {e + 1}** ({count}/{total_to_translate})")

    # write out translated epub, preserving original navigation structure
    with tempfile.NamedTemporaryFile(delete=False, suffix=".epub") as out_tf:
        output_temp_path = out_tf.name

    epub.write_epub(output_temp_path, book, {"plugins": []})
    progress_bar.progress(100)
    st.success(f"Translation finished: {count} chapter(s) translated.")

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
