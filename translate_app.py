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
    default_model = st.secrets.get("openai_model", "gpt-5.1")
elif provider_name == "anthropic":
    api_key = st.secrets.get("anthropic_key", "")
    default_model = st.secrets.get("anthropic_model", "claude-sonnet-4.6")

model = st.text_input("Model", value=default_model)

if not api_key:
    st.error(f"No API key found for {provider_name}. Add it to .streamlit/secrets.toml")
    st.stop()

provider = create_provider(provider_name, api_key=api_key, model=model)
translator = ChapterTranslator(provider=provider)

# --- All provider configs for preview mode ---
ALL_PROVIDERS = {
    "openai": {
        "key_secret": "openai_key",
        "model_secret": "openai_model",
        "default_model": "gpt-5.1",
    },
    "anthropic": {
        "key_secret": "anthropic_key",
        "model_secret": "anthropic_model",
        "default_model": "claude-sonnet-4.6",
    },
}


def _get_chapter_label(idx: int, chapter_item) -> str:
    """Return a human-readable label: 'Chapter N: <heading or first line>'."""
    try:
        content = chapter_item.get_content().decode("utf-8", errors="ignore")
        soup = BeautifulSoup(content, "html.parser")
        heading = soup.find(["h1", "h2", "h3", "h4"])
        if heading:
            text = heading.get_text(strip=True)
            if text:
                return f"Chapter {idx + 1}: {text[:80]}"
        first_p = soup.find("p")
        if first_p:
            text = first_p.get_text(strip=True)
            if text:
                return f"Chapter {idx + 1}: {text[:80]}"
    except Exception:
        pass
    return f"Chapter {idx + 1}: ({chapter_item.get_name()})"

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

# --- Chapter multiselect with readable labels ---
chapter_labels = [_get_chapter_label(i, ch) for i, ch in enumerate(chapters)]

# Select all / deselect all button
select_all = st.button("Select / deselect all", use_container_width=True)
if select_all:
    if len(st.session_state.get("selected_labels", [])) == len(chapter_labels):
        st.session_state["selected_labels"] = []
    else:
        st.session_state["selected_labels"] = chapter_labels
    st.rerun()

selected_labels = st.multiselect(
    "Select chapters to translate",
    options=chapter_labels,
    default=st.session_state.get("selected_labels", []),
    key="chapter_multiselect",
    help="Each entry shows the chapter number and its heading or first line.",
)

# Update session state to track selection
st.session_state["selected_labels"] = selected_labels

selected_indices = [chapter_labels.index(lbl) for lbl in selected_labels]
selected_indices.sort()

if not selected_indices:
    st.warning("Select at least one chapter.")
    st.stop()

# --- Action buttons ---
col_start, col_preview = st.columns(2)
with col_start:
    start_button = st.button("**Start translation**", type="primary", use_container_width=True)
with col_preview:
    preview_button = st.button("Preview translation", use_container_width=True)

# Keep a place for logs/progress
progress_bar = st.progress(0)
status = st.empty()
chapter_status = st.empty()
glossary_container = st.empty()

# We'll store translated book in a temporary file and then offer download
output_temp_path = None

# ========================
# Preview translation mode
# ========================
if preview_button:
    PREVIEW_SECTIONS = 5
    first_idx = selected_indices[0]
    preview_chapter = chapters[first_idx]

    soup = BeautifulSoup(preview_chapter.get_content(), "html.parser")
    blocks = soup.find_all(["p", "li", "h1", "h2", "h3", "h4", "blockquote"])
    raw_sections = [tag.get_text(strip=True) for tag in blocks if tag.get_text(strip=True)]
    preview_sections = raw_sections[:PREVIEW_SECTIONS]

    if not preview_sections:
        st.warning("No translatable text found in the selected chapter.")
        st.stop()

    st.subheader(f"Preview – {chapter_labels[first_idx]}")
    st.caption(f"Translating {len(preview_sections)} section(s) with every available model.")

    # Build a provider for every configured key
    preview_providers = {}
    for pname, cfg in ALL_PROVIDERS.items():
        pkey = st.secrets.get(cfg["key_secret"], "")
        if not pkey:
            continue
        pmodel = st.secrets.get(cfg["model_secret"], cfg["default_model"])
        preview_providers[f"{pname} / {pmodel}"] = create_provider(pname, api_key=pkey, model=pmodel)

    if not preview_providers:
        st.error("No API keys configured. Add them to .streamlit/secrets.toml")
        st.stop()

    # Show original text
    with st.expander("Original text", expanded=True):
        for sec in preview_sections:
            st.markdown(f"- {sec}")

    # Translate with each provider and show side-by-side
    cols = st.columns(len(preview_providers))
    for col, (label, prov) in zip(cols, preview_providers.items()):
        with col:
            st.markdown(f"**{label}**")
            try:
                tmp_translator = ChapterTranslator(provider=prov)
                translated = tmp_translator._translate_batch(preview_sections)
                for orig, trans in zip(preview_sections, translated):
                    st.markdown(f"**→** {trans}")
                    st.caption(orig)
                    st.divider()
            except Exception as exc:
                st.error(f"Error: {exc}")

    st.stop()

# ========================
# Full translation mode
# ========================
if start_button:
    total_to_translate = len(selected_indices)
    count = 0

    status.info(f"Translating {total_to_translate} chapter(s) ...")
    for i in selected_indices:
        chapter = chapters[i]
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
        chapter_status.info(f"**Translated chapter {i + 1}** ({count}/{total_to_translate})")

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
