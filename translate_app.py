import os
import tempfile
import base64
import hashlib
import json

import streamlit as st
import streamlit.components.v1 as components
from ebooklib import epub
import ebooklib
from bs4 import BeautifulSoup

from chapter_translator import ChapterTranslator
from llm_provider import create_provider

st.set_page_config(page_title="EPUB Chapter Translator", layout="centered")

# --- Version (update with each commit to verify deployment) ---
APP_VERSION = "1.3.1"

# --- Session state ---
for _key, _default in [
    ("translation_cache", {}),
    ("output_epub_bytes", None),
    ("output_filename", None),
]:
    if _key not in st.session_state:
        st.session_state[_key] = _default

# --- Disk cache (survives full session loss / page reload on mobile) ---
CACHE_DIR = os.path.join(tempfile.gettempdir(), "epub_translator_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def _disk_cache_path(file_hash, chapter_idx, prov, mdl):
    h = hashlib.sha256(f"{file_hash}:{chapter_idx}:{prov}:{mdl}".encode()).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"{h}.json")


def _load_cached(file_hash, chapter_idx, prov, mdl):
    p = _disk_cache_path(file_hash, chapter_idx, prov, mdl)
    if os.path.exists(p):
        try:
            with open(p) as f:
                data = json.load(f)
            return data["raw"], data["translated"]
        except Exception:
            return None
    return None


def _save_cached(file_hash, chapter_idx, prov, mdl, raw, translated):
    p = _disk_cache_path(file_hash, chapter_idx, prov, mdl)
    with open(p, "w") as f:
        json.dump({"raw": raw, "translated": translated}, f, ensure_ascii=False)


def _raw_response_path(file_hash, chapter_idx, prov, mdl):
    h = hashlib.sha256(f"{file_hash}:{chapter_idx}:{prov}:{mdl}".encode()).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"{h}_raw.txt")


def _save_raw_responses(file_hash, chapter_idx, prov, mdl, responses):
    p = _raw_response_path(file_hash, chapter_idx, prov, mdl)
    with open(p, "w") as f:
        f.write("\n===BATCH===\n".join(responses))


def _load_raw_responses(file_hash, chapter_idx, prov, mdl):
    p = _raw_response_path(file_hash, chapter_idx, prov, mdl)
    if os.path.exists(p):
        with open(p) as f:
            return f.read()
    return None


def _auto_download(data: bytes, filename: str, mime: str):
    """Best-effort automatic download via JS."""
    if len(data) > 50 * 1024 * 1024:  # skip for files > 50 MB
        return
    b64 = base64.b64encode(data).decode()
    components.html(
        f'<script>'
        f'try{{'
        f'var r=atob("{b64}");var a=new Uint8Array(r.length);'
        f'for(var i=0;i<r.length;i++)a[i]=r.charCodeAt(i);'
        f'var b=new Blob([a],{{type:"{mime}"}});var u=URL.createObjectURL(b);'
        f'var l=document.createElement("a");l.href=u;l.download="{filename}";'
        f'document.body.appendChild(l);l.click();'
        f'setTimeout(function(){{URL.revokeObjectURL(u)}},5000);'
        f'}}catch(e){{console.log("auto-dl failed",e)}}'
        f'</script>',
        height=0,
    )


def _build_raw_text(cache_entries):
    """Build a plain-text version of all cached translations."""
    lines = []
    for ch_idx, (raw, translated) in sorted(cache_entries.items()):
        lines.append(f"=== Chapter {ch_idx + 1} ===")
        for orig, trans in zip(raw, translated):
            lines.append(f"{trans}")
            lines.append(f"[{orig}]")
            lines.append("")
        lines.append("")
    return "\n".join(lines)


st.title("EPUB Chapter Translator")
st.caption(f"v{APP_VERSION}")
_cache_btn_slot = st.empty()

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
translator = ChapterTranslator(provider=provider, max_tokens=30000)

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

# Compute file hash for cache keying
file_hash = hashlib.md5(uploaded.getvalue()).hexdigest()

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

st.write(f"Found **{n_chapters}** translatable chapter(s).")

# --- Fill cache download button in header ---
_all_cache_text_parts = []
for _i in range(n_chapters):
    _cached = _load_cached(file_hash, _i, provider_name, model)
    if _cached:
        _r, _t = _cached
        _all_cache_text_parts.append(f"=== Chapter {_i + 1} ===\n" + "\n".join(
            f"{t}\n[{o}]\n" for o, t in zip(_r, _t)
        ))
    else:
        _raw_resp = _load_raw_responses(file_hash, _i, provider_name, model)
        if _raw_resp:
            _all_cache_text_parts.append(f"=== Chapter {_i + 1} (raw LLM) ===\n{_raw_resp}\n")

if _all_cache_text_parts:
    _cache_btn_slot.download_button(
        label="💾",
        data="\n".join(_all_cache_text_parts).encode("utf-8"),
        file_name=f"{name_root}_cache.txt",
        mime="text/plain",
        help="Download cached translations",
        key="cache_dl_header",
    )
else:
    _cache_btn_slot.download_button(
        label="💾",
        data="",
        file_name="empty",
        mime="text/plain",
        help="No cached translations yet",
        disabled=True,
        key="cache_dl_header",
    )

# --- Chapter multiselect with readable labels ---
chapter_labels = [_get_chapter_label(i, ch) for i, ch in enumerate(chapters)]

selected_labels = st.multiselect(
    "Select chapters to translate",
    options=chapter_labels,
    default=[],
    help="Each entry shows the chapter number and its heading or first line.",
)

selected_indices = [chapter_labels.index(lbl) for lbl in selected_labels]
selected_indices.sort()

if not selected_indices:
    st.warning("Select at least one chapter.")
    st.stop()

# --- Show cache status ---
_cached_count = sum(
    1 for i in selected_indices
    if _load_cached(file_hash, i, provider_name, model) is not None
)
if _cached_count:
    st.info(f"{_cached_count}/{len(selected_indices)} chapter(s) found in cache. No LLM cost for these.")

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
    blocks = soup.find_all(["p", "li", "blockquote"])
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
            except Exception as exc:
                st.error(f"Error: {exc}")

    st.stop()

# ========================
# Full translation mode
# ========================
if start_button:
    total_to_translate = len(selected_indices)
    count = 0
    all_translations = {}  # {chapter_idx: (raw, translated)}
    
    # Track batches for accurate progress
    _total_batches_completed = [0]
    _total_batches = [0]

    status.info(f"Translating {total_to_translate} chapter(s) ...")
    for i in selected_indices:
        chapter = chapters[i]

        # Save raw LLM response to disk THE INSTANT it arrives (before parsing)
        _resp_acc = []
        def _on_resp(resp, _idx=i, _acc=_resp_acc):
            _acc.append(resp)
            _save_raw_responses(file_hash, _idx, provider_name, model, _acc)
        translator.on_response = _on_resp
        
        # Batch progress callback
        def _on_batch(current, total, _chapter_idx=i):
            _total_batches_completed[0] += 1
            if _total_batches[0] > 0:
                pct = int((_total_batches_completed[0] / _total_batches[0]) * 100)
                progress_bar.progress(pct)
                chapter_status.info(
                    f"**Chapter {_chapter_idx + 1}**: batch {current}/{total} "
                    f"({_total_batches_completed[0]}/{_total_batches[0]} total batches)"
                )
        translator.on_batch_progress = _on_batch

        try:
            cached = _load_cached(file_hash, i, provider_name, model)
            if cached:
                raw_sections, translated_sections = cached
                translator.apply_cached(chapter, raw_sections, translated_sections)
                chapter_status.info(
                    f"**Chapter {i + 1}** loaded from cache ({count + 1}/{total_to_translate})"
                )
            else:
                # Get raw sections from chapter
                soup = BeautifulSoup(chapter.content, "html.parser")
                raw_sections = []
                for tag in soup.find_all(["p", "li", "h1", "h2", "h3", "h4", "blockquote"]):
                    raw_sections.append(tag.get_text(strip=True))

                # Calculate batches for this chapter to update total
                _batches = translator._make_batches(raw_sections)
                _total_batches[0] += len(_batches)

                # Translate only (without applying to soup) and cache immediately
                translated_sections = translator.translate_only(raw_sections)
                _save_cached(file_hash, i, provider_name, model, raw_sections, translated_sections)

                # Now apply to soup (this might fail, but cache is already saved)
                formatted_sections = soup.find_all(["p", "li", "h1", "h2", "h3", "h4", "blockquote"])
                translator._apply_to_soup(soup, formatted_sections, translated_sections, raw_sections)
                chapter.content = str(soup).encode("utf-8")

            all_translations[i] = (raw_sections, translated_sections)

        except Exception as exc:
            st.error(f"Error translating chapter {i + 1}: {exc}")
            # Save raw LLM responses even if parsing/application failed
            if translator._raw_responses:
                _save_raw_responses(file_hash, i, provider_name, model, translator._raw_responses)

        count += 1

    # --- Build EPUB ---
    default_out_name = f"{name_root}_de.epub"
    epub_built = False
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".epub") as out_tf:
            output_temp_path = out_tf.name
        epub.write_epub(output_temp_path, book, {"plugins": []})
        with open(output_temp_path, "rb") as f:
            out_bytes = f.read()
        os.remove(output_temp_path)

        st.session_state.output_epub_bytes = out_bytes
        st.session_state.output_filename = default_out_name
        epub_built = True
    except Exception as exc:
        st.error(f"Failed to build EPUB: {exc}")

    progress_bar.progress(100)

    if epub_built:
        st.success(f"Translation finished: {count} chapter(s) translated.")
        # Auto-trigger download so the file is saved before user switches away
        _auto_download(out_bytes, default_out_name, "application/epub+zip")
        st.download_button(
            label="Download translated EPUB",
            data=out_bytes,
            file_name=default_out_name,
            mime="application/epub+zip",
            key="dl_epub_main",
        )

    try:
        os.remove(temp_input_path)
    except Exception:
        pass

# --- Persistent EPUB download (visible on reruns within same session) ---
if not start_button and not preview_button:
    if st.session_state.output_epub_bytes is not None:
        st.success("Previous translation available for download.")
        st.download_button(
            label="Download translated EPUB",
            data=st.session_state.output_epub_bytes,
            file_name=st.session_state.output_filename or "translated.epub",
            mime="application/epub+zip",
            key="dl_epub_persist",
        )
