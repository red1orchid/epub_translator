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

from chapter_translator import ChapterTranslator, SECTION_DELIMITER
from llm_cache import CachedLLMProvider, LLMResponseCache, estimate_tokens, format_tokens
from llm_provider import create_provider
from model_catalog import FALLBACK_PRICES_DATE, fetch_live_prices, get_model_options

st.set_page_config(page_title="EPUB Chapter Translator", layout="centered")

# --- Version (update with each commit to verify deployment) ---
APP_VERSION = "1.6.0"

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


def _request_keys_path(file_hash, chapter_idx, prov, mdl):
    """Path of the file listing LLM response-cache keys used by a chapter."""
    h = hashlib.sha256(f"{file_hash}:{chapter_idx}:{prov}:{mdl}".encode()).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"{h}_requests.json")


def _save_request_keys(file_hash, chapter_idx, prov, mdl, keys):
    with open(_request_keys_path(file_hash, chapter_idx, prov, mdl), "w") as f:
        json.dump(keys, f)


def _load_request_keys(file_hash, chapter_idx, prov, mdl):
    p = _request_keys_path(file_hash, chapter_idx, prov, mdl)
    if os.path.exists(p):
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            return None
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
    secret_model = st.secrets.get("openai_model", "")
elif provider_name == "anthropic":
    api_key = st.secrets.get("anthropic_key", "")
    secret_model = st.secrets.get("anthropic_model", "")


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def _cached_live_prices():
    try:
        return fetch_live_prices()
    except Exception:
        return None


_live_prices = _cached_live_prices()
_model_options = get_model_options(provider_name, _live_prices)
_model_ids = [o["id"] for o in _model_options]
_model_labels = [o["label"] for o in _model_options]
_CUSTOM = "Custom model…"

_default_idx = _model_ids.index(secret_model) if secret_model in _model_ids else 0
_selected_label = st.selectbox("Model", options=_model_labels + [_CUSTOM], index=_default_idx)
if _selected_label == _CUSTOM:
    model = st.text_input("Custom model ID", value=secret_model or _model_ids[0])
else:
    model = _model_ids[_model_labels.index(_selected_label)]

if _live_prices:
    st.caption("Prices per 1M tokens (input/output), fetched live from LiteLLM's price table.")
else:
    st.caption(f"Prices per 1M tokens (input/output), as of {FALLBACK_PRICES_DATE} (live price fetch unavailable).")

# --- Debug mode ---
debug_mode = st.checkbox("Debug mode (show LLM request without sending)", value=False)

if not api_key:
    st.error(f"No API key found for {provider_name}. Add it to .streamlit/secrets.toml")
    st.stop()

provider = create_provider(provider_name, api_key=api_key, model=model, debug=debug_mode)

# Raw LLM response cache: identical requests are replayed from disk for free,
# and responses are persisted the instant they arrive (before parsing can fail).
response_cache = LLMResponseCache(CACHE_DIR, namespace=f"{provider_name}:{model}")
if not debug_mode:
    provider = CachedLLMProvider(provider, cache=response_cache)

translator = ChapterTranslator(provider=provider, max_tokens=4000)

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
        "default_model": "claude-sonnet-4-6",
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


# ebooklib 0.20 parses the TOC from nav.xhtml with uid=None on every link;
# writing the NCX then crashes on those (navPoint id must be a string).
# Assign uids so books shipping both nav.xhtml and toc.ncx can be rebuilt.
def _ensure_toc_uids(entries, _counter=None):
    if _counter is None:
        _counter = [0]
    for entry in entries:
        if isinstance(entry, (tuple, list)):
            _ensure_toc_uids(entry[1], _counter)
        else:
            _counter[0] += 1
            if getattr(entry, "uid", None) in (None, ""):
                entry.uid = f"navpoint-{_counter[0]}"


_ensure_toc_uids(book.toc)

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
        _keys = _load_request_keys(file_hash, _i, provider_name, model)
        if _keys:
            _responses = [e["response"] for e in (response_cache.load(k) for k in _keys) if e]
            if _responses:
                _all_cache_text_parts.append(
                    f"=== Chapter {_i + 1} (raw LLM) ===\n" + "\n===BATCH===\n".join(_responses) + "\n"
                )

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
log_slot = st.empty()

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
    _log_lines = []

    def _status(msg):
        chapter_status.info(msg)
        _log_lines.append(msg)
        log_slot.markdown("\n".join(f"- {l}" for l in _log_lines[-15:]))

    for i in selected_indices:
        chapter = chapters[i]
        status.info(f"Translating chapter {i + 1} ({count + 1}/{total_to_translate}) ...")

        # Per-chapter batch tracking (for status messages)
        _chapter_batches = [0]
        _chapter_batch_done = [0]
        _chapter_keys = []

        # LLM cache events: record which cache entries this chapter used
        # (so failed chapters can be debugged/edited below) and report activity
        def _on_llm_event(kind, key, tokens, _idx=i, _keys=_chapter_keys,
                          _nb=_chapter_batches, _done=_chapter_batch_done):
            if key not in _keys:
                _keys.append(key)
                _save_request_keys(file_hash, _idx, provider_name, model, _keys)
            _batch = f"batch {_done[0] + 1}/{_nb[0]}"
            if kind == "cache_hit":
                _status(f"Chapter {_idx + 1} · {_batch} · reusing cached LLM response ({format_tokens(tokens)}, no API cost)")
            elif kind == "llm_call":
                _status(f"Chapter {_idx + 1} · {_batch} · LLM call {format_tokens(tokens)} ...")
            elif kind == "llm_done":
                _status(f"Chapter {_idx + 1} · {_batch} · response received ({format_tokens(tokens)}), cached to disk")
        if isinstance(provider, CachedLLMProvider):
            provider.on_event = _on_llm_event

        # Batch progress callback
        def _on_batch(current, total, _done=_chapter_batch_done):
            _done[0] = current
            _total_batches_completed[0] += 1
            if _total_batches[0] > 0:
                pct = int((_total_batches_completed[0] / _total_batches[0]) * 100)
                progress_bar.progress(min(pct, 100))
        translator.on_batch_progress = _on_batch

        try:
            cached = _load_cached(file_hash, i, provider_name, model)
            if cached:
                _status(f"Chapter {i + 1} · loading translation from cache (no API cost)")
                raw_sections, translated_sections = cached
                translator.apply_cached(chapter, raw_sections, translated_sections)
            else:
                _status(f"Chapter {i + 1} · extracting text")
                soup = BeautifulSoup(chapter.content, "html.parser")
                raw_sections = []
                for tag in soup.find_all(["p", "li", "h1", "h2", "h3", "h4", "blockquote"]):
                    raw_sections.append(tag.get_text(strip=True))

                # Calculate batches for this chapter to update total
                _batches = translator._make_batches(raw_sections)
                _chapter_batches[0] = len(_batches)
                _total_batches[0] += len(_batches)

                _status(
                    f"Chapter {i + 1} · {len(raw_sections)} sections, "
                    f"{len(_batches)} batch(es), {format_tokens(estimate_tokens(''.join(raw_sections)))} of text"
                )

                # Translate only (without applying to soup) and cache immediately
                translated_sections = translator.translate_only(raw_sections)
                _save_cached(file_hash, i, provider_name, model, raw_sections, translated_sections)

                # Now apply to soup (this might fail, but cache is already saved)
                _status(f"Chapter {i + 1} · applying translation to chapter HTML")
                formatted_sections = soup.find_all(["p", "li", "h1", "h2", "h3", "h4", "blockquote"])
                translator._apply_to_soup(soup, formatted_sections, translated_sections, raw_sections)
                chapter.content = str(soup).encode("utf-8")
                _status(f"Chapter {i + 1} · done")

            all_translations[i] = (raw_sections, translated_sections)

        except Exception as exc:
            st.error(f"Error translating chapter {i + 1}: {exc}")
            st.warning(
                f"The raw LLM responses for chapter {i + 1} are cached on disk. "
                f"Use **Fix failed chapters** below to inspect/edit them, then press "
                f"Start translation again — cached batches are replayed for free."
            )

        count += 1

    # --- Build EPUB ---
    status.info("Building EPUB ...")
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

    # --- Show debug info if in debug mode ---
    if debug_mode and hasattr(provider, 'last_request') and provider.last_request:
        st.subheader("🐛 Debug: Last LLM Request")
        st.json(provider.last_request)
        st.caption("Note: In debug mode, no actual LLM calls were made. Dummy responses were used.")

    if epub_built:
        status.empty()
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

# ========================
# Fix failed chapters: inspect/edit cached raw LLM responses and retry for free
# ========================
_debuggable = []
for _i in range(n_chapters):
    _keys = _load_request_keys(file_hash, _i, provider_name, model)
    if _keys and _load_cached(file_hash, _i, provider_name, model) is None:
        _debuggable.append((_i, _keys))

if _debuggable:
    st.divider()
    st.subheader("🔧 Fix failed chapters")
    st.caption(
        "These chapters got LLM responses, but assembling the translation failed. "
        "Inspect each batch below, fix the response (usually a missing/extra "
        f"'{SECTION_DELIMITER}' delimiter), press **Save**, then run **Start translation** "
        "again — saved batches are replayed from cache without calling the LLM."
    )
    for _i, _keys in _debuggable:
        with st.expander(f"Chapter {_i + 1} — {len(_keys)} cached batch(es)"):
            for _bi, _key in enumerate(_keys):
                _entry = response_cache.load(_key)
                if _entry is None:
                    st.warning(f"Batch {_bi + 1}: no cached response (the LLM call never completed — it will be re-sent).")
                    continue
                _user_msg = next((m["content"] for m in _entry["messages"] if m["role"] == "user"), "")
                _expected = _user_msg.count(SECTION_DELIMITER) + 1
                _got = _entry["response"].count(SECTION_DELIMITER) + 1
                _ok = _expected == _got
                st.markdown(
                    f"**Batch {_bi + 1}** — expected **{_expected}** sections, "
                    f"response has **{_got}** {'✅' if _ok else '❌ (fix the delimiters below)'}"
                )
                _edited = st.text_area(
                    f"Response for batch {_bi + 1}",
                    value=_entry["response"],
                    height=220,
                    key=f"edit_{_key}",
                    label_visibility="collapsed",
                )
                _c1, _c2 = st.columns(2)
                if _c1.button("💾 Save edited response", key=f"save_{_key}"):
                    response_cache.update_response(_key, _edited)
                    st.success("Saved. Press **Start translation** to retry — this batch costs nothing.")
                if _c2.button("🗑 Discard (re-translate via LLM)", key=f"del_{_key}"):
                    response_cache.delete(_key)
                    st.info("Discarded. This batch will be sent to the LLM again on the next run.")

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
