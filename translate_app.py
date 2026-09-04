import os
import re
import tempfile
import hashlib
import json
import threading
import time
import traceback

import streamlit as st
import streamlit.components.v1 as components
from ebooklib import epub
import ebooklib
from bs4 import BeautifulSoup

from chapter_translator import ChapterTranslator, PROMPT_VERSION, parse_batch_response, section_text
from llm_cache import CachedLLMProvider, LLMResponseCache, estimate_tokens, format_tokens
from llm_provider import create_provider
from model_catalog import FALLBACK_PRICES_DATE, fetch_live_prices, get_model_options

st.set_page_config(page_title="EPUB Chapter Translator", layout="centered")

# --- Version (update with each commit to verify deployment) ---
APP_VERSION = "1.11.0"

# --- Session state ---
if "auto_downloaded_jobs" not in st.session_state:
    st.session_state.auto_downloaded_jobs = set()

# --- Disk cache (survives full session loss / page reload on mobile) ---
CACHE_DIR = os.path.join(tempfile.gettempdir(), "epub_translator_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def _read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False)


def _chapter_key(file_hash, chapter_idx, prov, mdl):
    """Cache identity of one chapter. Includes PROMPT_VERSION, so editing a
    prompt invalidates old translations instead of serving them from cache."""
    return hashlib.sha256(
        f"{file_hash}:{chapter_idx}:{prov}:{mdl}:{PROMPT_VERSION}".encode()
    ).hexdigest()[:16]


def _chapter_path(file_hash, chapter_idx, prov, mdl, suffix=".json"):
    return os.path.join(CACHE_DIR, _chapter_key(file_hash, chapter_idx, prov, mdl) + suffix)


def _load_cached(file_hash, chapter_idx, prov, mdl):
    data = _read_json(_chapter_path(file_hash, chapter_idx, prov, mdl))
    return (data["raw"], data["translated"]) if data else None


def _save_cached(file_hash, chapter_idx, prov, mdl, raw, translated):
    _write_json(_chapter_path(file_hash, chapter_idx, prov, mdl),
                {"raw": raw, "translated": translated})


def _save_request_keys(file_hash, chapter_idx, prov, mdl, keys):
    """List of LLM response-cache keys used by a chapter."""
    _write_json(_chapter_path(file_hash, chapter_idx, prov, mdl, "_requests.json"), keys)


def _load_request_keys(file_hash, chapter_idx, prov, mdl):
    return _read_json(_chapter_path(file_hash, chapter_idx, prov, mdl, "_requests.json"))


def _save_error_report(file_hash, chapter_idx, prov, mdl, text):
    with open(_chapter_path(file_hash, chapter_idx, prov, mdl, "_error.txt"), "w") as f:
        f.write(text)


def _load_error_report(file_hash, chapter_idx, prov, mdl):
    try:
        with open(_chapter_path(file_hash, chapter_idx, prov, mdl, "_error.txt")) as f:
            return f.read()
    except OSError:
        return None


def _clear_error_report(file_hash, chapter_idx, prov, mdl):
    try:
        os.remove(_chapter_path(file_hash, chapter_idx, prov, mdl, "_error.txt"))
    except FileNotFoundError:
        pass


def _expected_sections(user_msg):
    """Section count a batch request asked for (stated explicitly in the prompt)."""
    match = re.search(r"exactly (\d+) sections", user_msg)
    return int(match.group(1)) if match else None


def _build_error_report(chapter_num, exc_traceback, request_keys, prov, mdl, cache):
    """Plain-text debug report: full traceback plus each batch's request/response."""
    lines = [
        f"EPUB Translator v{APP_VERSION} — error report",
        f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Provider / model: {prov} / {mdl}",
        f"Chapter: {chapter_num}",
        "",
        "=== Exception (full traceback) ===",
        exc_traceback.rstrip(),
        "",
    ]
    for bi, key in enumerate(request_keys or []):
        entry = cache.load(key)
        if entry is None:
            lines.append(f"=== Batch {bi + 1}: no cached response (LLM call never completed) ===")
            lines.append("")
            continue
        user_msg = next((m["content"] for m in entry["messages"] if m["role"] == "user"), "")
        expected = _expected_sections(user_msg) or "?"
        got = len(parse_batch_response(entry["response"]))
        lines += [
            f"=== Batch {bi + 1} — LLM RESPONSE (expected {expected} sections, {got} parseable) ===",
            entry["response"],
            "",
            f"=== Batch {bi + 1} — REQUEST (user message) ===",
            user_msg,
            "",
        ]
    return "\n".join(lines)


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


def _translatable_chapters(book):
    """Document items minus navigation documents. Deterministic — the UI and
    the background worker must agree on chapter indices."""
    all_items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
    nav_item_names = set()
    for item in all_items:
        try:
            content = item.get_content().decode("utf-8", errors="ignore")
            soup = BeautifulSoup(content, "html.parser")
            if soup.find("nav", attrs={"epub:type": "toc"}) or soup.find("nav", id="toc"):
                nav_item_names.add(item.get_name())
        except Exception:
            pass
    for item in all_items:
        name = item.get_name().lower()
        if "nav" in name or "toc" in name:
            nav_item_names.add(item.get_name())
    return [item for item in all_items if item.get_name() not in nav_item_names]


# --- Background translation jobs -------------------------------------------
# The translation runs in a server-side daemon thread and communicates only
# through files in CACHE_DIR, so it survives the browser tab being suspended
# (mobile), page reloads, and even full session loss. One job per book+model.

def _translated_name(name_root):
    """Name the finished book is downloaded under.

    It must differ from the uploaded file: a download that keeps the original
    name lands on top of the source book in the user's download folder.
    """
    return f"{name_root}_de.epub"


def _job_key(file_hash, prov, mdl):
    return hashlib.sha256(f"job:{file_hash}:{prov}:{mdl}:{PROMPT_VERSION}".encode()).hexdigest()[:16]


def _job_state_path(job_key):
    return os.path.join(CACHE_DIR, f"{job_key}_job.json")


def _job_input_path(job_key):
    return os.path.join(CACHE_DIR, f"{job_key}_input.epub")


def _job_output_path(job_key):
    return os.path.join(CACHE_DIR, f"{job_key}_output.epub")


def _read_job_state(job_key):
    return _read_json(_job_state_path(job_key))


def _write_job_state(job_key, state):
    # Written atomically so a concurrent poll never reads a half-written file
    tmp = _job_state_path(job_key) + ".tmp"
    _write_json(tmp, state)
    os.replace(tmp, _job_state_path(job_key))


def _run_translation_job(job_key, input_path, output_path, selected_indices,
                         prov_name, mdl, api_key, book_hash, state):
    """Background worker. Runs in a daemon thread: NO streamlit calls in here —
    all progress goes through the job state file. `state` is the initial job
    state the UI already wrote before starting the thread."""

    def flush():
        state["heartbeat"] = time.time()
        _write_job_state(job_key, state)

    try:
        cache = LLMResponseCache(CACHE_DIR, namespace=f"{prov_name}:{mdl}:{PROMPT_VERSION}")
        provider = CachedLLMProvider(create_provider(prov_name, api_key=api_key, model=mdl), cache=cache)
        translator = ChapterTranslator(provider=provider, max_tokens=4000)

        book = epub.read_epub(input_path)
        _ensure_toc_uids(book.toc)
        chapters = _translatable_chapters(book)

        for pos, i in enumerate(selected_indices, start=1):
            chapter = chapters[i]
            head = f"Chapter {pos}/{len(selected_indices)}"
            _nb, _done, _keys = [0], [0], []

            def _on_event(kind, key, tokens, keys=_keys, nb=_nb, done=_done, head=head, idx=i):
                if key not in keys:
                    keys.append(key)
                    _save_request_keys(book_hash, idx, prov_name, mdl, keys)
                batch = f"batch {done[0] + 1}/{max(nb[0], 1)}"
                if kind == "llm_call":
                    state["message"] = f"{head} · {batch} · LLM call {format_tokens(tokens)}"
                    flush()
                elif kind == "cache_hit":
                    state["message"] = f"{head} · {batch} · cached response ({format_tokens(tokens)}, free)"
                    flush()
            provider.on_event = _on_event

            def _on_batch(current, total, done=_done, nb=_nb):
                done[0] = current
                nb[0] = total
            translator.on_batch_progress = _on_batch

            def _on_heal(msg, head=head):
                state["message"] = f"{head} · self-healing: {msg}"
                flush()
            translator.on_heal = _on_heal

            try:
                cached = _load_cached(book_hash, i, prov_name, mdl)
                if cached:
                    state["message"] = f"{head} · from cache (free)"
                    flush()
                    raw_sections, translated_sections = cached
                    translator.apply_cached(chapter, raw_sections, translated_sections)
                else:
                    raw_sections, translated_sections = translator.translate(chapter)
                    _save_cached(book_hash, i, prov_name, mdl, raw_sections, translated_sections)

                state["translated"] += 1
                _clear_error_report(book_hash, i, prov_name, mdl)
            except Exception as exc:
                tb = traceback.format_exc()
                _save_error_report(book_hash, i, prov_name, mdl,
                                   _build_error_report(i + 1, tb, _keys, prov_name, mdl, cache))
                state["failed"].append({"chapter": i + 1, "error": str(exc)})
            flush()

        if state["translated"]:
            state["message"] = "Building EPUB ..."
            flush()
            tmp = output_path + ".tmp"
            epub.write_epub(tmp, book, {"plugins": []})
            os.replace(tmp, output_path)
            state["status"] = "done"
            state["message"] = "finished"
        else:
            state["status"] = "error"
            state["error"] = "No chapters were translated — see 'Fix failed chapters' below."
    except Exception:
        state["status"] = "error"
        state["error"] = traceback.format_exc()
    flush()


def _auto_download(label):
    """Click the page's own download button once, in the top-level document.

    The file name has to come from Streamlit's Content-Disposition header. A
    blob built in here would belong to this sandboxed component iframe, and
    browsers ignore `download` on a cross-origin blob — they then name the file
    themselves, which is how a translation could come down under the original
    book's name and overwrite it. The button may not be rendered yet when this
    script first runs, so poll briefly for it.
    """
    components.html(
        """<script>
(function () {
  var doc;
  try { doc = window.parent.document; } catch (e) { return; }
  var label = %s;
  var tries = 0;
  var timer = setInterval(function () {
    var groups = doc.querySelectorAll('[data-testid="stDownloadButton"]');
    for (var i = 0; i < groups.length; i++) {
      if (groups[i].innerText.indexOf(label) !== -1) {
        var el = groups[i].querySelector('a, button');
        if (el) { clearInterval(timer); el.click(); return; }
      }
    }
    if (++tries > 40) { clearInterval(timer); }
  }, 100);
})();
</script>""" % json.dumps(label),
        height=0,
    )


DOWNLOAD_LABEL = "Download translated EPUB"


st.title("EPUB Chapter Translator")
st.caption(f"v{APP_VERSION}")

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

if not api_key:
    st.error(f"No API key found for {provider_name}. Add it to .streamlit/secrets.toml")
    st.stop()

# Raw LLM response cache: identical requests are replayed from disk for free,
# and responses are persisted the instant they arrive (before parsing can fail).
response_cache = LLMResponseCache(CACHE_DIR, namespace=f"{provider_name}:{model}:{PROMPT_VERSION}")
provider = CachedLLMProvider(
    create_provider(provider_name, api_key=api_key, model=model),
    cache=response_cache,
)

translator = ChapterTranslator(provider=provider, max_tokens=4000)


def _get_chapter_label(idx: int, chapter_item) -> str:
    """Chapter's own heading when it has one; 'Chapter N' framing only as fallback."""
    try:
        content = chapter_item.get_content().decode("utf-8", errors="ignore")
        soup = BeautifulSoup(content, "html.parser")
        heading = soup.find(["h1", "h2", "h3", "h4"])
        if heading:
            text = heading.get_text(strip=True)
            if text:
                return text[:80]
        first_p = soup.find("p")
        if first_p:
            text = first_p.get_text(strip=True)
            if text:
                return f"Chapter {idx + 1}: {text[:80]}"
    except Exception:
        pass
    return f"Chapter {idx + 1} ({chapter_item.get_name()})"

# --- File upload ---
uploaded = st.file_uploader("Choose an EPUB file")

if uploaded is None:
    st.info("Upload an EPUB to begin.")
    st.stop()

# Compute file hash for cache keying
file_hash = hashlib.md5(uploaded.getvalue()).hexdigest()

# read uploaded bytes and write to a temp file (ebooklib reads from path)
orig_filename = uploaded.name
name_root, _ = os.path.splitext(orig_filename)

with tempfile.NamedTemporaryFile(delete=False, suffix=".epub") as tf:
    tf.write(uploaded.getvalue())
    temp_input_path = tf.name

# load book (read_epub rejects anything that is not a valid EPUB/ZIP)
try:
    book = epub.read_epub(temp_input_path)
except Exception as e:
    st.error(f"Could not read the file as an EPUB: {e}")
    st.stop()
_ensure_toc_uids(book.toc)

chapters = _translatable_chapters(book)
n_chapters = len(chapters)

st.write(f"Found **{n_chapters}** translatable chapter(s).")

# --- Chapter multiselect with readable labels (made unique for selection) ---
chapter_labels = []
_label_counts = {}
for _i, _ch in enumerate(chapters):
    _lbl = _get_chapter_label(_i, _ch)
    _label_counts[_lbl] = _label_counts.get(_lbl, 0) + 1
    if _label_counts[_lbl] > 1:
        _lbl = f"{_lbl} ({_label_counts[_lbl]})"
    chapter_labels.append(_lbl)
_label_to_idx = {lbl: i for i, lbl in enumerate(chapter_labels)}

selected_labels = st.multiselect(
    "Select chapters to translate",
    options=chapter_labels,
    default=[],
    help="Chapter headings from the book; 'Chapter N' only where a chapter has no heading.",
)
selected_indices = sorted(_label_to_idx[lbl] for lbl in selected_labels)

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

if (start_button or preview_button) and not selected_indices:
    st.warning("Select at least one chapter first.")
    start_button = preview_button = False

# ========================
# Preview translation mode
# ========================
if preview_button:
    PREVIEW_SECTIONS = 5
    first_idx = selected_indices[0]
    preview_chapter = chapters[first_idx]

    soup = BeautifulSoup(preview_chapter.get_content(), "html.parser")
    blocks = ChapterTranslator.extract_blocks(soup)
    raw_sections = [s for s in (section_text(tag) for tag in blocks) if s]
    preview_sections = raw_sections[:PREVIEW_SECTIONS]

    if not preview_sections:
        st.warning("No translatable text found in the selected chapter.")
        st.stop()

    st.subheader(f"Preview – {chapter_labels[first_idx]}")
    st.caption(
        f"Translating {len(preview_sections)} section(s) with {provider_name} / {model} "
        f"(~{estimate_tokens(''.join(preview_sections)) // 1000 + 1}k tokens)."
    )

    # Uses the selected provider through the response cache, so repeating a
    # preview of the same sections costs nothing.
    try:
        with st.spinner("Calling LLM ..."):
            translated = translator._translate_batch(preview_sections)
        for orig, trans in zip(preview_sections, translated):
            # Sections carry inline <em>/<strong> markup; show it as plain text
            # rather than as escaped tags in the middle of a sentence.
            st.markdown(f"**→** {BeautifulSoup(trans, 'html.parser').get_text()}")
            st.caption(BeautifulSoup(orig, "html.parser").get_text())
    except Exception as exc:
        st.error(f"Error: {exc}")
        with st.expander("Error details"):
            st.code(traceback.format_exc())

    st.stop()

# ========================
# Full translation mode
# ========================
# ========================
# Full translation mode: runs in a background thread on the server, so it is
# safe to switch tabs or leave the page (mobile). Progress and the finished
# EPUB live on disk; the page just polls and auto-downloads when done.
# ========================
job_key = _job_key(file_hash, provider_name, model)

if start_button:
    _existing = _read_job_state(job_key)
    if _existing and _existing.get("status") == "running" and time.time() - _existing.get("heartbeat", 0) < 300:
        st.info("A translation for this book and model is already running — progress below.")
    else:
        with open(_job_input_path(job_key), "wb") as f:
            f.write(uploaded.getvalue())
        _init_state = {
            "status": "running", "message": "starting ...", "heartbeat": time.time(),
            "selected": selected_indices, "total": len(selected_indices),
            "translated": 0, "failed": [], "out_name": _translated_name(name_root),
        }
        _write_job_state(job_key, _init_state)
        threading.Thread(
            target=_run_translation_job,
            args=(job_key, _job_input_path(job_key), _job_output_path(job_key),
                  selected_indices, provider_name, model, api_key, file_hash,
                  _init_state),
            daemon=True,
        ).start()

_job = _read_job_state(job_key)
if _job and _job.get("status") == "running":
    if time.time() - _job.get("heartbeat", 0) > 300:
        st.error(
            "The background translation stopped unexpectedly (no progress for "
            "5 minutes) — press **Start translation** to run it again. Finished "
            "chapters were cached and will not be paid for twice."
        )
    else:
        st.info(
            "Translating in the background — it is safe to switch tabs or close "
            "this page. When you come back after it finishes, the EPUB downloads "
            "automatically."
        )

        @st.fragment(run_every=2)
        def _job_monitor():
            j = _read_job_state(job_key)
            if not j or j.get("status") != "running":
                st.rerun(scope="app")
            st.status(
                f"{j.get('message', '...')} · chapter {min(j.get('translated', 0) + 1, j.get('total', 1))}/{j.get('total', '?')}",
                state="running",
            )

        _job_monitor()

elif _job and _job.get("status") == "done":
    _failed = _job.get("failed", [])
    if _failed:
        _failed_list = ", ".join(str(f["chapter"]) for f in _failed)
        st.warning(
            f"Translation finished with errors: {_job.get('translated', 0)} chapter(s) translated, "
            f"{len(_failed)} failed (chapter(s) {_failed_list} left untranslated — see "
            f"**Fix failed chapters** below)."
        )
    else:
        st.success(f"Translation finished: {_job.get('translated', 0)} chapter(s) translated.")
    try:
        with open(_job_output_path(job_key), "rb") as f:
            _out_bytes = f.read()
        _out_name = _job.get("out_name") or ""
        # A job state written by an older version could still carry the
        # uploaded name; downloading under it would overwrite the original.
        if not _out_name.endswith("_de.epub"):
            _out_name = _translated_name(name_root)
        st.download_button(
            label=DOWNLOAD_LABEL,
            data=_out_bytes,
            file_name=_out_name,
            mime="application/epub+zip",
            key="dl_epub_main",
        )
        st.caption(f"Downloads as `{_out_name}`")
        if job_key not in st.session_state.auto_downloaded_jobs:
            st.session_state.auto_downloaded_jobs.add(job_key)
            _auto_download(DOWNLOAD_LABEL)
    except FileNotFoundError:
        st.error("The finished EPUB is no longer on the server — press **Start translation** to rebuild it (cached chapters are free).")

elif _job and _job.get("status") == "error":
    st.error("Translation failed.")
    with st.expander("Error details"):
        st.code(_job.get("error", "unknown error"))

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
        "Inspect each batch below, fix the response (it must be a JSON object whose "
        "keys match the request), press **Save**, then run **Start translation** "
        "again — saved batches are replayed from cache without calling the LLM."
    )
    for _i, _keys in _debuggable:
        with st.expander(f"Chapter {_i + 1} — {len(_keys)} cached batch(es)"):
            _report_txt = _load_error_report(file_hash, _i, provider_name, model)
            if _report_txt:
                st.download_button(
                    "⬇️ Download error report (.txt)",
                    data=_report_txt.encode("utf-8"),
                    file_name=f"chapter_{_i + 1}_error.txt",
                    mime="text/plain",
                    key=f"err_dl_fix_{_i}",
                )
            for _bi, _key in enumerate(_keys):
                _entry = response_cache.load(_key)
                if _entry is None:
                    st.warning(f"Batch {_bi + 1}: no cached response (the LLM call never completed — it will be re-sent).")
                    continue
                _user_msg = next((m["content"] for m in _entry["messages"] if m["role"] == "user"), "")
                _expected = _expected_sections(_user_msg)
                _got = len(parse_batch_response(_entry["response"]))
                _ok = _expected == _got
                st.markdown(
                    f"**Batch {_bi + 1}** — expected **{_expected}** sections, "
                    f"response has **{_got}** parseable {'✅' if _ok else '❌ (fix the JSON below)'}"
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
