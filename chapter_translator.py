import hashlib
import json
import re
from typing import List

from bs4 import BeautifulSoup, NavigableString
from ebooklib.epub import EpubHtml

from llm_provider import LLMProvider

SECTION_DELIMITER = "⟪§⟫"

SYSTEM_PROMPT = (
    "You translate books into simple German for adult learners at CEFR level "
    "B1. B1 is a hard ceiling: never write a sentence that a B1 learner "
    "cannot read.\n"
    "\n"
    "PRIORITY: keep the meaning, not the sentence structure. When a faithful "
    "complex sentence and a simple B1 sentence conflict, always choose the "
    "simple one. Simplify the wording — never leave out content.\n"
    "\n"
    "HOW TO WRITE B1 GERMAN:\n"
    "- Aim for sentences of 10-18 words, never more than 20. Split a long "
    "original sentence only when it would break this limit — often two "
    "sentences are enough.\n"
    "- Vary sentence length: a mix of short and medium sentences reads "
    "naturally. Do not chop every sentence short.\n"
    "- At most one subordinate clause (Nebensatz) per sentence. Never nest them.\n"
    "- Use frequent, everyday words. Replace rare, literary or Latinate words "
    "with common ones (e.g. 'zeigen' not 'demonstrieren').\n"
    "- Prefer active voice and verbal style. Simple werden-Passiv is fine when "
    "it is the natural choice ('Das Haus wurde verkauft', 'er wurde geboren'). "
    "Never use Passiv with modal verbs ('muss beachtet werden') or Nominalstil "
    "('er untersuchte das Haus', not 'die Untersuchung des Hauses erfolgte').\n"
    "- Never use extended participial attributes ('der von allen bewunderte "
    "Mann' — write a relative clause instead) or chained Genitives.\n"
    "- No Konjunktiv I ('er sei', 'er habe' — write 'sagte, dass' with a normal "
    "verb instead). Simple Konjunktiv II (würde, wäre, hätte, könnte) is fine "
    "for wishes and unreal conditions.\n"
    "- Use common connectors: und, aber, oder, weil, denn, dass, als, wenn, ob, "
    "damit, bevor, nachdem, dann, danach, obwohl, trotzdem, deshalb. Never "
    "literary ones: obgleich, indessen, wenngleich, dessen ungeachtet, "
    "nichtsdestotrotz.\n"
    "- Präteritum is the normal tense for telling a story — use it with common "
    "verbs ('prüfte', 'fragte', 'lief'). If a Präteritum form is rare or hard "
    "to recognize ('barg', 'entwich'), use Perfekt or a more common verb.\n"
    "- Keep names, places and recurring terms consistent.\n"
    "\n"
    "EXAMPLE\n"
    "Too complex (B2/C1): 'Nachdem er die von seinem Vater hinterlassenen "
    "Unterlagen geprüft hatte, gelangte er zu der Überzeugung, dass eine "
    "Rückkehr unmöglich sei.'\n"
    "Correct (B1): 'Er prüfte die Unterlagen, die sein Vater ihm hinterlassen "
    "hatte. Danach war er sicher, dass er nicht mehr nach Hause zurückkommen "
    "konnte.' (two sentences of natural length, one Nebensatz each, no "
    "participle attribute, no Konjunktiv I)\n"
    "\n"
    "Output ONLY the German translation, in the exact format the task asks for. "
    "No notes, no explanations, no added text."
)

# The literal "exactly {section_count} sections" phrasing is parsed back out of
# the request by the app (to show expected/actual counts) — keep it intact.
USER_PROMPT_TEMPLATE = (
    "Translate the values of this JSON object into German at CEFR level B1 "
    "(never above B1).\n"
    "It contains exactly {section_count} sections, with keys \"1\" to "
    "\"{section_count}\".\n"
    "RULES:\n"
    "- The sections are paragraphs of one book chapter, in reading order. Use "
    "the neighboring sections as context — who is speaking, what pronouns refer "
    "to, recurring names and terms — but keep each translation strictly under "
    "its own key.\n"
    "- Answer with ONE valid JSON object with exactly the same keys. Each value "
    "must be the German translation of the input value with the same key.\n"
    "- Never merge or skip sections: the value for key \"7\" is the translation "
    "of input \"7\" and nothing else. Keep every key, even for very short "
    "sections.\n"
    "- Inside a value you may split a long sentence into shorter ones.\n"
    "- Check every sentence: longer than 20 words, more than one Nebensatz, or "
    "a word a B1 learner would not know? Rewrite it simpler.\n"
    "\n"
    "{sections_json}\n"
    "\n"
    "[REMINDER: answer with ONE JSON object, keys \"1\" to \"{section_count}\", "
    "values = German at B1 (sentences under 20 words, common words, no "
    "Nominalstil, no participial attributes). No text outside the JSON.]"
)

# Fallback for self-healing: one section per call. A single-text request has no
# structure the model could break, so it can never miscount. The neighboring
# paragraphs are passed as context so pronouns/speakers still resolve correctly.
SINGLE_PROMPT_TEMPLATE = (
    "Translate the TRANSLATE text into German at CEFR level B1 (never above "
    "B1).\n"
    "CONTEXT BEFORE and CONTEXT AFTER are the surrounding paragraphs of the "
    "same book chapter. Use them to resolve pronouns, speakers and recurring "
    "terms — but translate ONLY the TRANSLATE text.\n"
    "Answer with ONLY the translation. No labels, no notes.\n"
    "\n"
    "CONTEXT BEFORE: {before}\n"
    "\n"
    "TRANSLATE: {text}\n"
    "\n"
    "CONTEXT AFTER: {after}"
)

# Editing any prompt above changes this automatically, which invalidates the
# on-disk caches — otherwise chapters translated with an older prompt would keep
# being served from cache and the new instructions would appear to do nothing.
PROMPT_VERSION = hashlib.sha256(
    (SYSTEM_PROMPT + USER_PROMPT_TEMPLATE + SINGLE_PROMPT_TEMPLATE).encode("utf-8")
).hexdigest()[:8]


def parse_batch_response(response: str) -> dict:
    """Best-effort parse of a batch answer into {key: german_text}.

    Handles code fences and text around the JSON. If the JSON itself is broken
    (e.g. truncated mid-string), salvages every complete "key": "value" pair —
    the healing loop retranslates whatever could not be recovered.
    """
    text = response.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(text[start:end + 1])
            if isinstance(data, dict):
                return {str(k): v for k, v in data.items() if isinstance(v, str)}
        except json.JSONDecodeError:
            pass
    salvaged = {}
    for m in re.finditer(r'"(\d+)"\s*:\s*"((?:[^"\\]|\\.)*)"', text):
        try:
            salvaged[m.group(1)] = json.loads(f'"{m.group(2)}"')
        except json.JSONDecodeError:
            continue
    return salvaged


class ChapterTranslator:
    def __init__(self, provider: LLMProvider, max_tokens: int = 100000, on_response=None,
                 on_batch_progress=None, on_heal=None):
        self.provider = provider
        self.max_tokens = max_tokens
        self._raw_responses: List[str] = []
        self.on_response = on_response
        self.on_batch_progress = on_batch_progress
        self.on_heal = on_heal  # called with a short message when self-healing kicks in

    def translate(self, chapter: EpubHtml):
        """Translate chapter in-place. Returns (raw_sections, translated_sections)."""
        self._raw_responses = []
        soup = BeautifulSoup(chapter.content, "html.parser")
        formatted_sections = []
        raw_sections = []

        blocks = soup.find_all(["p", "li", "h1", "h2", "h3", "h4", "blockquote"])
        for tag in blocks:
            formatted_sections.append(tag)
            raw_sections.append(tag.get_text(strip=True))

        translated_sections = self._translate_chapter(raw_sections)

        self._apply_to_soup(soup, formatted_sections, translated_sections, raw_sections)
        chapter.content = str(soup).encode("utf-8")
        return raw_sections, translated_sections

    def translate_only(self, raw_sections: List[str]) -> List[str]:
        """Translate raw sections without applying to soup. Returns translated_sections."""
        self._raw_responses = []
        return self._translate_chapter(raw_sections)

    def apply_cached(self, chapter: EpubHtml, raw_sections: List[str], translated_sections: List[str]):
        """Apply previously cached translations to a chapter without calling LLM."""
        soup = BeautifulSoup(chapter.content, "html.parser")
        formatted_sections = soup.find_all(["p", "li", "h1", "h2", "h3", "h4", "blockquote"])
        self._apply_to_soup(soup, formatted_sections, translated_sections, raw_sections)
        chapter.content = str(soup).encode("utf-8")

    def _apply_to_soup(self, soup, formatted_sections, translated_sections, raw_sections):
        for tag, new_text, original_text in zip(formatted_sections, translated_sections, raw_sections):
            # Blocks with no text (spacers, image-only paragraphs, pure anchors)
            # have nothing to translate: leave them untouched — clearing them
            # would destroy images, and '[<empty>]' would render as stray '[]'
            if not original_text.strip():
                continue
            # For links only replace link name, preserve href
            if tag.name == "li" and tag.find("a"):
                a_tag = tag.find("a")
                a_tag.string = new_text
            else:
                # Attributes (incl. the tag's own id) survive; descendant anchors
                # are preserved by _set_translated_text so TOC fragments keep working
                self._set_translated_text(tag, new_text)

                original_tag = soup.new_tag(tag.name)
                original_tag.append(NavigableString(f"[{original_text}]"))
                if tag.parent:
                    tag.insert_after(original_tag)
                else:
                    tag.append(original_tag)

    @staticmethod
    def _set_translated_text(tag, new_text):
        """Replace a tag's content with translated text, keeping descendant
        elements that carry an id (e.g. <a id="ch3"/> inside a heading) —
        they are navigation targets and wiping them breaks the book's TOC."""
        anchors = tag.find_all(attrs={"id": True})
        for anchor in anchors:
            anchor.extract()
            anchor.clear()  # its text was part of the original; translation replaces it
        tag.clear()
        for anchor in anchors:
            tag.append(anchor)
        tag.append(NavigableString(new_text))

    def _translate_chapter(self, raw_sections: List[str]) -> List[str]:
        """Translate entire chapter as flowing text, split back by delimiters."""
        if not raw_sections:
            return []

        # Empty sections must not enter the LLM protocol: consecutive delimiters
        # with nothing between them (or a trailing delimiter after an empty last
        # section) make the model merge or drop delimiters and break the count.
        # Translate only non-empty sections and stitch the empties back after.
        non_empty_indices = [i for i, s in enumerate(raw_sections) if s.strip()]
        to_translate = [raw_sections[i] for i in non_empty_indices]
        if not to_translate:
            return [""] * len(raw_sections)

        batches = self._make_batches(to_translate)
        translated = []
        for batch_idx, batch in enumerate(batches):
            translated.extend(self._translate_batch(batch))
            if self.on_batch_progress:
                self.on_batch_progress(batch_idx + 1, len(batches))

        if len(translated) != len(to_translate):
            raise Exception(
                f"Translated sections count mismatch. "
                f"Expected {len(to_translate)}, got {len(translated)}."
            )

        translated_sections = [""] * len(raw_sections)
        for idx, text in zip(non_empty_indices, translated):
            translated_sections[idx] = text
        return translated_sections

    def _translate_batch(self, sections: List[str]) -> List[str]:
        """Translate sections as a keyed JSON object, healing automatically.

        Ladder: whole batch -> one retry with only the missing sections ->
        one call per still-missing section (which cannot miscount). The batch
        therefore always completes; nothing depends on the model keeping a
        fragile global count.
        """
        remaining = list(enumerate(sections))  # (original_index, text)
        results = {}

        for attempt in (1, 2):
            if not remaining:
                break
            payload = {str(j + 1): text for j, (_, text) in enumerate(remaining)}
            response = self._call_llm(self._batch_prompt(payload))
            parsed = parse_batch_response(response)

            still_missing = []
            for j, (orig_idx, text) in enumerate(remaining):
                value = parsed.get(str(j + 1), "")
                if isinstance(value, str) and value.strip():
                    results[orig_idx] = value.strip()
                else:
                    still_missing.append((orig_idx, text))

            if still_missing and attempt == 1:
                if len(still_missing) == len(remaining):
                    # Nothing parsed: an identical retry would be answered from
                    # the response cache with the same bad reply — go one-by-one
                    self._heal(f"response unusable ({len(remaining)} sections) — translating one by one")
                    break
                self._heal(f"{len(still_missing)}/{len(remaining)} sections missing — retrying them as a smaller batch")
            elif still_missing:
                self._heal(f"{len(still_missing)} section(s) still missing — translating one by one")
            remaining = still_missing

        for orig_idx, text in remaining:
            before = sections[orig_idx - 1][-300:] if orig_idx > 0 else "(none)"
            after = sections[orig_idx + 1][:300] if orig_idx + 1 < len(sections) else "(none)"
            results[orig_idx] = self._call_llm(
                SINGLE_PROMPT_TEMPLATE.format(before=before, text=text, after=after)
            ).strip()

        return [results[i] for i in range(len(sections))]

    def _heal(self, message: str):
        if self.on_heal:
            self.on_heal(message)

    @staticmethod
    def _batch_prompt(payload: dict) -> str:
        return USER_PROMPT_TEMPLATE.format(
            section_count=len(payload),
            sections_json=json.dumps(payload, ensure_ascii=False, indent=1),
        )

    def _make_batches(self, raw_sections: List[str]) -> List[List[str]]:
        """Split sections into batches that fit within token limits."""
        batches = [[]]
        current_length = 0

        for section in raw_sections:
            section_tokens = len(section) // 3  # rough char-to-token estimate
            if current_length + section_tokens > self.max_tokens and batches[-1]:
                batches.append([])
                current_length = 0
            batches[-1].append(section)
            current_length += section_tokens

        return batches

    def _call_llm(self, user_msg: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        response = self.provider.chat(messages)
        self._raw_responses.append(response)
        if self.on_response:
            self.on_response(response)
        return response

