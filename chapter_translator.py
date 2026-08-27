import hashlib
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
    "Output ONLY the German translation. No notes, no explanations, no added text."
)

# The literal "exactly {section_count} sections" phrasing is parsed back out of
# the request by the app (to show expected/actual counts) — keep it intact.
USER_PROMPT_TEMPLATE = (
    "Translate the following text into German at CEFR level B1 (never above B1).\n"
    "It contains exactly {section_count} sections separated by '{delimiter}'.\n"
    "RULES:\n"
    "- Preserve every '{delimiter}' exactly as-is. Output must have exactly "
    "{delimiter_count} delimiters ({section_count} sections). "
    "Do not merge or drop any section.\n"
    "- Inside a section you may split a long sentence into shorter ones. "
    "That does not change the number of sections.\n"
    "- Check every sentence before you answer: longer than 20 words, more than one "
    "Nebensatz, or a word a B1 learner would not know? Rewrite it simpler.\n"
    "\n"
    "{chapter_text}\n"
    "\n"
    "[REMINDER: {section_count} sections, {delimiter_count} delimiters "
    "('{delimiter}'). German at B1, never above: sentences under 20 words, common "
    "words, no Nominalstil, no participial attributes. No added text.]"
)

# Editing either prompt above changes this automatically, which invalidates the
# on-disk caches — otherwise chapters translated with an older prompt would keep
# being served from cache and the new instructions would appear to do nothing.
PROMPT_VERSION = hashlib.sha256(
    (SYSTEM_PROMPT + USER_PROMPT_TEMPLATE).encode("utf-8")
).hexdigest()[:8]


class ChapterTranslator:
    def __init__(self, provider: LLMProvider, max_tokens: int = 100000, on_response=None, on_batch_progress=None):
        self.provider = provider
        self.max_tokens = max_tokens
        self._raw_responses: List[str] = []
        self.on_response = on_response
        self.on_batch_progress = on_batch_progress

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

        batches = self._make_batches(raw_sections)
        translated_sections = []
        for batch_idx, batch in enumerate(batches):
            translated_sections.extend(self._translate_batch(batch))
            if self.on_batch_progress:
                self.on_batch_progress(batch_idx + 1, len(batches))

        if len(translated_sections) != len(raw_sections):
            raise Exception(
                f"Translated sections count mismatch. "
                f"Expected {len(raw_sections)}, got {len(translated_sections)}."
            )

        return translated_sections

    def _translate_batch(self, sections: List[str]) -> List[str]:
        """Send sections as flowing text separated by delimiters, parse back."""
        chapter_text = f"\n{SECTION_DELIMITER}\n".join(sections)
        response = self._call_llm(chapter_text, len(sections))

        # Split response by delimiter
        parts = re.split(r'\s*' + re.escape(SECTION_DELIMITER) + r'\s*', response.strip())

        if len(parts) != len(sections):
            raise Exception(
                f"Section count mismatch after translation. "
                f"Expected {len(sections)}, got {len(parts)}.\n"
                f"Raw response:\n{response}"
            )

        return [p.strip() for p in parts]

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

    def _call_llm(self, chapter_text: str, section_count: int) -> str:
        user_msg = USER_PROMPT_TEMPLATE.format(
            section_count=section_count,
            delimiter_count=section_count - 1,
            delimiter=SECTION_DELIMITER,
            chapter_text=chapter_text,
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        response = self.provider.chat(messages)
        self._raw_responses.append(response)
        if self.on_response:
            self.on_response(response)
        return response

