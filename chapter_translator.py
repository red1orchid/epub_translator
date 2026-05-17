import re
from typing import List

from bs4 import BeautifulSoup, NavigableString
from ebooklib.epub import EpubHtml

from llm_provider import LLMProvider

SECTION_DELIMITER = "⟪§⟫"


class ChapterTranslator:
    def __init__(self, provider: LLMProvider, max_tokens: int = 100000, on_response=None):
        self.provider = provider
        self.max_tokens = max_tokens
        self._raw_responses: List[str] = []
        self.on_response = on_response

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
            elif tag.has_attr("id"):
                # Preserve tags with id (navigation anchors) but still show translation
                original_id = tag["id"]
                tag.string = new_text
                tag["id"] = original_id

                original_tag = soup.new_tag(tag.name)
                original_tag.append(NavigableString(f"[{original_text}]"))
                if tag.parent:
                    tag.insert_after(original_tag)
                else:
                    tag.append(original_tag)
            else:
                tag.string = new_text

                original_tag = soup.new_tag(tag.name)
                original_tag.append(NavigableString(f"[{original_text}]"))
                if tag.parent:
                    tag.insert_after(original_tag)
                else:
                    tag.append(original_tag)

    def _translate_chapter(self, raw_sections: List[str]) -> List[str]:
        """Translate entire chapter as flowing text, split back by delimiters."""
        if not raw_sections:
            return []

        batches = self._make_batches(raw_sections)
        translated_sections = []
        for batch in batches:
            translated_sections.extend(self._translate_batch(batch))

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
        system_msg = (
            "You are a professional book translator. You translate text into German.\n"
            "Rules:\n"
            "- Keep the translation close to the original meaning.\n"
            "- Use standard modern German grammar and vocabulary (A2–B1 level).\n"
            "- Avoid poetic, archaic, or overly complex phrasing.\n"
            "- Keep names, places, and book-specific terms consistent.\n"
            "- Do not add explanations, notes, or extra text.\n"
            f"- The text has sections separated by the delimiter {SECTION_DELIMITER}\n"
            f"- Translate the text naturally as a whole, keeping full context across sections.\n"
            f"- Preserve EVERY {SECTION_DELIMITER} delimiter exactly as-is in your output.\n"
            f"- The output must have exactly {section_count} sections "
            f"(i.e. exactly {section_count - 1} delimiters).\n"
            "- Output ONLY the translated text with delimiters. No other text."
        )

        user_msg = (
            f"Translate the following chapter into German. "
            f"Keep the {SECTION_DELIMITER} delimiters in place:\n\n"
            f"{chapter_text}"
        )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        response = self.provider.chat(messages)
        self._raw_responses.append(response)
        if self.on_response:
            self.on_response(response)
        return response

