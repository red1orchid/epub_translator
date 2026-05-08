import json
import re
from typing import List

from bs4 import BeautifulSoup, NavigableString
from ebooklib.epub import EpubHtml

from llm_provider import LLMProvider


class ChapterTranslator:
    def __init__(self, provider: LLMProvider, max_tokens: int = 100000):
        self.provider = provider
        self.max_tokens = max_tokens
        self.glossary: dict = {}

    def translate(self, chapter: EpubHtml):
        soup = BeautifulSoup(chapter.content, "html.parser")
        formatted_sections = []
        raw_sections = []

        blocks = soup.find_all(["p", "li", "h1", "h2", "h3", "h4", "blockquote"])
        for tag in blocks:
            formatted_sections.append(tag)
            raw_sections.append(tag.get_text(strip=True))

        translated_sections = self._translate_sections(raw_sections)

        for tag, new_text, original_text in zip(formatted_sections, translated_sections, raw_sections):
            # For links only replace link name, preserve href
            if tag.name == "li" and tag.find("a"):
                a_tag = tag.find("a")
                a_tag.string = new_text
            elif tag.has_attr("id"):
                # Preserve tags with id (navigation anchors) but still show translation
                original_id = tag["id"]
                # Set translated text but keep the id
                tag.string = new_text
                tag["id"] = original_id

                # Add original after
                original_tag = soup.new_tag(tag.name)
                original_tag.append(NavigableString(f"[{original_text}]"))
                tag.insert_after(original_tag)
            else:
                # Update original tag with translated text
                tag.string = new_text

                # Create tag for original content
                original_tag = soup.new_tag(tag.name)
                original_tag.append(NavigableString(f"[{original_text}]"))

                # Insert the duplicate after the translated one
                tag.insert_after(original_tag)

        chapter.content = str(soup).encode("utf-8")

    def _translate_sections(self, raw_sections: List[str]) -> List[str]:
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

    def _translate_batch(self, batch: List[str]) -> List[str]:
        json_batch = json.dumps(batch, indent=2, ensure_ascii=False)
        response = self._call_llm(json_batch)
        try:
            # Find the JSON array in the response
            # Use a greedy match to get the outermost brackets
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if not match:
                raise ValueError("No JSON array found in response")
            result = json.loads(match.group(0))
            if len(result) != len(batch):
                raise ValueError(
                    f"Array length mismatch: expected {len(batch)}, got {len(result)}"
                )
            return result
        except Exception as e:
            raise Exception(f"Failed to parse response: {e}\nRaw response: {response}") from e

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

    def _call_llm(self, json_batch: str) -> str:
        glossary_text = ""
        if self.glossary:
            glossary_lines = [f"  {k} → {v}" for k, v in self.glossary.items()]
            glossary_text = (
                "\n\nUse this glossary for consistency with previous chapters:\n"
                + "\n".join(glossary_lines)
            )

        system_msg = (
            "You are a professional book translator. You translate text into German. "
            "Rules:\n"
            "- Keep the translation close to the original meaning.\n"
            "- Use standard modern German grammar and vocabulary (A2–B1 level).\n"
            "- Avoid poetic, archaic, or overly complex phrasing.\n"
            "- Keep names, places, and book-specific terms consistent.\n"
            "- Do not add explanations, notes, or extra text.\n"
            "- Output ONLY a valid JSON array with the same number of elements "
            "and in the same order as the input."
            f"{glossary_text}"
        )

        user_msg = (
            f"Translate this JSON array of text sections into German. "
            f"Return ONLY the JSON array with {json_batch.count(chr(10))} translated elements.\n\n"
            f"{json_batch}"
        )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        return self.provider.chat(messages)

    def update_glossary(self, raw_sections: List[str], translated_sections: List[str]):
        """Ask the LLM to extract key terms from the translated chapter for future consistency."""
        # Only do this if we have content
        if not raw_sections or not translated_sections:
            return

        # Take a sample of sections (first 20) to extract terms
        sample_pairs = list(zip(raw_sections[:20], translated_sections[:20]))
        pairs_text = "\n".join(
            f"  Original: {orig}\n  German: {trans}"
            for orig, trans in sample_pairs
        )

        messages = [
            {"role": "system", "content": (
                "You extract key proper nouns, character names, places, and recurring "
                "terms from translated text. Output ONLY a JSON object mapping "
                "original terms to their German translations. Include only important "
                "recurring terms (names, places, titles, special terms). "
                "Output 5-20 terms maximum. Output ONLY valid JSON, no other text."
            )},
            {"role": "user", "content": f"Extract key terms from these translation pairs:\n{pairs_text}"},
        ]

        try:
            response = self.provider.chat(messages)
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                new_terms = json.loads(match.group(0))
                self.glossary.update(new_terms)
        except Exception:
            pass  # glossary update is best-effort
