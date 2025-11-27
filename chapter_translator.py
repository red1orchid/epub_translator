import json
import re
from typing import List

from bs4 import BeautifulSoup, NavigableString
from ebooklib.epub import EpubHtml
from openai import OpenAI


class ChapterTranslator:
    def __init__(self, api_key, model, max_tokens=30000):
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.model = model

    def translate(self, chapter: EpubHtml):
        soup = BeautifulSoup(chapter.content, "html.parser")
        formatted_sections = []
        raw_sections = []

        blocks = soup.find_all(["p", "li", "h1", "h2", "h3", "h4", "blockquote"])
        for tag in blocks:
            formatted_sections.append(tag)
            raw_sections.append(tag.get_text(strip=True))

        translated_sections = self._translate_sections(chapter, raw_sections)

        for tag, new_text, original_text in zip(formatted_sections, translated_sections, raw_sections):
            # For links only replace link name
            if tag.name == "li" and tag.find("a"):
                a_tag = tag.find("a")
                a_tag.string = new_text
            elif not tag.has_attr("id"):
                # Update original tag with translated text
                tag.string = new_text

                # Clone the tag for original content
                original_tag = tag.__copy__()  # Use copy of the tag structure
                original_tag.clear()
                original_tag.append(NavigableString(f"[{original_text}]"))

                # Insert the duplicate after the translated one
                tag.insert_after(original_tag)

        chapter.content = str(soup).encode("utf-8")

    def _translate_sections(self, chapter, raw_sections):
        translated_sections = []
        if raw_sections:
            batches = self._make_batches(raw_sections)
            for batch in batches:
                translated_sections.extend(self._translate_batch(batch))

        if len(translated_sections) != len(raw_sections):
            raise Exception(f"Translated sections length is different from original. "
                            f"Translated: {translated_sections}. Original: {raw_sections}")

        return translated_sections

    def _translate_batch(self, batch) -> List[str]:
        json_batch = json.dumps(batch, indent=2, ensure_ascii=False)
        response = self._translate(json_batch)
        try:
            lists = re.findall(r'\[.*?\]', response, re.DOTALL)
            return json.loads(lists[0])
        except:
            print(f"Failed to parse a response: {response}")
            return batch

    def _make_batches(self, raw_sections) -> List[List[str]]:
        batches = []
        batches.append([])
        current_length = 0

        for section in raw_sections:
            if current_length // 4 < self.max_tokens:
                batches[-1].append(section)
            else:
                batches.append([section])

        return batches

    def _translate(self, json_batch):
        client = OpenAI(api_key=self.api_key)

        prompt = f"""You are a translator. Translate the following chapter (given as a JSON list of sections) into German. Follow these rules:
    - Keep the translation close to the original meaning.
    - Use standard modern German grammar and vocabulary (A2–B1 level).
    - Avoid poetic, archaic, or overly complex phrasing.
    - Do not add explanations, notes, or extra text.
    - Output only the JSON list, with the same number of elements and in the same order as the input.

        List:
        {json_batch}"""

        completion = client.chat.completions.create(
            model=self.model,
            store=True,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return completion.choices[0].message.content
