"""Integrity tests for the JSON translation protocol.

Run with:  python -m unittest test_translator -v
"""
import json
import re
import unittest

from bs4 import BeautifulSoup

from chapter_translator import ChapterTranslator, parse_batch_response
from llm_provider import LLMProvider


class RecordingProvider(LLMProvider):
    """Echo-translates and records every request it receives."""

    def __init__(self):
        self.requests = []

    def chat(self, messages):
        user = next(m["content"] for m in messages if m["role"] == "user")
        self.requests.append(user)
        if user.startswith("Translate the TRANSLATE text"):
            m = re.search(r"TRANSLATE: (.*)\n\nCONTEXT AFTER:", user, re.S)
            return f"DE:{m.group(1).strip()}"
        body = user[user.index("{"):user.rindex("}") + 1]
        payload = json.loads(body)
        return json.dumps({k: f"DE:{v}" for k, v in payload.items()}, ensure_ascii=False)

    def batch_payloads(self):
        """The parsed JSON payload of every batch request that was sent."""
        out = []
        for user in self.requests:
            if user.startswith("Translate the values"):
                out.append(json.loads(user[user.index("{"):user.rindex("}") + 1]))
        return out


def make_translator(provider):
    return ChapterTranslator(provider=provider, max_tokens=4000)


class TestJsonRequestIsClean(unittest.TestCase):
    def test_request_is_valid_json_with_contiguous_keys(self):
        prov = RecordingProvider()
        make_translator(prov).translate_only(["First.", "Second.", "Third."])
        (payload,) = prov.batch_payloads()  # exactly one batch request
        self.assertEqual(sorted(payload, key=int), [str(i) for i in range(1, 4)])
        self.assertEqual(list(payload.values()), ["First.", "Second.", "Third."])

    def test_declared_section_count_matches_payload(self):
        prov = RecordingProvider()
        make_translator(prov).translate_only(["A.", "B."])
        user = prov.requests[0]
        declared = int(re.search(r"exactly (\d+) sections", user).group(1))
        (payload,) = prov.batch_payloads()
        self.assertEqual(declared, len(payload))

    def test_no_empty_values_in_request(self):
        prov = RecordingProvider()
        out = make_translator(prov).translate_only(["A.", "", "   ", "B."])
        (payload,) = prov.batch_payloads()
        self.assertEqual(list(payload.values()), ["A.", "B."])
        self.assertEqual(out, ["DE:A.", "", "", "DE:B."])

    def test_unicode_and_quotes_survive_roundtrip(self):
        tricky = 'Er sagte: "Über Nacht war\'s kalt — 5° draußen."'
        prov = RecordingProvider()
        out = make_translator(prov).translate_only([tricky])
        (payload,) = prov.batch_payloads()
        self.assertEqual(payload["1"], tricky)
        self.assertEqual(out, [f"DE:{tricky}"])

    def test_legit_repeated_paragraphs_are_kept(self):
        # scene separators like '***' repeat legitimately — must not be deduped
        prov = RecordingProvider()
        out = make_translator(prov).translate_only(["***", "Text.", "***"])
        (payload,) = prov.batch_payloads()
        self.assertEqual(list(payload.values()), ["***", "Text.", "***"])
        self.assertEqual(out, ["DE:***", "DE:Text.", "DE:***"])


class TestExtractionFeedsCleanRequests(unittest.TestCase):
    NESTED_HTML = """<html><body>
        <p>Plain paragraph.</p>
        <blockquote><p>Quoted text.</p></blockquote>
        <ul><li><p>List item text.</p></li></ul>
        <li>Outer item<ul><li>Inner item</li></ul></li>
        <h1>Heading</h1>
    </body></html>"""

    def test_nested_blocks_are_not_extracted_twice(self):
        soup = BeautifulSoup(self.NESTED_HTML, "html.parser")
        texts = [t.get_text(strip=True) for t in ChapterTranslator.extract_blocks(soup)]
        # each piece of text exactly once — no parent/child duplicates
        self.assertEqual(texts.count("Quoted text."), 1, texts)
        self.assertEqual(texts.count("List item text."), 1, texts)
        self.assertEqual(sum("Inner item" in t for t in texts), 1, texts)
        self.assertIn("Plain paragraph.", texts)
        self.assertIn("Heading", texts)

    def test_request_payload_from_nested_html_has_no_duplicates(self):
        soup = BeautifulSoup(self.NESTED_HTML, "html.parser")
        raw = [t.get_text(strip=True) for t in ChapterTranslator.extract_blocks(soup)]
        prov = RecordingProvider()
        make_translator(prov).translate_only(raw)
        (payload,) = prov.batch_payloads()
        values = list(payload.values())
        self.assertEqual(len(values), len(set(values)), f"duplicate sections sent to LLM: {values}")

    def test_apply_after_extraction_renders_each_text_once(self):
        soup = BeautifulSoup(self.NESTED_HTML, "html.parser")
        tags = ChapterTranslator.extract_blocks(soup)
        raw = [t.get_text(strip=True) for t in tags]
        prov = RecordingProvider()
        tr = make_translator(prov)
        translated = tr.translate_only(raw)
        tr._apply_to_soup(soup, tags, translated, raw)
        html = str(soup)
        self.assertEqual(html.count("DE:Quoted text."), 1, html)
        self.assertEqual(html.count("[Quoted text.]"), 1, html)
        self.assertNotIn("[]", html)


class TestParseBatchResponse(unittest.TestCase):
    def test_plain_fenced_and_truncated(self):
        self.assertEqual(parse_batch_response('{"1": "a"}'), {"1": "a"})
        self.assertEqual(parse_batch_response('```json\n{"1": "a"}\n```'), {"1": "a"})
        self.assertEqual(
            parse_batch_response('{"1": "a", "2": "b", "3": "cut of'),
            {"1": "a", "2": "b"},
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
