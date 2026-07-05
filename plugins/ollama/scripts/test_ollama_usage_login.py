"""Self-check for the Playwright login capture's pure cookie-extraction helper. The
browser-driving path is verified live (and is the measurement-proven flow); this test
locks the extract-and-store logic that a wrong-cookie bug would otherwise slip through.
Importing the module does NOT require playwright (it is imported lazily inside capture).
"""
import os
import shutil
import tempfile
import unittest

import ollama_usage as ou
import ollama_usage_login as oul


class _Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._prev = os.environ.get("OLLAMA_USAGE_DIR")
        os.environ["OLLAMA_USAGE_DIR"] = self._tmp

    def tearDown(self):
        if self._prev is None:
            os.environ.pop("OLLAMA_USAGE_DIR", None)
        else:
            os.environ["OLLAMA_USAGE_DIR"] = self._prev
        shutil.rmtree(self._tmp, ignore_errors=True)


class TestStoreFromCookies(_Base):
    def test_extracts_secure_session_and_stores(self):
        cookies = [{"name": "aid", "value": "1", "domain": ".ollama.com"},
                   {"name": "__Secure-session", "value": "XYZ", "domain": ".ollama.com"}]
        self.assertEqual(oul.store_from_cookies(cookies), "__Secure-session=XYZ")
        self.assertEqual(ou._load_cookie(), "__Secure-session=XYZ")

    def test_absent_secure_session_returns_none(self):
        self.assertIsNone(oul.store_from_cookies(
            [{"name": "aid", "value": "1", "domain": ".ollama.com"}]))

    def test_ignores_wrong_domain(self):
        self.assertIsNone(oul.store_from_cookies(
            [{"name": "__Secure-session", "value": "X", "domain": ".evil.com"}]))

    def test_ignores_lookalike_domain(self):
        # 'evilollama.com' contains 'ollama.com' as a substring -- must NOT match
        self.assertIsNone(oul.store_from_cookies(
            [{"name": "__Secure-session", "value": "X", "domain": "evilollama.com"}]))

    def test_accepts_dot_ollama_com(self):
        self.assertEqual(oul.store_from_cookies(
            [{"name": "__Secure-session", "value": "OK", "domain": ".ollama.com"}]),
            "__Secure-session=OK")


if __name__ == "__main__":
    unittest.main()
