"""
Test 2 — api_base trailing-slash logic.
Verifies that the URL normalization prevents the 'v1chat' bug.
"""

import pytest


class TestApiBaseTrailingSlash:
    """Reproduce the logic from main.py line 93 in isolation."""

    @staticmethod
    def normalize(url: str | None) -> str:
        return url.rstrip("/") + "/" if url else "http://localhost:11434/v1/"

    def test_url_without_slash(self):
        assert self.normalize("https://api.regolo.ai/v1") == "https://api.regolo.ai/v1/"

    def test_url_with_slash(self):
        assert self.normalize("https://api.regolo.ai/v1/") == "https://api.regolo.ai/v1/"

    def test_url_double_slash(self):
        """Extra slashes should be collapsed to exactly one."""
        assert self.normalize("https://api.regolo.ai/v1//") == "https://api.regolo.ai/v1/"

    def test_url_none_fallback(self):
        assert self.normalize(None) == "http://localhost:11434/v1/"

    def test_url_empty_fallback(self):
        assert self.normalize("") == "http://localhost:11434/v1/"

    def test_concatenation_produces_valid_path(self):
        """Simulates what DSPy does: api_base + 'chat/completions'."""
        base = self.normalize("https://api.regolo.ai/v1")
        full = base + "chat/completions"
        assert full == "https://api.regolo.ai/v1/chat/completions"
        assert "v1chat" not in full  # the original bug
