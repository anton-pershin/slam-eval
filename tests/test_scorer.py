import pytest
from slam_eval.scorer import ExactMatch


class TestExactMatch:
    def test_init(self):
        scorer = ExactMatch("test_scorer")
        assert scorer.name == "test_scorer"

    def test_exact_match_strings_equal(self):
        scorer = ExactMatch("test_scorer")
        result = scorer("hello", "hello")
        assert result == 1

    def test_exact_match_strings_not_equal(self):
        scorer = ExactMatch("test_scorer")
        result = scorer("hello", "world")
        assert result == 0

    def test_exact_match_numbers_equal(self):
        scorer = ExactMatch("test_scorer")
        result = scorer(42, 42)
        assert result == 1

    def test_exact_match_numbers_not_equal(self):
        scorer = ExactMatch("test_scorer")
        result = scorer(42, 43)
        assert result == 0

    def test_exact_match_mixed_types_not_equal(self):
        scorer = ExactMatch("test_scorer")
        result = scorer("42", 42)
        assert result == 0

    def test_exact_match_none_values(self):
        scorer = ExactMatch("test_scorer")
        result = scorer(None, None)
        assert result == 1

    def test_exact_match_none_vs_string(self):
        scorer = ExactMatch("test_scorer")
        result = scorer(None, "None")
        assert result == 0

    def test_exact_match_lists_equal(self):
        scorer = ExactMatch("test_scorer")
        result = scorer([1, 2, 3], [1, 2, 3])
        assert result == 1

    def test_exact_match_lists_not_equal(self):
        scorer = ExactMatch("test_scorer")
        result = scorer([1, 2, 3], [1, 2, 4])
        assert result == 0

    def test_exact_match_case_sensitive(self):
        scorer = ExactMatch("test_scorer")
        result = scorer("Hello", "hello")
        assert result == 0
