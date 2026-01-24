import re
from unittest.mock import Mock

import pytest

from slam_eval.scorer import ExactMatch, IgnoreAllWhitespaces, json_string_to_dict
from slam_eval.ifbench.scorer import IFBenchScorer
from slam_eval.ifbench.checker_factory import IFBenchCheckerFactory


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

    def test_exact_match_with_preprocessing(self):
        scorer = ExactMatch("test_scorer", preprocessing_func=json_string_to_dict)
        result = scorer({"a": 1}, '{"a": 1}')
        assert result == 1

    def test_exact_match_case_sensitive(self):
        scorer = ExactMatch("test_scorer")
        result = scorer("Hello", "hello")
        assert result == 0


class TestIgnoreAllWhitespaces:
    def test_init(self):
        scorer = IgnoreAllWhitespaces("test_scorer")
        assert scorer.name == "test_scorer"

    def test_ignore_whitespaces_exact_match(self):
        scorer = IgnoreAllWhitespaces("test_scorer")
        result = scorer("hello", "hello")
        assert result == 1

    def test_ignore_whitespaces_with_spaces(self):
        scorer = IgnoreAllWhitespaces("test_scorer")
        result = scorer("hello world", "helloworld")
        assert result == 1

    def test_ignore_whitespaces_extra_spaces_in_pred(self):
        scorer = IgnoreAllWhitespaces("test_scorer")
        result = scorer("hello", "h e l l o")
        assert result == 1

    def test_ignore_whitespaces_extra_spaces_in_both(self):
        scorer = IgnoreAllWhitespaces("test_scorer")
        result = scorer("h e l l o", "hello")
        assert result == 1

    def test_ignore_whitespaces_different_content(self):
        scorer = IgnoreAllWhitespaces("test_scorer")
        result = scorer("hello", "world")
        assert result == 0

    def test_ignore_whitespaces_with_numbers(self):
        scorer = IgnoreAllWhitespaces("test_scorer")
        result = scorer("123 456", "123456")
        assert result == 1

    def test_ignore_whitespaces_with_special_chars(self):
        scorer = IgnoreAllWhitespaces("test_scorer")
        result = scorer("$19.99", "$ 1 9 . 9 9")
        assert result == 1

    def test_ignore_whitespaces_empty_strings(self):
        scorer = IgnoreAllWhitespaces("test_scorer")
        result = scorer("", "   ")
        assert result == 1

    def test_ignore_whitespaces_only_whitespace_true(self):
        scorer = IgnoreAllWhitespaces("test_scorer")
        result = scorer("   ", "")
        assert result == 1

    def test_ignore_whitespaces_converts_non_strings(self):
        scorer = IgnoreAllWhitespaces("test_scorer")
        result = scorer(123, "1 2 3")
        assert result == 1

    def test_ignore_whitespaces_tabs_and_newlines(self):
        scorer = IgnoreAllWhitespaces("test_scorer")
        result = scorer("hello\tworld\n", "hello world")
        assert result == 1

    def test_ignore_whitespaces_case_sensitive(self):
        scorer = IgnoreAllWhitespaces("test_scorer")
        result = scorer("Hello", "hello")
        assert result == 0


class TestIFBenchScorer:
    def test_partial_credit(self):
        factory = IFBenchCheckerFactory()

        checker_one = Mock()
        checker_one.build_description = Mock()
        checker_one.check_following = Mock(return_value=True)

        checker_two = Mock()
        checker_two.build_description = Mock()
        checker_two.check_following = Mock(return_value=False)

        factory.register("checker_one", lambda instruction_id=0: checker_one)
        factory.register("checker_two", lambda instruction_id=0: checker_two)

        scorer = IFBenchScorer(name="ifbench", checker_factory=factory)
        y_true = {
            "instruction_id_list": ["checker_one", "checker_two"],
            "kwargs": [{}, {}],
        }

        score = scorer(y_true, "response")

        assert score == 0.5
        checker_one.build_description.assert_called_once()
        checker_two.build_description.assert_called_once()
        checker_one.check_following.assert_called_once_with("response")
        checker_two.check_following.assert_called_once_with("response")
