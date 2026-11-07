"""Tests for the hatchery retrieval-evaluation harness.

The harness is a measuring instrument, so it needs its own tests: a metric that
silently reports success is worse than no metric. These pin the scoring rules
that are easy to get subtly wrong — whitespace normalisation, the difference
between a retrieval failure and a ranking weakness, and the gate threshold.
"""

import pytest

from src.agent import rag_eval
from src.agent.rag_eval import EvalCase, evaluate_case, summarise


def _passages(*texts):
    return [{"content": t, "score": float(i)} for i, t in enumerate(texts)]


class TestNormalisation:
    def test_token_split_across_a_line_break_still_matches(self):
        """The manual is hard-wrapped, so evidence spans lines."""
        case = EvalCase(
            question="q",
            expected_section="s",
            evidence=("12 eggs per second",),
        )
        passages = _passages("rated for 12 eggs per\n    second at most")
        assert evaluate_case(case, passages).hit

    def test_matching_is_case_insensitive(self):
        case = EvalCase(question="q", expected_section="s", evidence=("flock health",))
        assert evaluate_case(case, _passages("escalate to the Flock Health team")).hit


class TestScoring:
    def test_missing_evidence_is_a_retrieval_failure(self):
        case = EvalCase(question="q", expected_section="s", evidence=("blood ring",))
        result = evaluate_case(case, _passages("unrelated text"))
        assert not result.hit
        assert result.missing_evidence == ["blood ring"]

    def test_evidence_in_a_later_passage_still_counts_as_a_hit(self):
        case = EvalCase(question="q", expected_section="s", evidence=("blood ring",))
        result = evaluate_case(case, _passages("noise", "noise", "a blood ring"))
        assert result.hit
        assert result.rank == 3

    def test_forbidden_token_in_a_later_passage_is_not_a_failure(self):
        case = EvalCase(
            question="q",
            expected_section="s",
            evidence=("blastoderm",),
            forbidden=("10 percent",),
        )
        result = evaluate_case(case, _passages("blastoderm disc", "10 percent"))
        assert result.hit
        assert result.forbidden_hits == []

    def test_forbidden_token_in_the_top_passage_is_flagged(self):
        case = EvalCase(
            question="q",
            expected_section="s",
            evidence=("blastoderm",),
            forbidden=("10 percent",),
        )
        result = evaluate_case(case, _passages("10 percent and blastoderm"))
        assert result.forbidden_hits == ["10 percent"]


class TestSummary:
    def test_recall_counts_hits_and_mrr_rewards_early_ranks(self):
        cases = (
            EvalCase(question="q1", expected_section="s", evidence=("a",)),
            EvalCase(question="q2", expected_section="s", evidence=("b",)),
        )
        results = [
            evaluate_case(cases[0], _passages("a")),
            evaluate_case(cases[1], _passages("noise", "b")),
        ]
        summary = summarise(results, k=4)
        assert summary["recall_at_k"] == 1.0
        assert summary["hit_at_1"] == 0.5
        assert summary["mrr"] == pytest.approx((1.0 + 0.5) / 2, abs=1e-4)

    def test_a_total_miss_lowers_recall(self):
        cases = (
            EvalCase(question="q1", expected_section="s", evidence=("a",)),
            EvalCase(question="q2", expected_section="s", evidence=("b",)),
        )
        results = [
            evaluate_case(cases[0], _passages("a")),
            evaluate_case(cases[1], _passages("nothing")),
        ]
        summary = summarise(results, k=4)
        assert summary["recall_at_k"] == 0.5
        assert len(summary["failures"]) == 1

    def test_empty_case_set_does_not_divide_by_zero(self):
        assert summarise([], k=4) == {"cases": 0}


class TestCaseSetIntegrity:
    """The cases are data, and bad data would make the gate meaningless."""

    def test_every_case_has_evidence(self):
        for case in rag_eval.OVOSCAN_CASES:
            assert case.evidence, f"{case.question} has no evidence tokens"

    def test_evidence_tokens_actually_exist_in_the_knowledge_base(self):
        """A typo in an evidence token would look like a retrieval failure."""
        from pathlib import Path

        kb = (
            Path(__file__).resolve().parents[1]
            / "knowledge_base"
            / "hatchery_manual.txt"
        )
        text = rag_eval._normalise(kb.read_text(encoding="utf-8"))
        for case in rag_eval.OVOSCAN_CASES:
            for token in case.evidence:
                assert rag_eval._normalise(token) in text, (
                    f"evidence token {token!r} for {case.question!r} "
                    "is not in the knowledge base"
                )

    def test_questions_are_unique(self):
        questions = [c.question for c in rag_eval.OVOSCAN_CASES]
        assert len(questions) == len(set(questions))
