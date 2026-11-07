"""Retrieval-quality evaluation for the hatchery quality-assistant RAG pipeline.

The assistant retrieves passages from the hatchery manual and asks an LLM to
answer a disposition question from them. Nothing measured whether the retrieval
step found the right section, so an answer could be fluent and grounded in the
wrong part of the manual without any test noticing.

Same design as AeroPredict's harness: a labelled question set, evidence tokens
that must appear in the retrieved context, and metrics that need no LLM. See
``aeropredict-platform/src/rag_eval.py`` for the reasoning behind separating
retrieval metrics (deterministic, CI-gated) from generation metrics (opt-in).

The evidence tokens are chosen to be section-specific. "8 percent" alone would
match several sections, so the infertility case requires "Flock Health" too.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------- cases


@dataclass(frozen=True)
class EvalCase:
    """One question with the evidence a correct retrieval must surface."""

    question: str
    expected_section: str
    evidence: tuple[str, ...]
    #: Tokens that must not appear in the top-ranked passage. Used where a
    #: neighbouring section is a plausible but wrong answer.
    forbidden: tuple[str, ...] = ()


OVOSCAN_CASES: tuple[EvalCase, ...] = (
    EvalCase(
        question="What visual indicators identify an infertile egg?",
        expected_section="2.2 INFERTILE",
        evidence=("blood ring", "blastoderm"),
    ),
    EvalCase(
        question="What action is required for a cracked shell?",
        expected_section="6.2 CRACKED SHELL - ACTION",
        evidence=("collection belt", "2 percent"),
    ),
    EvalCase(
        question="At what infertility rate is the Flock Health team escalated?",
        expected_section="SECTION 7 ESCALATION THRESHOLDS",
        evidence=("8 percent", "Flock Health"),
    ),
    EvalCase(
        question="What happens when a fertile egg also has a shell defect?",
        expected_section="SECTION 5 DISPOSITION RULES",
        evidence=("shell defect overrides",),
    ),
    EvalCase(
        question="How much shell contamination is grounds for rejection?",
        expected_section="3.5 DIRTY SHELL",
        evidence=("10 percent",),
    ),
    EvalCase(
        question="What nutrition deficiency causes soft-shelled eggs?",
        expected_section="6.4 SOFT-SHELLED - ACTION",
        evidence=("calcium", "vitamin D3"),
    ),
    EvalCase(
        question="What lighting level is required for transmitted light inspection?",
        expected_section="8.1 LIGHTING",
        evidence=("2000", "3000", "lux"),
    ),
    EvalCase(
        question="What is the maximum conveyor speed for the inspection station?",
        expected_section="8.2 CONVEYOR SPEED",
        evidence=("12 eggs per second",),
    ),
    EvalCase(
        question="What mean model confidence triggers a model review?",
        expected_section="SECTION 7 ESCALATION THRESHOLDS",
        evidence=("0.75",),
    ),
    EvalCase(
        question="How long are inspection records retained?",
        expected_section="SECTION 9 RECORD KEEPING",
        evidence=("24 months",),
    ),
)

# ------------------------------------------------------------------- metrics


def _normalise(text: str) -> str:
    """Collapse whitespace so a token split across a line break still matches.

    The manual is hard-wrapped at 79 columns, so "12 eggs per second" appears as
    "12 eggs per\\n    second". Matching raw text would report a retrieval
    failure that is really a formatting artefact.
    """
    return re.sub(r"\s+", " ", text).lower()


@dataclass
class CaseResult:
    case: EvalCase
    retrieved: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    hit: bool = False
    missing_evidence: list[str] = field(default_factory=list)
    forbidden_hits: list[str] = field(default_factory=list)
    rank: int | None = None


def evaluate_case(case: EvalCase, passages: list[dict]) -> CaseResult:
    """Score one case against the passages the retriever returned.

    ``forbidden`` is checked against the first passage only: with k=4 the
    retriever routinely returns a neighbouring section alongside the right one,
    which is harmless because the generator sees both. What matters is whether
    the top-ranked passage is the right one.
    """
    result = CaseResult(
        case=case,
        retrieved=[p["content"] for p in passages],
        scores=[p.get("score", 0.0) for p in passages],
    )
    joined = _normalise("\n".join(result.retrieved))

    result.missing_evidence = [t for t in case.evidence if _normalise(t) not in joined]
    result.hit = not result.missing_evidence

    if result.retrieved:
        top = _normalise(result.retrieved[0])
        result.forbidden_hits = [t for t in case.forbidden if _normalise(t) in top]

    for i, passage in enumerate(result.retrieved, start=1):
        text = _normalise(passage)
        if all(_normalise(t) in text for t in case.evidence):
            result.rank = i
            break
    return result


def summarise(results: list[CaseResult], k: int) -> dict:
    """Aggregate case results into the numbers a CI gate reads."""
    n = len(results)
    if n == 0:
        return {"cases": 0}

    hits = sum(1 for r in results if r.hit)
    ranks = [r.rank for r in results if r.rank is not None]
    wrong_top = [r for r in results if r.forbidden_hits]

    return {
        "cases": n,
        "k": k,
        "recall_at_k": round(hits / n, 4),
        "mrr": round(sum(1.0 / r for r in ranks) / n, 4) if ranks else 0.0,
        "hit_at_1": round(sum(1 for r in ranks if r == 1) / n, 4),
        "wrong_section_at_1": len(wrong_top),
        "failures": [
            {
                "question": r.case.question,
                "expected": r.case.expected_section,
                "missing_evidence": r.missing_evidence,
            }
            for r in results
            if not r.hit
        ],
        "ranking_weaknesses": [
            {
                "question": r.case.question,
                "expected": r.case.expected_section,
                "top_passage_from": r.forbidden_hits,
                "evidence_rank": r.rank,
            }
            for r in wrong_top
        ],
    }


# ------------------------------------------------------------------ runners


def run_retrieval_eval(agent, cases: tuple[EvalCase, ...], k: int = 4) -> dict:
    """Score retrieval only. No LLM, no network, deterministic."""
    results = [evaluate_case(case, agent.retrieve(case.question, k=k)) for case in cases]
    return summarise(results, k)


# ---------------------------------------------------------------------- cli


def _load_agent():
    # The app imports this as `src.agent.rag`, so the package root is the
    # repository directory, not src/.
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from src.agent.rag import HatcheryAgent  # noqa: PLC0415

    kb = repo_root / "knowledge_base" / "hatchery_manual.txt"
    return HatcheryAgent(str(kb)), OVOSCAN_CASES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.8,
        help="Fail when recall@k falls below this. Default 0.8.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON only.")
    args = parser.parse_args(argv)

    agent, cases = _load_agent()
    agent.ingest_knowledge()
    summary = run_retrieval_eval(agent, cases, k=args.k)

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"retrieval eval: {summary['cases']} cases, k={summary['k']}")
        print(f"  recall@{summary['k']}      {summary['recall_at_k']:.3f}")
        print(f"  mrr                {summary['mrr']:.3f}")
        print(f"  hit@1              {summary['hit_at_1']:.3f}")
        print(f"  wrong section @1   {summary['wrong_section_at_1']}")
        if summary["failures"]:
            print("\nretrieval failures (evidence not in top-k):")
            for f in summary["failures"]:
                print(f"  - {f['question']}")
                print(f"    expected: {f['expected']}")
                print(f"    missing:  {f['missing_evidence']}")
        if summary["ranking_weaknesses"]:
            print("\nranking weaknesses (right passage present, wrong one first):")
            for w in summary["ranking_weaknesses"]:
                print(f"  - {w['question']}")
                print(f"    expected: {w['expected']}, evidence at rank {w['evidence_rank']}")

    return 0 if summary["recall_at_k"] >= args.min_recall else 1


if __name__ == "__main__":
    raise SystemExit(main())
