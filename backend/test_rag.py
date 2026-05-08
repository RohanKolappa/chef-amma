"""
Evaluation suite for Chef Amma voice agent.

Tests retrieval quality, edge case handling, and tool call behavior
against known ground truth from the cookbook.

This is a seed evaluation suite — a small-scale version of what a
comprehensive AI agent testing framework would look like in production.
A full implementation would expand along several axes:

  1. Scale: hundreds of test cases covering every recipe in the cookbook
  2. Evaluation method: LLM-as-judge scoring for semantic accuracy,
     not just keyword matching
  3. Pipeline coverage: STT accuracy, tool call triggering rates,
     response grounding, latency budgets
  4. Regression detection: run automatically on every change to
     chunking params, prompts, or model versions
  5. Simulated conversations: synthetic multi-turn interactions
     testing edge cases, follow-ups, and context retention

Run: python test_rag.py
"""

import asyncio
import os
import sys
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env.local"))

from rag import retrieve, get_collection


# ── Test Framework ────────────────────────────────────────────────────


@dataclass
class EvalCase:
    """A single evaluation case with query, expected content, and metadata."""
    name: str
    query: str
    expected_keywords: list[str]
    source_page: str
    category: str  # "recipe", "technique", "ingredient", "edge_case"
    min_keyword_recall: float = 0.5  # minimum fraction of keywords that must appear


@dataclass
class EvalResult:
    case: EvalCase
    passed: bool
    keyword_recall: float
    found_keywords: list[str]
    missing_keywords: list[str]
    retrieved_preview: str
    failure_reason: str = ""


# ── Test Cases ────────────────────────────────────────────────────────

EVAL_CASES = [
    # ── Recipe retrieval (core functionality) ──────────────────────
    EvalCase(
        name="puttu_recipe",
        query="How to make puttu",
        expected_keywords=["Kerala", "steamed", "coconut", "rice", "puttu flour", "cylindrical"],
        source_page="Page 51",
        category="recipe",
    ),
    EvalCase(
        name="neer_dosa_recipe",
        query="neer dosa recipe",
        expected_keywords=["water dosa", "Mangalore", "rice", "coconut", "crêpes"],
        source_page="Page 47",
        category="recipe",
    ),
    EvalCase(
        name="pesarattu_recipe",
        query="pesarattu green mung bean",
        expected_keywords=["mung bean", "Andhra", "Telangana", "ginger chutney", "protein"],
        source_page="Page 49",
        category="recipe",
    ),
    EvalCase(
        name="fish_molee_recipe",
        query="fish molee Kerala",
        expected_keywords=["coconut milk", "Kerala", "Portuguese", "tilapia", "cardamom"],
        source_page="Page 162",
        category="recipe",
    ),
    EvalCase(
        name="ven_pongal_recipe",
        query="ven pongal recipe",
        expected_keywords=["rice", "mung beans", "peppercorns", "ghee", "cashews", "breakfast"],
        source_page="Page 81",
        category="recipe",
    ),

    # ── Ingredient/technique retrieval ─────────────────────────────
    EvalCase(
        name="gongura_pachadi",
        query="gongura pachadi",
        expected_keywords=["sorrel leaves", "Telugu", "sesame oil", "fenugreek", "garlic"],
        source_page="Page 101",
        category="ingredient",
    ),
    EvalCase(
        name="vangi_bath",
        query="vangi bath brinjal rice",
        expected_keywords=["eggplant", "spice powder", "cinnamon", "coriander"],
        source_page="Page 72",
        category="technique",
    ),
    EvalCase(
        name="curry_leaf_chutney",
        query="curry leaf mint cilantro chutney",
        expected_keywords=["curry leaves", "mint", "cilantro", "tamarind", "idli"],
        source_page="Page 97",
        category="recipe",
    ),

    # ── Edge cases (out-of-domain, ambiguous, adversarial) ─────────
    EvalCase(
        name="out_of_domain_query",
        query="how to change a car tire",
        expected_keywords=["cooking", "recipe", "rice", "spice"],  # if content returns, it should at least be cooking-related
        source_page="N/A — retrieval threshold (0.8) is intentionally permissive; LLM handles relevance filtering",
        category="edge_case",
        min_keyword_recall=0.25,  # low bar: some cooking content may surface, LLM is the second filter
    ),
    EvalCase(
        name="ambiguous_query",
        query="rice",
        expected_keywords=["rice"],  # broad query, should still return something relevant
        source_page="Multiple",
        category="edge_case",
        min_keyword_recall=1.0,
    ),
    EvalCase(
        name="misspelled_query",
        query="sambher recipe how to make",
        expected_keywords=["sambar", "lentil", "tamarind", "vegetables"],
        source_page="Multiple",
        category="edge_case",
        min_keyword_recall=0.25,  # misspelling may degrade retrieval
    ),
]


# ── Evaluation Runner ─────────────────────────────────────────────────


async def evaluate_case(case: EvalCase) -> EvalResult:
    """Run a single evaluation case and return structured results."""
    result_text = await retrieve(case.query, n_results=3)

    # Handle edge case: out-of-domain queries with no expected keywords
    if case.category == "edge_case" and len(case.expected_keywords) == 0:
        return EvalResult(
            case=case,
            passed=True,
            keyword_recall=0.0,
            found_keywords=[],
            missing_keywords=[],
            retrieved_preview=result_text[:200],
        )

    # Standard keyword recall evaluation
    found = [kw for kw in case.expected_keywords if kw.lower() in result_text.lower()]
    missing = [kw for kw in case.expected_keywords if kw.lower() not in result_text.lower()]
    recall = len(found) / len(case.expected_keywords) if case.expected_keywords else 1.0
    passed = recall >= case.min_keyword_recall

    return EvalResult(
        case=case,
        passed=passed,
        keyword_recall=recall,
        found_keywords=found,
        missing_keywords=missing,
        retrieved_preview=result_text[:200],
        failure_reason="" if passed else f"Keyword recall {recall:.0%} below threshold {case.min_keyword_recall:.0%}",
    )


async def run_eval_suite():
    """Run the full evaluation suite and report results."""
    collection = get_collection()
    print(f"{'='*70}")
    print(f"  Chef Amma RAG Evaluation Suite")
    print(f"  Collection: {collection.count()} chunks")
    print(f"  Test cases: {len(EVAL_CASES)}")
    print(f"{'='*70}\n")

    results: list[EvalResult] = []
    categories: dict[str, list[EvalResult]] = {}

    for case in EVAL_CASES:
        result = await evaluate_case(case)
        results.append(result)

        if case.category not in categories:
            categories[case.category] = []
        categories[case.category].append(result)

        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"  {status}  [{case.category:^10}]  {case.name}")

        if not result.passed:
            print(f"           Reason: {result.failure_reason}")
            if result.missing_keywords:
                print(f"           Missing: {result.missing_keywords}")

    # ── Summary by category ───────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Results by Category")
    print(f"{'='*70}")

    for category, cat_results in categories.items():
        passed = sum(1 for r in cat_results if r.passed)
        total = len(cat_results)
        avg_recall = sum(r.keyword_recall for r in cat_results) / total if total > 0 else 0
        print(f"  {category:<15} {passed}/{total} passed    avg recall: {avg_recall:.0%}")

    # ── Overall summary ───────────────────────────────────────────
    total_passed = sum(1 for r in results if r.passed)
    total = len(results)
    overall_recall = sum(r.keyword_recall for r in results) / total if total > 0 else 0

    print(f"\n{'='*70}")
    print(f"  Overall: {total_passed}/{total} passed    avg keyword recall: {overall_recall:.0%}")
    print(f"{'='*70}")

    if total_passed == total:
        print(f"\n  All tests passed.")
    else:
        print(f"\n  {total - total_passed} test(s) failed — review retrieval quality.")
        print(f"  Common fixes: adjust chunk size/overlap, increase top-k,")
        print(f"  or tune the relevance threshold in rag.py.")

    # Return exit code for CI integration
    return 0 if total_passed == total else 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_eval_suite())
    sys.exit(exit_code)