"""Headless full-pipeline test for NeuroWriter self-evolution loop.

Usage:
    python test_full_pipeline.py
"""
import json
import logging
import sys
import time
from dotenv import load_dotenv

load_dotenv()

import config
from core.pipeline_orchestrator import (
    PipelineOrchestrator,
    MAX_EVOLUTION_ITERATIONS,
    FACTUAL_ACCURACY_THRESHOLD,
    OVERALL_SCORE_THRESHOLD,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("pipeline_test")

# ── config ──────────────────────────────────────────────────────────
TOPIC = (
    "Deep learning-based automatic sleep staging using polysomnography "
    "in patients with obstructive sleep apnea"
)

def fmt_time(sec):
    m, s = divmod(int(sec), 60)
    return f"{m}m {s}s"


def print_scores(evaluation):
    scores = evaluation.get("scores", {})
    print("\n  %-22s  %s" % ("Criterion", "Score"))
    print("  " + "-" * 35)
    for k, v in scores.items():
        bar = "#" * int(v) + "." * (10 - int(v))
        print("  %-22s  %s  [%s]" % (k, v, bar))
    print("  " + "-" * 35)
    print("  %-22s  %.1f" % ("OVERALL", evaluation.get("overall_score", 0)))
    fa = scores.get("factual_accuracy", "?")
    print(f"\n  factual_accuracy={fa}  (threshold={FACTUAL_ACCURACY_THRESHOLD})")
    print(f"  overall={evaluation.get('overall_score',0)}  (threshold={OVERALL_SCORE_THRESHOLD})")


def main():
    api_key = config.OPENAI_API_KEY
    model = config.OPENAI_MODEL or "gpt-4o"
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set in .env")
        sys.exit(1)

    print("=" * 60)
    print("  NeuroWriter Full Pipeline Test")
    print("=" * 60)
    print(f"  Model:  {model}")
    print(f"  Topic:  {TOPIC[:70]}...")
    print(f"  Thresholds:  FA={FACTUAL_ACCURACY_THRESHOLD}, Overall={OVERALL_SCORE_THRESHOLD}")
    print(f"  Max iterations: {MAX_EVOLUTION_ITERATIONS}")
    print("=" * 60)

    orch = PipelineOrchestrator(api_key=api_key, model=model)
    t_total = time.time()

    # ── Stage 1: Parse topic ────────────────────────────────────────
    print("\n[1/6] Parsing topic...")
    t0 = time.time()
    topic_analysis = orch.parse_topic(TOPIC)
    print(f"  Done ({fmt_time(time.time()-t0)})")
    print(f"  Disease: {topic_analysis.get('disease')}")
    print(f"  Intervention: {topic_analysis.get('key_intervention_or_focus')}")
    print(f"  Queries: {len(topic_analysis.get('search_queries', []))}")

    # ── Stage 2: Research ───────────────────────────────────────────
    print("\n[2/6] Running research (collect -> landscape -> select)...")
    t0 = time.time()
    paper_pool, landscape, reference_pool = orch.run_research(topic_analysis, TOPIC)
    print(f"  Done ({fmt_time(time.time()-t0)})")
    print(f"  Papers collected: {len(paper_pool)}")
    print(f"  References selected: {len(reference_pool)}")
    print(f"  Key findings: {len(landscape.get('key_findings', []))}")
    print(f"  Knowledge gaps: {len(landscape.get('knowledge_gaps', []))}")

    # ── Stage 3: Writing strategy ───────────────────────────────────
    print("\n[3/6] Generating writing strategy...")
    t0 = time.time()
    writing_strategy = orch.generate_writing_strategy(topic_analysis, reference_pool, landscape)
    print(f"  Done ({fmt_time(time.time()-t0)})")
    paragraphs = writing_strategy.get("paragraphs", [])
    print(f"  Paragraphs planned: {len(paragraphs)}")

    # ── Stage 4: Generate introduction ──────────────────────────────
    print("\n[4/6] Generating introduction...")
    t0 = time.time()
    introduction = orch.generate_introduction(
        topic_analysis, reference_pool, landscape,
        writing_strategy=writing_strategy,
    )
    introduction = PipelineOrchestrator.validate_citation_range(
        introduction, len(reference_pool)
    )
    introduction, reference_pool = PipelineOrchestrator.renumber_citations(
        introduction, reference_pool
    )
    print(f"  Done ({fmt_time(time.time()-t0)})")
    word_count = len(introduction.split())
    print(f"  Word count: {word_count}")

    # ── Stage 5: Evaluate ───────────────────────────────────────────
    iteration = 0
    while True:
        label = "Draft" if iteration == 0 else f"Rev{iteration}"
        print(f"\n[5/6] Evaluating ({label})...")
        t0 = time.time()

        evaluation = orch.evaluate_introduction(
            introduction, reference_pool, topic_analysis, landscape
        )

        # Fact-check & adjust factual_accuracy
        fact_result = orch.run_fact_check(introduction, reference_pool)
        fc_accuracy = fact_result.get("overall_accuracy", "HIGH")
        fa_score = evaluation["scores"].get("factual_accuracy", 10)
        if fc_accuracy == "LOW":
            evaluation["scores"]["factual_accuracy"] = max(0, fa_score - 2)
        elif fc_accuracy == "MEDIUM":
            evaluation["scores"]["factual_accuracy"] = max(0, fa_score - 1)

        # Recalculate overall
        scores_vals = list(evaluation["scores"].values())
        evaluation["overall_score"] = round(sum(scores_vals) / len(scores_vals), 1)

        print(f"  Done ({fmt_time(time.time()-t0)})")
        print(f"  Fact-check accuracy: {fc_accuracy}")
        print_scores(evaluation)

        needs_evo = orch.needs_self_evolution(evaluation)
        can_iterate = iteration < MAX_EVOLUTION_ITERATIONS
        print(f"\n  needs_self_evolution={needs_evo}, iteration={iteration}/{MAX_EVOLUTION_ITERATIONS}")

        if not needs_evo or not can_iterate:
            if not needs_evo:
                print("  >> Scores above thresholds. Pipeline COMPLETE.")
            else:
                print("  >> Max iterations reached. Pipeline COMPLETE.")
            break

        # ── Stage 6: Self-evolution ─────────────────────────────────
        iteration += 1
        print(f"\n[6/6] Self-evolution iteration {iteration}...")
        t0 = time.time()

        # 6a: Extract unsupported claims
        print("  6a: Extracting unsupported claims...")
        claims = orch.extract_unsupported_claims(evaluation, introduction)
        print(f"      Found {len(claims)} claims")
        if not claims:
            print("      No claims to fix. COMPLETE.")
            break

        for i, c in enumerate(claims[:3]):
            print(f"      [{i+1}] {c.get('claim', '')[:80]}...")

        # 6b: Generate supplementary queries
        print("  6b: Generating supplementary queries...")
        queries = orch.generate_supplementary_queries(claims, topic_analysis)
        print(f"      Generated {len(queries)} queries")
        for q in queries[:3]:
            print(f"      - {q}")

        # 6c: Run supplementary search
        print("  6c: Searching PubMed...")
        new_papers = orch.run_supplementary_search(queries, paper_pool)
        paper_pool = paper_pool + new_papers
        print(f"      Found {len(new_papers)} new papers (pool: {len(paper_pool)})")

        # 6d: Expand reference pool
        if new_papers:
            print("  6d: Expanding reference pool...")
            reference_pool = orch.expand_reference_pool(
                reference_pool, new_papers, landscape,
                current_introduction=introduction,
            )
            print(f"      Reference pool: {len(reference_pool)}")

        # 6e: Regenerate introduction
        print("  6e: Regenerating introduction...")
        introduction = orch.generate_introduction(
            topic_analysis, reference_pool, landscape,
            writing_strategy=writing_strategy,
            evaluation_feedback=evaluation,
            unsupported_claims=claims,
        )
        introduction = PipelineOrchestrator.validate_citation_range(
            introduction, len(reference_pool)
        )
        introduction, reference_pool = PipelineOrchestrator.renumber_citations(
            introduction, reference_pool
        )
        word_count = len(introduction.split())
        print(f"  Done ({fmt_time(time.time()-t0)})")
        print(f"  New word count: {word_count}")

    # ── Summary ─────────────────────────────────────────────────────
    elapsed = time.time() - t_total
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Total time:    {fmt_time(elapsed)}")
    print(f"  Iterations:    {iteration} (Draft + {iteration} revisions)")
    print(f"  Final words:   {len(introduction.split())}")
    print(f"  Final refs:    {len(reference_pool)}")
    print(f"  Final overall: {evaluation.get('overall_score', '?')}")
    fa = evaluation.get('scores', {}).get('factual_accuracy', '?')
    print(f"  Final FA:      {fa}")
    triggered = iteration > 0
    print(f"  Self-evolution triggered: {'YES' if triggered else 'NO'}")
    print("=" * 60)

    # Print final intro (first 500 chars)
    print("\n--- Final Introduction (preview) ---")
    print(introduction[:500])
    if len(introduction) > 500:
        print("...")
    print("--- End preview ---\n")


if __name__ == "__main__":
    main()
