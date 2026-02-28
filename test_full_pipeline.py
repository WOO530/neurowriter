"""Headless full-pipeline test for NeuroWriter self-evolution loop.

Usage:
    python test_full_pipeline.py
"""
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

load_dotenv()

import config
from core.pipeline_orchestrator import (
    PipelineOrchestrator,
    MAX_EVOLUTION_ITERATIONS,
    FACTUAL_ACCURACY_THRESHOLD,
    OVERALL_SCORE_THRESHOLD,
    COMPLETENESS_THRESHOLD,
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
    model = "gpt-5.2"
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

        # Fact-check & adjust factual_accuracy (deterministic, claim_mapping based)
        fact_result = orch.run_fact_check(introduction, reference_pool)
        claim_mapping = fact_result.get("claim_mapping", {})
        fa_score = evaluation["scores"].get("factual_accuracy", 10)
        unsupported = [
            c for c in claim_mapping.get("claim_mappings", [])
            if not c.get("is_supported", True)
        ]
        major_mismatches = sum(
            1 for nm in claim_mapping.get("numerical_mismatches", [])
            if nm.get("severity") == "major"
        )
        num_mismatches = len(claim_mapping.get("numerical_mismatches", []))
        # Compute fact-checker implied score using ratio-based thresholds
        total_claims = len(claim_mapping.get("claim_mappings", []))
        unsupported_ratio = len(unsupported) / max(total_claims, 1)

        if major_mismatches >= 3 or unsupported_ratio >= 0.30:
            fc_implied = 5      # 30%+ unsupported = severe
        elif major_mismatches >= 1 or unsupported_ratio >= 0.15 or len(unsupported) >= 5:
            fc_implied = 7      # 15-30% unsupported = significant
        elif len(unsupported) >= 1 or num_mismatches >= 1:
            fc_implied = 8      # Minor issues
        elif num_mismatches == 0 and len(unsupported) == 0:
            fc_implied = 9      # No issues
        else:
            fc_implied = fa_score
        evaluation["scores"]["factual_accuracy"] = min(fa_score, fc_implied)

        # Recalculate overall
        scores_vals = list(evaluation["scores"].values())
        evaluation["overall_score"] = round(sum(scores_vals) / len(scores_vals), 1)

        print(f"  Done ({fmt_time(time.time()-t0)})")
        print(f"  Fact-check: {len(unsupported)} unsupported, {num_mismatches} mismatches -> FA={evaluation['scores'].get('factual_accuracy', '?')}")
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

        # 6a + 6a+: Extract unsupported claims and completeness gaps in parallel
        print("  6a: Extracting unsupported claims + completeness gaps (parallel)...")
        with ThreadPoolExecutor(max_workers=2) as executor:
            f_claims = executor.submit(orch.extract_unsupported_claims, evaluation, introduction)
            f_gaps = executor.submit(orch.extract_completeness_gaps, evaluation, introduction, landscape)
            claims = f_claims.result()
            comp_gaps = f_gaps.result()
        print(f"      Found {len(claims)} claims, {len(comp_gaps)} completeness gaps")

        for i, c in enumerate(claims[:3]):
            print(f"      [{i+1}] {c.get('claim', '')[:80]}...")

        if not claims and not comp_gaps:
            # Fallback: regenerate using evaluation feedback only
            print("      No claims/gaps — feedback-only regeneration.")
            introduction = orch.generate_introduction(
                topic_analysis, reference_pool, landscape,
                writing_strategy=writing_strategy,
                evaluation_feedback=evaluation,
                unsupported_claims=[],
                current_introduction=introduction,
            )
        else:
            # 6b: Generate supplementary queries
            all_items = list(claims) + orch._completeness_gaps_as_claims(comp_gaps)
            print("  6b: Generating supplementary queries...")
            queries = orch.generate_supplementary_queries(all_items, topic_analysis)
            print(f"      Generated {len(queries)} queries")
            for q in queries[:3]:
                q_str = q.get("query", q) if isinstance(q, dict) else q
                print(f"      - {q_str}")

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

                # Refresh landscape if new papers exceed 20% of pool
                if len(new_papers) > len(reference_pool) * 0.2:
                    print("  6d+: Refreshing landscape (new papers > 20% of pool)...")
                    landscape = orch.intro_generator.step_analyze_landscape(
                        paper_pool, TOPIC
                    )

            # 6e: Regenerate introduction
            print("  6e: Regenerating introduction...")
            introduction = orch.generate_introduction(
                topic_analysis, reference_pool, landscape,
                writing_strategy=writing_strategy,
                evaluation_feedback=evaluation,
                unsupported_claims=claims,
                current_introduction=introduction,
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
