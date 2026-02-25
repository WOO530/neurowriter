"""Prompts for self-evaluation of generated introductions"""


def get_evaluation_prompt(
    criterion: str,
    introduction: str,
    reference_pool: list,
    topic_analysis: dict,
    landscape: dict,
    criterion_info: dict
) -> tuple[str, str]:
    """Get system and user prompts for evaluating a specific criterion

    Args:
        criterion: Criterion name (topic_specificity, reference_density, etc.)
        introduction: Generated introduction text
        reference_pool: List of available references
        topic_analysis: Topic analysis with concept hierarchy
        landscape: Literature landscape analysis
        criterion_info: Description and hints for this criterion

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    system_prompt = """You are an expert evaluator of medical research writing.
Evaluate the provided introduction against a SPECIFIC criterion, giving:
1. A numerical score (0-10, where 10 is perfect)
2. Detailed feedback explaining the score
3. If score < 7: a specific suggestion for improvement

Respond ONLY as JSON with this structure:
{
    "score": NUMBER (0-10),
    "feedback": "Explanation of why this score",
    "improvement": "Specific actionable suggestion if needed, or null if score >= 7"
}"""

    # Format references for context
    refs_summary = _format_references_summary(reference_pool)

    user_prompt = f"""EVALUATE THIS CRITERION:
Criterion: {criterion.replace('_', ' ').upper()}
Description: {criterion_info.get('description', '')}
Hint: {criterion_info.get('prompt_hint', '')}

TOPIC CONTEXT:
Disease: {topic_analysis.get('disease', '')}
Key Intervention: {topic_analysis.get('key_intervention_or_focus', '')}
Concepts:
{_format_concepts(topic_analysis.get('concept_hierarchy', []))}

RESEARCH LANDSCAPE:
{_format_landscape_summary(landscape)}

AVAILABLE REFERENCES ({len(reference_pool)} papers):
{refs_summary}

INTRODUCTION TO EVALUATE:
{introduction}

EVALUATION GUIDELINES:

{_get_criterion_guidelines(criterion, topic_analysis, landscape)}

Score and evaluate this introduction on the criterion above."""

    return system_prompt, user_prompt


def _get_criterion_guidelines(criterion: str, topic_analysis: dict, landscape: dict) -> str:
    """Get specific evaluation guidelines for each criterion

    Args:
        criterion: Criterion name
        topic_analysis: Topic analysis
        landscape: Landscape analysis

    Returns:
        Guidelines string
    """
    guidelines = {
        "topic_specificity": f"""For TOPIC SPECIFICITY (is it specific to {topic_analysis.get('disease')} + {topic_analysis.get('key_intervention_or_focus')} or generic?):
- Score 9-10: Deeply specific to the exact research question. Mentions {topic_analysis.get('key_intervention_or_focus')} explicitly multiple times with specific findings
- Score 7-8: Good specificity with some general content
- Score 5-6: Mix of specific and generic content
- Score 0-4: Mostly generic; could apply to many related topics""",

        "reference_density": """For REFERENCE DENSITY (citations portfolio and distribution):
- Score 9-10: Total 15-25 unique references, ≤15% sentences without citations, 30%+ multiple citations (showcasing field consensus). Single citation points max 5 studies [1-5]
- Score 7-8: Total 12-18 references, ≤20% sentences without citations, 20%+ multiple citations
- Score 5-6: Total 8-12 references, ≤30% sentences without citations
- Score 0-4: Total <8 references OR >40% sentences without citations
Quality check: When citing multiple studies for same claim, verify most impactful/high-tier journals are selected""",

        "reference_quality": """For REFERENCE QUALITY (landmark papers, high-impact):
- Score 9-10: Multiple tier-1 journals cited; landmark papers included
- Score 7-8: Mix of high-impact and good journals; key papers cited
- Score 5-6: Mostly lower-tier journals or missing key papers
- Score 0-4: Poor journal distribution; missing landmark papers""",

        "academic_tone": """For ACADEMIC TONE (Nature Medicine / NEJM / JAMA Psychiatry level):
- Score 9-10: Impeccable formal academic writing. Precise terminology. No colloquialisms
- Score 7-8: Good academic tone with minor issues
- Score 5-6: Acceptable but some awkward phrasing
- Score 0-4: Casual tone, vague language, or inconsistent formality""",

        "logical_flow": """For LOGICAL FLOW (coherence, smooth transitions):
- Score 9-10: Excellent narrative arc. Each paragraph builds on previous. Perfect transitions
- Score 7-8: Good flow with minor transition issues
- Score 5-6: Adequate flow but some abrupt topic changes
- Score 0-4: Disjointed; hard to follow; poor paragraph organization""",

        "depth": """For DEPTH (substantive detail, avoiding superficiality):
- Score 9-10: Each claim has supporting evidence/detail. Nuanced discussion
- Score 7-8: Good detail with minor superficial claims
- Score 5-6: Mix of detailed and superficial content
- Score 0-4: Many unsupported claims; lacks depth""",

        "completeness": f"""For COMPLETENESS (covers key concepts from landscape):
Knowledge gaps identified: {len(landscape.get('knowledge_gaps', []))}
Key findings that should be mentioned: {len(landscape.get('key_findings', []))}
- Score 9-10: Addresses all/nearly all key findings and gaps
- Score 7-8: Covers most key areas
- Score 5-6: Covers about half of key areas
- Score 0-4: Misses major areas""",

        "factual_accuracy": """For FACTUAL ACCURACY (claims match cited references):
- Score 9-10: All claims accurately reflect cited papers. No misrepresentations
- Score 7-8: Minor inaccuracies or unsupported claims (1-2)
- Score 5-6: Several questionable claims or misattributions (3-5)
- Score 0-4: Significant factual errors or many unsupported claims"""
    }

    return guidelines.get(criterion, "Evaluate this criterion fairly and objectively.")


def _format_concepts(concept_hierarchy: list) -> str:
    """Format concept hierarchy for display

    Args:
        concept_hierarchy: List of concepts

    Returns:
        Formatted string
    """
    if not concept_hierarchy:
        return "  (No concepts provided)"

    formatted = []
    for i, concept in enumerate(concept_hierarchy):
        indent = "  " * (i % 3)  # Simple indentation
        formatted.append(f"{indent} • {concept}")

    return "\n".join(formatted)


def _format_landscape_summary(landscape: dict) -> str:
    """Format landscape analysis for reference

    Args:
        landscape: Landscape analysis

    Returns:
        Formatted summary
    """
    lines = []

    field_overview = landscape.get("field_overview", "")
    if field_overview:
        lines.append(f"Overview: {field_overview[:200]}...")

    key_findings = landscape.get("key_findings", [])
    if key_findings:
        lines.append(f"\nKey Findings ({len(key_findings)}):")
        for finding in key_findings[:3]:
            lines.append(f"  - {finding[:80]}...")

    knowledge_gaps = landscape.get("knowledge_gaps", [])
    if knowledge_gaps:
        lines.append(f"\nKnowledge Gaps ({len(knowledge_gaps)}):")
        for gap in knowledge_gaps[:3]:
            lines.append(f"  - {gap[:80]}...")

    return "\n".join(lines) if lines else "Landscape analysis not available"


def _format_references_summary(reference_pool: list) -> str:
    """Format reference pool summary with paper titles for accurate evaluation

    Includes each paper's number, first author, title, journal, and year
    so the evaluator can judge whether cited claims match actual papers.

    Args:
        reference_pool: List of reference papers

    Returns:
        Formatted summary
    """
    if not reference_pool:
        return "No references available"

    summary_lines = []

    tier_count = {"tier_1": 0, "tier_2": 0, "tier_3": 0, "tier_4": 0}
    for paper in reference_pool:
        journal = (paper.get("journal") or "").lower()
        if any(t in journal for t in ["nature", "science", "nejm", "lancet", "jama"]):
            tier_count["tier_1"] += 1
        elif any(t in journal for t in ["psychiatry", "brain", "biological"]):
            tier_count["tier_2"] += 1
        elif any(t in journal for t in ["neuroimage", "clinical"]):
            tier_count["tier_3"] += 1
        else:
            tier_count["tier_4"] += 1

    summary_lines.append(f"Total: {len(reference_pool)} references")
    summary_lines.append(f"  Tier 1 (high-impact): {tier_count['tier_1']}")
    summary_lines.append(f"  Tier 2 (specialty): {tier_count['tier_2']}")
    summary_lines.append(f"  Tier 3-4 (other): {tier_count['tier_3'] + tier_count['tier_4']}")

    # Individual paper listing for accurate factual evaluation
    summary_lines.append("\nReference list:")
    for i, paper in enumerate(reference_pool, 1):
        authors = paper.get("authors", [])
        first_author = authors[0] if authors else "Unknown"
        if len(authors) > 1:
            first_author += " et al."
        title = paper.get("title", "No title")
        if len(title) > 80:
            title = title[:77] + "..."
        journal = paper.get("journal", "")
        year = paper.get("pub_year", "")
        summary_lines.append(f"  [{i}] {first_author}. {title}. {journal} {year}")

    return "\n".join(summary_lines)
