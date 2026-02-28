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
3. If score < 8: a specific suggestion for improvement

CALIBRATION ANCHORS — use these to calibrate your scoring:
- 10: Publication-ready for Nature Medicine / NEJM
- 8-9: Strong writing with only minor issues; meets the standard for a well-crafted draft
- 6-7: Adequate but with clear areas for improvement
- 4-5: Below average — multiple significant issues
- 0-3: Fundamentally flawed

SCORING PRINCIPLES:
- Judge the writing on its actual merits. A well-revised draft CAN score 8-9.
- Award high scores when the criterion is genuinely met, regardless of whether the text was AI-assisted.
- Focus on substance: does the text achieve what the criterion asks for?

Respond ONLY as JSON with this structure:
{
    "score": NUMBER (0-10),
    "feedback": "Explanation of why this score",
    "improvement": "Specific actionable suggestion if needed, or null if score >= 8"
}"""

    # Criteria that benefit from abstract context for accurate evaluation
    ABSTRACT_CRITERIA = {"factual_accuracy", "depth", "reference_quality", "originality"}
    use_abstracts = criterion in ABSTRACT_CRITERIA
    refs_summary = _format_references_summary(
        reference_pool,
        include_abstracts=use_abstracts,
        abstract_max_len=600
    )

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


def _format_checklist_items(items: list, max_items: int = 10, max_chars: int = 300) -> str:
    """Format a list of items as a numbered checklist

    Args:
        items: List of text items
        max_items: Maximum number of items to include
        max_chars: Maximum characters per item

    Returns:
        Numbered list string like "#1: ...\n#2: ..."
    """
    if not items:
        return "  (none)"
    lines = []
    for i, item in enumerate(items[:max_items], 1):
        text = str(item)[:max_chars]
        if len(str(item)) > max_chars:
            text += "..."
        lines.append(f"  #{i}: {text}")
    return "\n".join(lines)


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
- Score 9-10: Deeply specific to the exact research question. Mentions {topic_analysis.get('key_intervention_or_focus')} explicitly multiple times with specific findings. No more than 1-2 sentences of broad background
- Score 7-8: Mostly specific but has 1 paragraph of generic background that could apply to related topics
- Score 5-6: Mix of specific and generic content; several paragraphs could be reused for different studies
- Score 3-4: Mostly generic; only the final paragraph is specific to this study
- Score 0-2: Entirely generic; could apply to any study in this field""",

        "reference_density": """For REFERENCE DENSITY (citations portfolio and distribution):
- Score 9-10: Total 10-20 unique references cited in text, 50-70% of sentences have citations, multiple citations grouped where appropriate [N-M]
- Score 7-8: Total 8-15 unique references cited, 40-60% of sentences have citations
- Score 5-6: Total 5-8 references cited, <40% of sentences have citations
- Score 3-4: Total 3-5 references OR <25% of sentences have citations
- Score 0-2: Total <3 references OR citations nearly absent
NOTE: Not every sentence needs a citation. Common knowledge, transitions, and study aims are appropriately uncited. 30-50% uncited sentences is NORMAL for good academic writing.
Quality check: Count ACTUALLY CITED references in the text (not just available pool size). When citing multiple studies for same claim, verify most impactful/high-tier journals are selected""",

        "reference_quality": """For REFERENCE QUALITY (landmark papers, high-impact):
- Score 9-10: Multiple tier-1 journals cited; landmark/seminal papers included; systematic reviews and meta-analyses cited where relevant
- Score 7-8: Mix of high-impact and good journals; most key papers cited but may miss 1-2 landmark studies
- Score 5-6: Mostly lower-tier journals or missing several key papers; no meta-analyses cited
- Score 3-4: Predominantly obscure journals; most landmark papers missing
- Score 0-2: Poor journal distribution; no landmark papers""",

        "academic_tone": """For ACADEMIC TONE (Nature Medicine / NEJM / JAMA Psychiatry level):
- Score 9-10: Impeccable formal academic writing. Precise terminology. No colloquialisms. Hedging language used appropriately
- Score 7-8: Good academic tone with 1-2 minor issues (slightly informal phrasing, imprecise hedging)
- Score 5-6: Acceptable but multiple awkward phrasings or overly promotional language
- Score 3-4: Noticeably informal or reads like a textbook rather than a research paper
- Score 0-2: Casual tone, vague language, or inconsistent formality throughout""",

        "logical_flow": """For LOGICAL FLOW (coherence, smooth transitions):
- Score 9-10: Excellent narrative arc. Each paragraph builds on previous. Perfect transitions. Clear funnel structure (broad → specific → gap → aim)
- Score 7-8: Good flow with 1-2 minor transition issues or slightly abrupt topic changes
- Score 5-6: Adequate flow but multiple abrupt topic changes; paragraphs could be reordered without losing coherence
- Score 3-4: Disjointed sections; unclear why paragraphs are in current order
- Score 0-2: No logical structure; reads like a list of disconnected facts""",

        "depth": """For DEPTH (substantive detail, avoiding superficiality):
- Score 9-10: Key claims have supporting evidence/detail. Includes nuanced discussion of mechanisms, effect sizes, or methodological considerations where space permits
- Score 7-8: Good detail overall but 2-3 claims are stated without supporting evidence or specifics
- Score 5-6: Mix of detailed and superficial content; many claims are surface-level summaries
- Score 3-4: Mostly superficial; reads like an abstract rather than an introduction
- Score 0-2: No substantive detail; entirely unsupported generalizations
NOTE: Within a 500-600 word introduction (3-5 paragraphs), it is not feasible to provide granular detail for EVERY claim. Evaluate whether the MOST IMPORTANT claims have adequate depth, not whether every minor point is fully elaborated.""",

        "completeness": f"""For COMPLETENESS (covers key concepts from landscape):

WORD LIMIT CONTEXT: The introduction is limited to 500-600 words (3-5 paragraphs). It is physically impossible to address ALL items below in detail. Evaluate based on PRIORITIZED coverage.

KEY FINDINGS that should be mentioned ({len(landscape.get('key_findings', []))} total):
{_format_checklist_items(landscape.get('key_findings', []), max_items=10, max_chars=300)}

KNOWLEDGE GAPS that should be addressed ({len(landscape.get('knowledge_gaps', []))} total):
{_format_checklist_items(landscape.get('knowledge_gaps', []), max_items=10, max_chars=300)}

PRIORITIZATION GUIDE: Items directly related to the study's methodology, intervention, or clinical rationale should be covered first. Background epidemiology and tangential findings are lower priority.

- Score 9-10: Addresses the 5-7 most important key findings and 3-4 most relevant knowledge gaps; clear research rationale emerges. Related items synthesized together count as covered.
- Score 7-8: Covers most high-priority areas but misses 1-2 important findings or gaps directly relevant to the study rationale
- Score 5-6: Covers about half of key areas; several important themes absent
- Score 3-4: Misses multiple major areas; incomplete picture of the field
- Score 0-2: Covers only a narrow slice; most key areas absent

In your feedback, LIST which specific items above are MISSING from the introduction by their number (e.g., "Missing key finding #3, #7 and knowledge gap #2, #4"). Be specific about WHAT is missing, not just HOW MANY items are missing.""",

        "factual_accuracy": """For FACTUAL ACCURACY (claims match cited references):
- Score 9-10: All claims accurately reflect cited papers. Specific numbers match sources. No misrepresentations
- Score 7-8: Minor inaccuracies (1-2) such as slightly imprecise paraphrasing, but no clear misrepresentations. Synthesized claims (grouping multiple studies) are acceptable if the synthesis is fair
- Score 5-6: Several claims (3-5) that clearly misrepresent cited sources, OR 2+ fabricated specifics
- Score 3-4: Multiple clear misrepresentations or fabricated specifics not found in cited abstracts
- Score 0-2: Systematic misattributions or fabricated claims

GUIDELINES:
- Claims with specific numbers (sample sizes, percentages, p-values) MUST be verifiable from the cited abstract
- Synthesized claims citing multiple papers (e.g., "studies have demonstrated [1-3]") are ACCEPTABLE when the cited papers genuinely support the claim
- General qualitative claims that are consistent with (but not verbatim from) abstracts should NOT be penalized
- Focus on detecting actual ERRORS and MISREPRESENTATIONS, not on whether every claim is word-for-word verifiable""",

        "originality": """For ORIGINALITY (plagiarism risk — high score = low plagiarism risk):
- Score 9-10: Fully original phrasing with genuine synthesis across sources. Claims are reframed in the author's own analytical voice
- Score 7-8: Mostly original. Good synthesis overall. Standard medical terminology and established phrases (e.g., "obstructive sleep apnea", "randomized controlled trial", "major depressive disorder") do NOT count against originality
- Score 5-6: Multiple non-technical passages are near-verbatim from abstracts. Reads like restated individual abstracts rather than synthesis
- Score 3-4: Many passages are copied or minimally altered from abstracts. Patchwork of abstract sentences
- Score 0-2: Extensive verbatim copying from source abstracts

EVALUATION METHOD:
- EXCLUDE standard medical terms, disease names, methodology names, and established clinical phrases (≤8 words) from consecutive-word matching — these are domain conventions, not plagiarism
- Focus on whether the ARGUMENT and NARRATIVE are original, not individual phrases
- Check whether claims are synthesized across sources vs. merely restated from a single abstract""",

        "ai_detectability": """For AI DETECTABILITY (authenticity — high score = reads like human expert writing, low AI detection risk):
- Score 9-10: Reads like genuine expert academic writing. Varied sentence structure and paragraph rhythm. No repetitive AI patterns
- Score 7-8: Mostly natural writing. May have 1-2 minor patterns but overall reads authentically. Standard academic transitions ("Furthermore", "Moreover") used appropriately (1-2 times each) are ACCEPTABLE in scholarly writing
- Score 5-6: Several AI-typical patterns: SAME transition word used 3+ times, every paragraph identical structure, robotic hedging uniformity
- Score 3-4: Clearly reads like AI-generated text. Multiple telltale patterns throughout
- Score 0-2: Obviously AI-generated with pervasive patterns

AI-TYPICAL PATTERNS TO CHECK (only flag when EXCESSIVE, not when used naturally):
- SAME transition word repeated 3+ times (e.g., "Furthermore" appearing 3+ times). Using different transition words is fine
- Uniform paragraph structure: EVERY paragraph follows IDENTICAL pattern — slight variations are acceptable
- Hedging uniformity: the EXACT SAME hedging phrase used 4+ times
- Excessive balance: "While X... Y" pattern used 3+ times
- List-like structure: "First... Second... Third..." enumeration
- Uniform sentence length: ALL sentences approximately the same word count (low variance)

NOTE: Standard academic conventions (topic sentences, logical paragraph structure, appropriate hedging) are features of GOOD academic writing, not AI patterns. Only flag when patterns become repetitive and mechanical."""
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
        lines.append(f"Overview: {field_overview[:400]}")

    key_findings = landscape.get("key_findings", [])
    if key_findings:
        lines.append(f"\nKey Findings ({len(key_findings)}):")
        for i, finding in enumerate(key_findings[:8], 1):
            lines.append(f"  #{i}: {finding[:200]}")

    knowledge_gaps = landscape.get("knowledge_gaps", [])
    if knowledge_gaps:
        lines.append(f"\nKnowledge Gaps ({len(knowledge_gaps)}):")
        for i, gap in enumerate(knowledge_gaps[:8], 1):
            lines.append(f"  #{i}: {gap[:200]}")

    return "\n".join(lines) if lines else "Landscape analysis not available"


def _format_references_summary(
    reference_pool: list,
    include_abstracts: bool = False,
    abstract_max_len: int = 400
) -> str:
    """Format reference pool summary with paper titles for accurate evaluation

    Includes each paper's number, first author, title, journal, and year
    so the evaluator can judge whether cited claims match actual papers.

    When include_abstracts=True, appends a truncated abstract snippet to each
    paper entry. This gives the evaluator enough context (objective + key
    findings) to judge factual accuracy, depth, and reference quality without
    relying solely on titles.

    Args:
        reference_pool: List of reference papers
        include_abstracts: Whether to include abstract snippets (for criteria
            that need content-level evaluation: factual_accuracy, depth,
            reference_quality)
        abstract_max_len: Max characters per abstract snippet (default 400,
            covering ~2-3 sentences: objective + key findings)

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

        # Append abstract snippet for criteria that need content-level context
        if include_abstracts:
            abstract = paper.get("abstract", "")
            if abstract:
                snippet = abstract[:abstract_max_len]
                if len(abstract) > abstract_max_len:
                    snippet += "..."
                summary_lines.append(f"      Abstract: {snippet}")

    return "\n".join(summary_lines)
