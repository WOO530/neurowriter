"""Prompt templates for claim extraction and supplementary query generation"""


def get_claim_extraction_prompt(
    evaluation: dict,
    introduction: str
) -> tuple[str, str]:
    """Get prompts for extracting unsupported claims from evaluation feedback

    Args:
        evaluation: Self-evaluation result with scores and feedback
        introduction: The introduction text

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    system_prompt = """You are an expert at identifying factual claims in medical writing that lack adequate supporting evidence.

Analyze the evaluation feedback and the introduction to identify specific claims that:
1. Are cited but the citation may not actually support the claim
2. Are stated as fact but have no citation
3. Are overgeneralized beyond what the cited evidence shows

IMPORTANT: Do NOT flag these as unsupported claims:
- Well-known disease definitions and universally accepted facts
- Logical transitions and narrative connectives
- Study aims and rationale statements
- Commonly cited prevalence/incidence figures that are general medical knowledge
Only flag specific empirical claims that require evidence from the cited literature.

Respond ONLY with valid JSON, no additional text or markdown formatting."""

    # Gather feedback from low-scoring criteria
    feedback_lines = []
    for criterion, fb in evaluation.get("feedback", {}).items():
        score = evaluation.get("scores", {}).get(criterion, 10)
        if score < 8:
            feedback_lines.append(f"- {criterion} (score {score}/10): {fb}")

    improvements_text = ""
    for imp in evaluation.get("improvements", []):
        improvements_text += f"- {imp.get('criterion', '')}: {imp.get('improvement', '')}\n"

    user_prompt = f"""Based on this evaluation of a medical research introduction, extract the specific unsupported or weakly supported claims.

EVALUATION FEEDBACK (low-scoring criteria):
{chr(10).join(feedback_lines) if feedback_lines else '(no specific low-scoring criteria)'}

IMPROVEMENT SUGGESTIONS:
{improvements_text if improvements_text else '(none)'}

INTRODUCTION TEXT:
{introduction}

Return JSON:
{{
    "unsupported_claims": [
        {{
            "claim": "The exact claim text from the introduction",
            "issue": "Why this claim is unsupported or weakly supported",
            "needed_evidence": "What type of evidence would support this claim"
        }},
        ...
    ]
}}

REQUIREMENTS:
- Extract 3-8 specific claims that need better evidence
- Focus on factual claims, not opinions or transitions
- Prioritize claims where finding supporting literature would most improve the introduction"""

    return system_prompt, user_prompt


def get_feedback_to_queries_prompt(
    user_feedback: str,
    writing_strategy: dict,
    topic_analysis: dict,
    landscape: dict
) -> tuple[str, str]:
    """Get prompts for converting user feedback into targeted PubMed queries

    Args:
        user_feedback: Natural language feedback from the user
        writing_strategy: Current writing strategy
        topic_analysis: Topic analysis for context
        landscape: Literature landscape analysis

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    system_prompt = """You are an expert at interpreting researcher feedback and translating it into targeted PubMed search queries.

The user has reviewed a writing strategy for a medical research introduction and provided feedback about what is missing or needs strengthening. Your job is to:
1. Interpret the feedback to understand what additional evidence is needed
2. Generate 3-8 specific PubMed queries that will find the relevant papers

Use proper PubMed syntax (MeSH terms, Boolean operators, field tags where helpful).

Respond ONLY with valid JSON, no additional text or markdown formatting."""

    # Format current strategy summary
    strategy_summary = ""
    paragraphs = writing_strategy.get("paragraphs", [])
    if paragraphs:
        para_topics = [f"  P{p.get('paragraph_number', '?')}: {p.get('topic', '')}" for p in paragraphs]
        strategy_summary = "\n".join(para_topics)

    narrative_arc = writing_strategy.get("narrative_arc", "")

    # Concept hierarchy context
    concept_hierarchy = topic_analysis.get("concept_hierarchy", [])
    concept_chain = " → ".join(concept_hierarchy) if concept_hierarchy else ""

    # Knowledge gaps
    knowledge_gaps = landscape.get("knowledge_gaps", [])
    gaps_text = "\n".join(f"  - {g}" for g in knowledge_gaps[:5]) if knowledge_gaps else "  (none)"

    user_prompt = f"""A researcher reviewed the writing strategy for their introduction and gave this feedback:

USER FEEDBACK:
"{user_feedback}"

CURRENT WRITING STRATEGY:
{strategy_summary}
Narrative arc: {narrative_arc}

RESEARCH CONTEXT:
- Disease: {topic_analysis.get('disease', '')}
- Data type: {topic_analysis.get('data_type', '')}
- Methodology: {topic_analysis.get('methodology', '')}
- Focus: {topic_analysis.get('key_intervention_or_focus', '')}
{f"- Concept hierarchy: {concept_chain}" if concept_chain else ""}

KNOWN KNOWLEDGE GAPS:
{gaps_text}

Return JSON:
{{
    "interpretation": "Brief interpretation of what the user wants",
    "queries": [
        {{
            "query": "The PubMed search query string",
            "rationale": "Why this query addresses the feedback"
        }},
        ...
    ]
}}

REQUIREMENTS:
- Generate 3-8 queries that specifically address the user's feedback
- Use specific medical terminology relevant to the feedback
- Include recent date filters where appropriate (e.g., 2018:2026[dp])
- Keep queries focused but not so narrow they return zero results
- At least one query should directly target the core issue in the feedback"""

    return system_prompt, user_prompt


def get_supplementary_query_prompt(
    claims: list,
    topic_analysis: dict
) -> tuple[str, str]:
    """Get prompts for generating targeted PubMed queries for unsupported claims

    Args:
        claims: List of unsupported claim dicts with 'claim', 'issue', 'needed_evidence'
        topic_analysis: Topic analysis for context

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    system_prompt = """You are an expert at constructing PubMed search queries.

Your task is to generate targeted, specific queries that will find papers supporting the given claims. Use proper PubMed syntax (MeSH terms, Boolean operators, field tags where helpful).

Respond ONLY with valid JSON, no additional text or markdown formatting."""

    claims_text = ""
    for i, claim in enumerate(claims, 1):
        c = claim if isinstance(claim, dict) else {"claim": str(claim)}
        claims_text += f"{i}. Claim: {c.get('claim', '')}\n"
        claims_text += f"   Needed: {c.get('needed_evidence', 'Supporting evidence')}\n\n"

    user_prompt = f"""Generate targeted PubMed search queries to find evidence for these unsupported claims:

RESEARCH CONTEXT:
- Disease: {topic_analysis.get('disease', '')}
- Data type: {topic_analysis.get('data_type', '')}
- Methodology: {topic_analysis.get('methodology', '')}

UNSUPPORTED CLAIMS:
{claims_text}

Return JSON:
{{
    "queries": [
        {{
            "query": "The PubMed search query string",
            "target_claim": 1,
            "rationale": "Why this query should find relevant evidence"
        }},
        ...
    ]
}}

REQUIREMENTS:
- Generate 3-8 queries total (not one per claim necessarily; some queries may cover multiple claims)
- Use specific medical terminology
- Include recent date filters where appropriate (e.g., 2018:2026[dp])
- Keep queries focused but not so narrow they return zero results"""

    return system_prompt, user_prompt


def get_completeness_gap_prompt(
    landscape: dict,
    introduction: str,
    evaluation: dict
) -> tuple[str, str]:
    """Get prompts for extracting completeness gaps by comparing introduction to landscape

    Compares the introduction against key_findings and knowledge_gaps from the
    landscape analysis to identify specific items that are missing or
    inadequately covered.

    Args:
        landscape: Literature landscape analysis with key_findings, knowledge_gaps
        introduction: The generated introduction text
        evaluation: Self-evaluation result (used for completeness feedback)

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    system_prompt = """You are an expert at evaluating the completeness of medical research introductions.

Your task is to compare an introduction against the known literature landscape (key findings and knowledge gaps) and identify specific items that are MISSING or inadequately covered.

Focus on substantive gaps — items that would significantly improve the introduction if addressed. Do not flag minor wording differences.

Respond ONLY with valid JSON, no additional text or markdown formatting."""

    # Format key findings as numbered list
    key_findings = landscape.get("key_findings", [])
    findings_text = ""
    if key_findings:
        findings_lines = []
        for i, f in enumerate(key_findings, 1):
            findings_lines.append(f"  #{i}: {str(f)[:300]}")
        findings_text = "\n".join(findings_lines)
    else:
        findings_text = "  (none)"

    # Format knowledge gaps as numbered list
    knowledge_gaps = landscape.get("knowledge_gaps", [])
    gaps_text = ""
    if knowledge_gaps:
        gaps_lines = []
        for i, g in enumerate(knowledge_gaps, 1):
            gaps_lines.append(f"  #{i}: {str(g)[:300]}")
        gaps_text = "\n".join(gaps_lines)
    else:
        gaps_text = "  (none)"

    # Completeness feedback from evaluation
    comp_feedback = evaluation.get("feedback", {}).get("completeness", "")
    comp_score = evaluation.get("scores", {}).get("completeness", "N/A")

    user_prompt = f"""Compare this introduction against the literature landscape and identify what is MISSING.

LITERATURE LANDSCAPE — KEY FINDINGS ({len(key_findings)} total):
{findings_text}

LITERATURE LANDSCAPE — KNOWLEDGE GAPS ({len(knowledge_gaps)} total):
{gaps_text}

COMPLETENESS EVALUATION (score: {comp_score}/10):
{comp_feedback[:500] if comp_feedback else '(no feedback available)'}

INTRODUCTION TEXT:
{introduction}

Return JSON:
{{
    "completeness_gaps": [
        {{
            "source": "key_finding" or "knowledge_gap",
            "item_number": N,
            "item_text": "The text of the missing item from the landscape",
            "gap_description": "How this item is missing or inadequately covered in the introduction",
            "needed_evidence": "What type of evidence or content would fill this gap"
        }},
        ...
    ]
}}

REQUIREMENTS:
- Extract 2-6 completeness gaps (most important first)
- Only flag items that are truly missing or very inadequately covered
- For each gap, explain what specific content would need to be added
- Prioritize gaps that would most improve the introduction's coverage of the field"""

    return system_prompt, user_prompt
