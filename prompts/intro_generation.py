"""Prompt templates for introduction generation"""

from prompts.modality_config import SECTION_NAMES, ACADEMIC_EXAMPLES


def get_introduction_generation_prompt(
    parsed_topic: dict,
    selected_articles: list,
    output_instructions: dict = None,
    landscape: dict = None,
    modality: str = "eeg",
    writing_strategy: dict = None,
    evaluation_feedback: dict = None,
    unsupported_claims: list = None,
) -> tuple[str, str]:
    """Get system and user prompts for introduction generation

    Args:
        parsed_topic: Parsed topic with disease, data_type, methodology, etc.
        selected_articles: List of selected article metadata dictionaries
        output_instructions: Optional output formatting instructions
        landscape: Optional literature landscape analysis (from deep research)
        modality: Detected modality ("eeg", "psg", or "mixed")
        writing_strategy: Optional writing strategy with paragraph outline
        evaluation_feedback: Optional evaluation results from previous iteration
        unsupported_claims: Optional list of unsupported claims from previous iteration

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    mod = SECTION_NAMES.get(modality, SECTION_NAMES["eeg"])
    domain_label = mod["domain_label"]
    journal_examples = mod["journal_examples"]

    system_prompt = f"""You are a lead author of a peer-reviewed medical research article in {domain_label}, published in top-tier journals (Nature Medicine, NEJM, {journal_examples}).

YOUR WRITING VOICE:
- Formal, precise, and academically rigorous
- Use field-standard phrases: "a growing body of evidence", "remains a significant clinical challenge", "has garnered considerable attention", "warrants further investigation"
- Employ hedging language appropriately: "may indicate", "has been suggested", "evidence indicates", "emerging evidence", "it remains unclear"
- Avoid vague expressions and colloquialisms
- Minimize passive constructions; use active voice where possible
- Vary sentence structure: alternate between complex and concise sentences for readability

CITATION RULES (CRITICAL):
1. Use ONLY the provided references. Do NOT fabricate or hallucinate any citations.
2. Reference Portfolio: Use 15-35 unique references total (30-50 provided). Aim for high-density, purposeful citation coverage.
3. Citation Coverage: Attach citations to MOST factual statements, statistics, findings, and assertions. Sentences without citations should comprise no more than 20% of the introduction (exceptions: logical transitions, study aims, background context).
4. Multiple Independent Claims: When a single sentence contains multiple distinct factual claims, each MUST be cited separately.
   - BAD: "Depression affects 280 million people and is the leading cause of disability [1]."
   - GOOD: "Depression affects approximately 280 million people [1] and is among the leading causes of disability worldwide [2]."
5. Multiple Supporting Studies: When multiple studies support the same claim, cite them together using ranges: [3-5] or [3,4,5]. TARGET: 30-40% of your citations should be multiple (not single), showcasing consensus in the field.
6. Citation Density: Aim for approximately one citation per 60-80 words (medical baseline is ~1 per 95 words, but introduction should be denser). For a 600-800 word introduction, expect 10-15 distinct citation points.
7. Quality Over Quantity Per Point: MAXIMUM 5 studies per single citation point [1-5]. Do not pad citations unnecessarily; use only most impactful, directly relevant papers.
8. Sequential Numbering: Number citations sequentially as they appear in the text. If revisions occur, renumber for consistency.
9. Claim-Source Fidelity: Each cited claim MUST match the strength and specificity of the original source.
   - If the source reports a "trend" or "association", do NOT present it as "established" or "proven"
   - If the source uses a specific sample, do NOT generalize to a broader population unless the source explicitly does so
   - When uncertain about exact figures from an abstract, use approximations with hedging ("approximately", "roughly")

STRUCTURAL PRINCIPLES:
- Each paragraph should have a clear topic sentence
- Logical flow between paragraphs (no abrupt transitions)
- Build from general (disease background) to specific (your research rationale)
- End with a clear statement of study aims using: "In this study, we [aimed to/hypothesized that/investigated...]"

DO NOT:
- Speculate beyond what is supported by provided articles
- Use citations for definitions of the disorder or universally accepted facts
- Include references not in the provided list
- Vary the meaning of claims from what the original articles state
- Overstate or strengthen findings beyond what the cited source reports.
  If a study says "may contribute", do NOT write "has been established".
  Match the hedging level of the original paper (e.g., "suggested", "may indicate", "preliminary evidence").
- Cite a paper for a specific numerical claim (prevalence, percentage, sample size) unless the abstract explicitly states that number

SYNTHESIS REQUIREMENTS (CRITICAL):
- Do NOT list individual study results one by one ("Study X found Y. Study Z found W.")
- Instead, SYNTHESIZE findings across multiple papers into high-level claims supported by grouped citations
  BAD: "Smith et al. found 85% accuracy using CNN [1]. Lee et al. achieved 90% with LSTM [2]."
  GOOD: "Recent deep learning approaches have demonstrated promising classification accuracies (85-90%) across multiple architectures [1-3]."
- Each paragraph should make 2-4 synthesized claims, NOT 5-8 individual study summaries
- Specific numbers from individual papers should be used sparingly — only when a finding is particularly noteworthy or landmark
- Group related findings thematically, not paper-by-paper"""

    # Format articles with enhanced context for the prompt (800 char abstract limit for generation)
    articles_text = _format_articles_for_prompt(selected_articles, max_abstract_len=800)

    disease = parsed_topic.get("disease", "") or ""
    data_type = parsed_topic.get("data_type", "") or ""
    methodology = parsed_topic.get("methodology", "") or ""
    outcome = parsed_topic.get("outcome", "") or ""
    additional_context = parsed_topic.get("additional_context", "") or ""
    concept_hierarchy = parsed_topic.get("concept_hierarchy", []) or []
    key_intervention_or_focus = parsed_topic.get("key_intervention_or_focus", "") or ""

    # Determine primary focus for focus drift prevention
    primary_focus = concept_hierarchy[-1] if concept_hierarchy else key_intervention_or_focus or disease

    # Few-shot examples for academic tone (modality-specific)
    academic_examples = ACADEMIC_EXAMPLES.get(modality, ACADEMIC_EXAMPLES["eeg"])

    # Format landscape context if available
    landscape_context = ""
    if landscape:
        landscape_context = _format_landscape_context(landscape, parsed_topic)

    # Format writing strategy if available
    strategy_context = ""
    if writing_strategy:
        strategy_context = _format_writing_strategy(writing_strategy)

    # Format evaluation feedback for self-evolution
    feedback_context = ""
    if evaluation_feedback or unsupported_claims:
        feedback_context = _format_evaluation_feedback(evaluation_feedback, unsupported_claims)

    # Format concept hierarchy as a narrowing chain
    concept_text = ""
    if concept_hierarchy:
        concept_chain = " → ".join(concept_hierarchy)
        concept_text = f"\nCONCEPT HIERARCHY (broad → specific):\n{concept_chain}"

    # Focus constraint to prevent focus drift
    focus_constraint = ""
    if primary_focus and primary_focus.lower() != disease.lower():
        focus_constraint = f"""
FOCUS CONSTRAINT (MANDATORY):
This introduction is about "{primary_focus}", NOT about {disease} in general.
- You MUST mention "{primary_focus}" explicitly within the first 2 paragraphs
- After paragraph 1, ALL content MUST be specific to: {primary_focus}
- Generic statements about {disease} not connecting to {primary_focus} are PROHIBITED after paragraph 1
- Statistics should prefer the subtype/focus rather than the general disease
"""

    user_prompt = f"""Write a research paper introduction for a medical study on the following topic:

RESEARCH TOPIC:
- Disease/Condition: {disease}
- Neurophysiological Data: {data_type}
- Analysis Methodology: {methodology}
- Clinical Outcome: {outcome}
- Additional Context: {additional_context}
{concept_text}
{focus_constraint}
REQUIRED SECTIONS (in order — you may combine adjacent sections into a single paragraph):
1. **Disease Background & Clinical Burden ({disease}{f' — specifically {primary_focus}' if primary_focus.lower() != disease.lower() else ''})**: Prevalence, incidence, epidemiology, disease burden, mortality/morbidity impact, current management landscape
2. **Current Limitations & Emerging Biomarkers**: Why current approaches are inadequate, gaps in treatment response prediction; why {data_type} is relevant, what abnormalities have been observed, how biomarkers could improve outcomes
3. **{methodology} Approaches**: Survey of {methodology} approaches applied to {mod["condition_label"]} — discuss specific architectures and their results as a landscape overview, highlight methodological trends and remaining challenges
4. **Study Rationale & Aims**: Synthesis of unmet needs — why combining {data_type} with {methodology} is important, what remains unclear; clear statement of your study's purpose, novelty, and expected contributions

SPECIFICATIONS:
- LENGTH: 3-5 paragraphs, approximately 600-800 words. This is a STRICT upper limit — do NOT exceed 800 words. Write concisely: every sentence must earn its place.
- FORMAT: Separate each paragraph with a blank line. Do NOT output as a single block of text.
- TONE: Follow the academic examples provided above
- CITATIONS: Target 10-15 citation points across the introduction. Use 12-25 unique references total. See citation rules above for flexible strategy (1-5 per point, 30-40% multiple citations)
- FLOW: Smooth transitions between sections; avoid abrupt topic changes
- VOCABULARY: Use precise medical/neuroscience terminology

{academic_examples}

{landscape_context}

{strategy_context}

{feedback_context}

PROVIDED REFERENCES (ONLY use these for citations):
{articles_text}

IMPORTANT REMINDERS:
- If you cite a reference, [N] must correspond exactly to a numbered reference above
- Multiple independent factual claims in one sentence = multiple separate citations
- Ensure each claim is properly supported
- End the introduction with "In this study, we aimed to..." or similar

Write the introduction now:"""

    return system_prompt, user_prompt


def _format_landscape_context(landscape: dict, parsed_topic: dict) -> str:
    """Format literature landscape context for prompt

    Args:
        landscape: Landscape analysis dictionary
        parsed_topic: Parsed topic dictionary

    Returns:
        Formatted landscape context string
    """
    lines = ["LITERATURE LANDSCAPE CONTEXT (use to inform your introduction):"]

    # Field overview
    field_overview = landscape.get("field_overview", "")
    if field_overview:
        lines.append(f"\nField Overview:")
        lines.append(f"  {field_overview[:500]}")

    # Key findings
    key_findings = landscape.get("key_findings", [])
    if key_findings:
        lines.append(f"\nKey Findings in the Literature ({len(key_findings)} identified):")
        for finding in key_findings[:5]:
            lines.append(f"  - {finding[:200]}")

    # Knowledge gaps
    knowledge_gaps = landscape.get("knowledge_gaps", [])
    if knowledge_gaps:
        lines.append(f"\nIdentified Knowledge Gaps ({len(knowledge_gaps)} identified):")
        for gap in knowledge_gaps[:5]:
            lines.append(f"  - {gap[:200]}")

    # Methodological trends
    trends = landscape.get("methodological_trends", [])
    if trends:
        lines.append(f"\nMethodological Trends:")
        for trend in trends[:3]:
            lines.append(f"  - {trend[:200]}")

    lines.append("\nUse this context to ensure your introduction is specific to the research landscape and addresses key gaps.\n")
    return "\n".join(lines)


def _format_articles_for_prompt(articles: list, max_abstract_len: int = 2000) -> str:
    """Format articles for inclusion in prompt

    Args:
        articles: List of article dictionaries
        max_abstract_len: Maximum abstract length in characters.
            Use 800 for generation (prevents over-extraction of details).
            Use 2000 for fact-checking (needs full abstract for verification).

    Returns:
        Formatted string for prompt
    """
    formatted = []

    for i, article in enumerate(articles, 1):
        authors = ", ".join(article.get("authors", [])[:3])
        if len(article.get("authors", [])) > 3:
            authors += " et al."

        title = article.get("title", "No title")
        journal = article.get("journal", "Unknown Journal")
        year = article.get("pub_year", "Unknown Year")
        pmid = article.get("pmid", "Unknown PMID")

        abstract = article.get("abstract", "No abstract available")
        # Truncate abstract if too long
        if len(abstract) > max_abstract_len:
            abstract = abstract[:max_abstract_len - 3] + "..."

        # Article type label
        atype = article.get("article_type", "")
        type_label = ""
        if atype in ("review", "meta-analysis"):
            type_label = f" [{atype.upper()}]"

        formatted.append(
            f"[{i}]{type_label} {authors}. {title}. {journal}. {year}. PMID: {pmid}\n"
            f"    Abstract: {abstract}"
        )

    return "\n\n".join(formatted)


def _format_writing_strategy(writing_strategy: dict) -> str:
    """Format writing strategy as outline for generation prompt

    Args:
        writing_strategy: Strategy dict with paragraphs, narrative_arc

    Returns:
        Formatted strategy context string
    """
    if not writing_strategy or writing_strategy.get("parse_error"):
        return ""

    lines = ["WRITING OUTLINE (follow this structure):"]

    narrative_arc = writing_strategy.get("narrative_arc", "")
    if narrative_arc:
        lines.append(f"\nNarrative Arc: {narrative_arc[:300]}")

    paragraphs = writing_strategy.get("paragraphs", [])
    if paragraphs:
        lines.append(f"\nParagraph Plan ({len(paragraphs)} paragraphs):")
        for i, para in enumerate(paragraphs, 1):
            topic = para.get("topic", "")
            key_points = para.get("key_points", [])
            transition = para.get("transition_to_next", "")
            supporting = para.get("supporting_papers", [])

            lines.append(f"\n  Paragraph {i}: {topic}")
            for kp in key_points[:4]:
                lines.append(f"    - {kp[:150]}")
            if supporting:
                refs_str = ", ".join(str(s) for s in supporting[:8])
                lines.append(f"    Suggested refs: [{refs_str}] (reference numbers are from original pool — re-match by topic if pool has changed)")
            if transition:
                lines.append(f"    Transition: {transition[:100]}")

    lines.append("")
    return "\n".join(lines)


def _format_evaluation_feedback(evaluation_feedback: dict, unsupported_claims: list) -> str:
    """Format evaluation feedback and unsupported claims for self-evolution

    Args:
        evaluation_feedback: Evaluation results dict with scores and feedback
        unsupported_claims: List of claim dicts with 'claim', 'issue', 'needed_evidence'

    Returns:
        Formatted feedback context string
    """
    lines = ["PREVIOUS ISSUES (MUST address in this revision):"]

    if evaluation_feedback:
        scores = evaluation_feedback.get("scores", {})
        feedback = evaluation_feedback.get("feedback", {})
        improvements = evaluation_feedback.get("improvements", [])

        # Extract criteria with low scores
        weak_criteria = []
        for criterion, score in scores.items():
            if isinstance(score, (int, float)) and score < 7:
                criterion_feedback = feedback.get(criterion, "")
                weak_criteria.append((criterion, score, criterion_feedback))

        if weak_criteria:
            lines.append("\nLow-scoring criteria:")
            for criterion, score, fb in weak_criteria:
                lines.append(f"  - {criterion.replace('_', ' ').upper()} (score: {score}/10): {fb[:200]}")

        if improvements:
            lines.append("\nSuggested improvements:")
            for imp in improvements[:5]:
                if isinstance(imp, str):
                    lines.append(f"  - {imp[:200]}")
                elif isinstance(imp, dict):
                    lines.append(f"  - {imp.get('suggestion', imp.get('improvement', str(imp)))[:200]}")

    if unsupported_claims:
        lines.append(f"\nUnsupported claims found ({len(unsupported_claims)}):")
        for claim_info in unsupported_claims[:8]:
            claim_text = claim_info.get("claim", "")
            issue = claim_info.get("issue", "")
            lines.append(f"  - Claim: \"{claim_text[:150]}\"")
            if issue:
                lines.append(f"    Issue: {issue[:150]}")

    lines.append("\nYou MUST address ALL of these issues. Do NOT repeat the same errors.")
    lines.append("")
    return "\n".join(lines)


def get_fact_checking_prompt(
    introduction: str,
    original_articles: list
) -> tuple[str, str]:
    """Get prompts for fact-checking generated introduction

    Args:
        introduction: Generated introduction text
        original_articles: List of original articles used

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    system_prompt = """You are a fact-checking expert for medical research writing.
Your task is to verify that the introduction accurately represents the provided source material.

Provide your response as a JSON object with the following structure:
{
    "overall_accuracy": "HIGH/MEDIUM/LOW",
    "issues": [
        {
            "type": "MISSING_CITATION|MISREPRESENTATION|NUMERICAL_ERROR|UNFOUNDED_CLAIM",
            "description": "Description of the issue",
            "paragraph_number": X,
            "suggestion": "How to fix it"
        }
    ],
    "summary": "Brief overall assessment"
}"""

    articles_text = _format_articles_for_prompt(original_articles)

    user_prompt = f"""Fact-check the following introduction against the provided source articles.

Verify:
1. All numerical claims (prevalence, percentages, sample sizes) are supported by the articles
2. All statements about findings are accurately represented
3. No claims are made that are not supported by the provided articles
4. Citations [N] correspond to appropriate articles
5. Direct quotes or very specific findings reference the correct articles

Source Articles:
{articles_text}

Introduction to Check:
{introduction}

Provide your fact-checking assessment as JSON."""

    return system_prompt, user_prompt
