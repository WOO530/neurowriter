"""Prompt templates for introduction generation"""

from prompts.modality_config import SECTION_NAMES, ACADEMIC_EXAMPLES
from core.deep_researcher import _CATEGORY_PATTERNS


def get_introduction_generation_prompt(
    parsed_topic: dict,
    selected_articles: list,
    output_instructions: dict = None,
    landscape: dict = None,
    modality: str = "eeg",
    writing_strategy: dict = None,
    evaluation_feedback: dict = None,
    unsupported_claims: list = None,
    user_feedback: str = "",
    current_introduction: str = "",
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
        user_feedback: Optional user-provided feedback for revision (high priority)
        current_introduction: Optional current introduction text for revision mode

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
- Keep sentences concise: average 15-25 words per sentence. NO sentence may exceed 35 words. Split long compound sentences into two.

CITATION RULES (CRITICAL):
1. Use ONLY the provided references. Do NOT fabricate or hallucinate any citations.
2. Reference Portfolio: Use 10-20 unique references total. Aim for purposeful, targeted citation coverage.
3. Citation Coverage: Cite specific empirical findings, statistics, and study results.
   Common knowledge (disease definitions, well-known prevalence), logical transitions,
   conceptual framing, and study aims do NOT require citations.
   Roughly 50-70% of sentences should have citations; the rest carry the narrative.
4. Multiple Independent Claims: When a single sentence contains multiple distinct factual claims, each MUST be cited separately.
   - BAD: "Depression affects 280 million people and is the leading cause of disability [1]."
   - GOOD: "Depression affects approximately 280 million people [1] and is among the leading causes of disability worldwide [2]."
5. Multiple Supporting Studies: When multiple studies support the same claim, cite them together using ranges: [3-5] or [3,4,5]. TARGET: 30-40% of your citations should be multiple (not single), showcasing consensus in the field.
6. Citation Density: Aim for approximately one citation per 80-120 words. For a 500-600 word introduction, expect 6-8 distinct citation points.
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
- Each paragraph should make 2-4 synthesized claims, NOT 5-8 individual study summaries
- Specific numbers from individual papers should be used sparingly — only when a finding is particularly noteworthy or landmark
- Group related findings thematically, not paper-by-paper

HOW TO SYNTHESIZE — follow this 5-step mental process for each paragraph:
  1. CLUSTER: Group 3-5 papers that address the same sub-question
  2. ABSTRACT: Identify the shared finding, trend, or consensus across the cluster
  3. RANGE: If papers report different numbers, present a range (e.g., "70-90%")
  4. QUALIFY: Add appropriate hedging ("evidence suggests", "consistently shown")
  5. CITE: Attach the grouped citation [N-M] to the synthesized claim

SYNTHESIS PATTERNS (use at least 2-3 of these across your introduction):
  CONSENSUS — Multiple studies agree on a finding:
    BAD: "Smith et al. found 85% accuracy [1]. Lee et al. achieved 90% [2]. Park et al. reported 88% [3]."
    GOOD: "Multiple studies have demonstrated classification accuracies of 85-90% using deep learning architectures [1-3]."

  TREND — Showing evolution or progression over time:
    BAD: "In 2018, CNNs were used [1]. In 2020, LSTMs were adopted [2]. In 2022, transformers emerged [3]."
    GOOD: "The field has progressively shifted from convolutional architectures to recurrent and attention-based models, with each generation improving temporal feature extraction [1-3]."

  LIMITATION-SYNTHESIS — Combining weaknesses from multiple studies into a unified gap:
    BAD: "Study A used only 50 subjects [1]. Study B lacked external validation [2]. Study C used single-site data [3]."
    GOOD: "Despite promising results, existing studies share common methodological constraints including limited sample sizes, single-site designs, and absent external validation [1-3]."

  CONVERGENT — Different methods reaching the same conclusion:
    BAD: "EEG spectral analysis showed alpha reduction [1]. fMRI revealed prefrontal hypoactivation [2]."
    GOOD: "Converging evidence from electrophysiological and neuroimaging studies points to prefrontal dysfunction as a key neural correlate [1,2]."
"""

    # Format articles with enhanced context for the prompt (800 char abstract limit for generation)
    # Group thematically for generation to encourage synthesis across papers
    articles_text = _format_articles_for_prompt(
        selected_articles, max_abstract_len=800, group_thematically=True
    )

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
    is_revision = bool(evaluation_feedback or unsupported_claims or user_feedback)
    if is_revision:
        feedback_context = _format_evaluation_feedback(
            evaluation_feedback, unsupported_claims, user_feedback=user_feedback
        )

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

    # Revision-specific instructions
    revision_instructions = ""
    current_draft_section = ""
    if is_revision:
        if current_introduction:
            revision_instructions = """
REVISION MODE (IMPORTANT — you are EDITING the CURRENT DRAFT below, NOT writing from scratch):
- The CURRENT DRAFT is provided below. EDIT it to address the specific issues in the feedback.
- PRESERVE paragraphs and sentences that scored well — only modify what needs fixing.
- FOCUS your changes on the specific issues identified in the feedback.
- The word limit is flexible for revisions: 500-650 words is acceptable if needed to address feedback.
- PRIORITY ORDER: Fix factual errors first, then completeness gaps, then other criteria.
- If fixing one criterion would hurt another, prioritize: factual_accuracy > completeness > depth > reference_density > others.

"""
            current_draft_section = f"""
CURRENT DRAFT (EDIT this — do NOT discard and rewrite from scratch):
{current_introduction}

"""
        else:
            revision_instructions = """
REVISION MODE (IMPORTANT — this is a REVISION, not a first draft):
- You are REVISING a previous draft based on specific feedback below.
- PRESERVE sections that scored well — do NOT rewrite everything from scratch.
- FOCUS your changes on the specific issues identified in the feedback.
- The word limit is flexible for revisions: 500-650 words is acceptable if needed to address feedback.
- PRIORITY ORDER: Fix factual errors first, then completeness gaps, then other criteria.
- If fixing one criterion would hurt another, prioritize: factual_accuracy > completeness > depth > reference_density > others.

"""

    # Final instruction varies based on revision mode
    final_instruction = "Write the introduction now:"
    if is_revision and current_introduction:
        final_instruction = "Revise the introduction above, addressing the issues identified while preserving the overall structure:"

    user_prompt = f"""{revision_instructions}{current_draft_section}Write a research paper introduction for a medical study on the following topic:

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
2. **Prior Research & Biomarker Evidence**: Survey of prior approaches applied to this topic — include classical statistical analyses, traditional biomarker studies, and modern computational methods (ML/DL). Describe candidate biomarkers and analytical approaches that have been explored, their reported performance, and critically their limitations. If relevant, include findings from other data modalities. Weave classical and modern findings into a coherent narrative rather than treating them as separate blocks.
3. **Current Approach & Rationale**: Why combining {data_type} with {methodology} can address the remaining limitations. Advantages of the proposed modality over alternatives; how the proposed methodology improves upon prior approaches. Conclude with specific study aims and expected contributions.
4. **Study Aims** (you may combine this with section 3 into a single paragraph): Clear statement of your study's purpose, novelty, and expected contributions

SPECIFICATIONS:
- LENGTH: 3-5 paragraphs, approximately 500-600 words. This is a STRICT upper limit — do NOT exceed 600 words. Write concisely: every sentence must earn its place.
- DENSITY: If approaching the word limit, cut granular numerical details and individual case descriptions first. Preserve the core logical flow: disease burden → clinical gaps → prior research & their limitations → current approach → study aims.

CONTENT PRIORITY (what to KEEP vs what to CUT when space is tight):
KEEP (core narrative):
- Disease significance & unmet clinical need
- Key limitations of current approaches
- Core prior research findings that set up your study's rationale
- Knowledge gaps your study addresses
- Study aims
CUT FIRST (expendable details):
- Individual study sample sizes and p-values
- Exhaustive lists of specific architectures or accuracy numbers
- Redundant prevalence statistics from multiple sources
- Overly detailed mechanism explanations
- Background facts that any domain expert already knows
When in doubt, prefer ONE well-synthesized claim over THREE individual study details.
- FORMAT: Separate each paragraph with a blank line. Do NOT output as a single block of text.
- TONE: Follow the academic examples provided above
- CITATIONS: Target 6-9 citation points across the introduction. Use 10-20 unique references total. See citation rules above for flexible strategy (1-5 per point, 30-40% multiple citations)
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

{final_instruction}"""

    return system_prompt, user_prompt


def _format_landscape_context(landscape: dict, parsed_topic: dict) -> str:
    """Format literature landscape context for prompt

    Args:
        landscape: Landscape analysis dictionary
        parsed_topic: Parsed topic dictionary

    Returns:
        Formatted landscape context string
    """
    lines = ["LITERATURE LANDSCAPE CONTEXT (address the most important items — prioritize those most relevant to your study rationale):"]

    # Field overview
    field_overview = landscape.get("field_overview", "")
    if field_overview:
        lines.append(f"\nField Overview:")
        lines.append(f"  {field_overview[:800]}")

    # Key findings
    key_findings = landscape.get("key_findings", [])
    if key_findings:
        lines.append(f"\nKey Findings in the Literature ({len(key_findings)} identified — address the most important):")
        for i, finding in enumerate(key_findings[:10], 1):
            lines.append(f"  #{i}: {finding[:300]}")

    # Knowledge gaps
    knowledge_gaps = landscape.get("knowledge_gaps", [])
    if knowledge_gaps:
        lines.append(f"\nIdentified Knowledge Gaps ({len(knowledge_gaps)} identified — address the most important):")
        for i, gap in enumerate(knowledge_gaps[:10], 1):
            lines.append(f"  #{i}: {gap[:300]}")

    # Methodological trends
    trends = landscape.get("methodological_trends", [])
    if trends:
        lines.append(f"\nMethodological Trends:")
        for i, trend in enumerate(trends[:5], 1):
            lines.append(f"  #{i}: {trend[:300]}")

    lines.append("\nIMPORTANT: Address the 5-7 most important key findings and 3-4 most relevant knowledge gaps. Synthesize related items together rather than listing each separately.\n")
    return "\n".join(lines)


def _format_articles_for_prompt(
    articles: list, max_abstract_len: int = 2000, group_thematically: bool = False
) -> str:
    """Format articles for inclusion in prompt

    Args:
        articles: List of article dictionaries
        max_abstract_len: Maximum abstract length in characters.
            Use 800 for generation (prevents over-extraction of details).
            Use 2000 for fact-checking (needs full abstract for verification).
        group_thematically: If True, group articles by category with headers
            and synthesis hints. Use for generation, not for fact-checking.

    Returns:
        Formatted string for prompt
    """
    # Build per-article formatted strings (keyed by 1-based index)
    article_strings = {}
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

        article_strings[i] = (
            f"[{i}]{type_label} {authors}. {title}. {journal}. {year}. PMID: {pmid}\n"
            f"    Abstract: {abstract}"
        )

    if not group_thematically:
        return "\n\n".join(article_strings[i] for i in sorted(article_strings))

    # --- Thematic grouping ---
    _CATEGORY_LABELS = {
        "epidemiology": "EPIDEMIOLOGY & DISEASE BURDEN",
        "clinical_treatment": "CLINICAL TREATMENT & MANAGEMENT",
        "biomarkers": "BIOMARKERS & NEUROPHYSIOLOGY",
        "methodology": "METHODOLOGY",
        "reviews": "REVIEWS & META-ANALYSES",
    }

    # Classify each article into categories
    categorized: dict[str, list[int]] = {cat: [] for cat in _CATEGORY_PATTERNS}
    uncategorized: list[int] = []

    for i, article in enumerate(articles, 1):
        text = (
            (article.get("title", "") + " " + article.get("abstract", "")).lower()
        )
        matched = False
        for cat, keywords in _CATEGORY_PATTERNS.items():
            if any(kw in text for kw in keywords):
                categorized[cat].append(i)
                matched = True
                break  # assign to first matching category only
        if not matched:
            uncategorized.append(i)

    # Build grouped output
    sections = []
    for cat, indices in categorized.items():
        if not indices:
            continue
        label = _CATEGORY_LABELS.get(cat, cat.upper())
        sections.append(f"--- {label} ---")
        for idx in indices:
            sections.append(article_strings[idx])
        sections.append("(Synthesize findings across these papers into unified claims.)\n")

    if uncategorized:
        sections.append("--- OTHER ---")
        for idx in uncategorized:
            sections.append(article_strings[idx])
        sections.append("(Synthesize findings across these papers into unified claims.)\n")

    return "\n\n".join(sections)


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
                if isinstance(kp, dict):
                    # New synthesized key_point format
                    claim = kp.get("claim", "")
                    refs = kp.get("supporting_papers", [])
                    pattern = kp.get("synthesis_pattern", "")
                    refs_str = ",".join(str(r) for r in refs) if refs else ""
                    pattern_label = f"[{pattern}] " if pattern else ""
                    lines.append(f"    - {pattern_label}\"{claim[:150]}\" (refs: [{refs_str}])")
                else:
                    lines.append(f"    - {str(kp)[:150]}")
            if supporting:
                refs_str = ", ".join(str(s) for s in supporting[:8])
                lines.append(f"    Suggested refs: [{refs_str}] (reference numbers are from original pool — re-match by topic if pool has changed)")
            if transition:
                lines.append(f"    Transition: {transition[:100]}")

    lines.append("")
    return "\n".join(lines)


def _format_evaluation_feedback(
    evaluation_feedback: dict,
    unsupported_claims: list,
    user_feedback: str = ""
) -> str:
    """Format evaluation feedback and unsupported claims for self-evolution

    Args:
        evaluation_feedback: Evaluation results dict with scores and feedback
        unsupported_claims: List of claim dicts with 'claim', 'issue', 'needed_evidence'
        user_feedback: Optional user-provided feedback (high priority)

    Returns:
        Formatted feedback context string
    """
    lines = ["PREVIOUS ISSUES (MUST address in this revision):"]

    if user_feedback:
        lines.append("\nUSER FEEDBACK (high priority — address this first):")
        lines.append(f"  {user_feedback.strip()}")

    if evaluation_feedback:
        scores = evaluation_feedback.get("scores", {})
        feedback = evaluation_feedback.get("feedback", {})
        improvements = evaluation_feedback.get("improvements", [])

        # Extract criteria with low scores
        weak_criteria = []
        for criterion, score in scores.items():
            if isinstance(score, (int, float)) and score < 8:
                criterion_feedback = feedback.get(criterion, "")
                weak_criteria.append((criterion, score, criterion_feedback))

        if weak_criteria:
            lines.append("\nLow-scoring criteria:")
            for criterion, score, fb in weak_criteria:
                # Generous limits to preserve actionable detail
                max_len = 800 if criterion in ("factual_accuracy", "completeness") else 600
                lines.append(f"  - {criterion.replace('_', ' ').upper()} (score: {score}/10): {fb[:max_len]}")

        if improvements:
            lines.append("\nSuggested improvements:")
            for imp in improvements[:5]:
                imp_text = ""
                if isinstance(imp, str):
                    imp_text = imp
                elif isinstance(imp, dict):
                    imp_text = imp.get('suggestion', imp.get('improvement', str(imp)))
                max_len = 600
                lines.append(f"  - {imp_text[:max_len]}")

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
