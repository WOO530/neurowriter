"""Prompt templates for introduction generation"""


def get_introduction_generation_prompt(
    parsed_topic: dict,
    selected_articles: list,
    output_instructions: dict = None,
    landscape: dict = None
) -> tuple[str, str]:
    """Get system and user prompts for introduction generation

    Args:
        parsed_topic: Parsed topic with disease, data_type, methodology, etc.
        selected_articles: List of selected article metadata dictionaries
        output_instructions: Optional output formatting instructions
        landscape: Optional literature landscape analysis (from deep research)

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    system_prompt = """You are a lead author of a peer-reviewed medical research article in neuroscience and psychiatry, published in top-tier journals (Nature Medicine, NEJM, JAMA Psychiatry, Lancet Neurology).

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
6. Citation Density: Aim for approximately one citation per 60-80 words (medical baseline is ~1 per 95 words, but introduction should be denser). For a 900-1300 word introduction, expect 12-20+ distinct citation points.
7. Quality Over Quantity Per Point: MAXIMUM 5 studies per single citation point [1-5]. Do not pad citations unnecessarily; use only most impactful, directly relevant papers.
8. Sequential Numbering: Number citations sequentially as they appear in the text. If revisions occur, renumber for consistency.

STRUCTURAL PRINCIPLES:
- Each paragraph should have a clear topic sentence
- Logical flow between paragraphs (no abrupt transitions)
- Build from general (disease background) to specific (your research rationale)
- End with a clear statement of study aims using: "In this study, we [aimed to/hypothesized that/investigated...]"

DO NOT:
- Speculate beyond what is supported by provided articles
- Use citations for definitions of the disorder or universally accepted facts
- Include references not in the provided list
- Vary the meaning of claims from what the original articles state"""

    # Format articles with enhanced context for the prompt
    articles_text = _format_articles_for_prompt(selected_articles)

    disease = parsed_topic.get("disease", "")
    data_type = parsed_topic.get("data_type", "")
    methodology = parsed_topic.get("methodology", "")
    outcome = parsed_topic.get("outcome", "")
    additional_context = parsed_topic.get("additional_context", "")
    concept_hierarchy = parsed_topic.get("concept_hierarchy", [])
    key_intervention_or_focus = parsed_topic.get("key_intervention_or_focus", "")

    # Determine primary focus for focus drift prevention
    primary_focus = concept_hierarchy[-1] if concept_hierarchy else key_intervention_or_focus or disease

    # Few-shot examples for academic tone
    academic_examples = """EXAMPLE PARAGRAPH STYLES (for tone reference, do NOT cite):

Example 1 - Disease Background:
"Major depressive disorder (MDD) is a prevalent and debilitating psychiatric condition, affecting approximately 280 million individuals globally [1] and ranking among the leading causes of disability worldwide [2]. Despite the availability of multiple pharmacological interventions, treatment outcomes remain suboptimal [3], with approximately one-third of patients failing to achieve adequate remission following first-line antidepressant therapy [4-5]."

Example 2 - Technology/Methodology:
"Electroencephalography (EEG), a non-invasive neurophysiological modality with high temporal resolution, has emerged as a promising tool for identifying neural biomarkers associated with treatment response in psychiatric disorders [6-8]. Recent advances in deep learning architectures, particularly convolutional neural networks (CNNs) and recurrent neural networks (RNNs), have demonstrated remarkable capacity for extracting complex spatiotemporal features from raw EEG signals that may elude conventional analytical approaches [9,10]."

Example 3 - Unmet Need/Rationale:
"Although substantial progress has been made in characterizing EEG abnormalities in MDD [11], the translation of these findings to clinical practice has been limited [12]. Current diagnostic procedures rely exclusively on clinical assessment, lacking objective biological biomarkers [13]. The identification of reliable EEG-based predictors of antidepressant response could substantially improve treatment selection and outcomes [14], representing a significant unmet clinical need."""

    # Format landscape context if available
    landscape_context = ""
    if landscape:
        landscape_context = _format_landscape_context(landscape, parsed_topic)

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
REQUIRED SECTIONS (in order):
1. **Disease Background & Clinical Burden ({disease}{f' — specifically {primary_focus}' if primary_focus.lower() != disease.lower() else ''})**: Prevalence, incidence, epidemiology, disease burden, mortality/morbidity impact, current management landscape
2. **Current Diagnostic & Treatment Limitations**: Why current approaches are inadequate, gaps in treatment response prediction, challenges in clinical practice
3. **Neurophysiological Biomarkers**: Why {data_type} is relevant, what abnormalities have been observed, how biomarkers could improve outcomes
4. **Machine Learning & {methodology} Applications**: State of the art in applying {methodology} to psychiatric conditions, what previous studies have shown, technological advances
5. **Integration & Unmet Needs**: Synthesis - why combining {data_type} with {methodology} is important, what remains unclear, why this specific research is needed
6. **Study Rationale & Aims**: Clear statement of your study's purpose, novelty, and expected contributions

SPECIFICATIONS:
- LENGTH: 4-6 paragraphs, approximately 900-1300 words
- TONE: Follow the academic examples provided above
- CITATIONS: Target 12-20+ citation points across the introduction. Use 15-35 unique references total. See citation rules above for flexible strategy (1-5 per point, 30-40% multiple citations)
- FLOW: Smooth transitions between sections; avoid abrupt topic changes
- VOCABULARY: Use precise medical/neuroscience terminology

{academic_examples}

{landscape_context}

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


def _format_articles_for_prompt(articles: list) -> str:
    """Format articles for inclusion in prompt

    Args:
        articles: List of article dictionaries

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
        if len(abstract) > 2000:
            abstract = abstract[:1997] + "..."

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
