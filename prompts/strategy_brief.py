"""Prompt templates for writing strategy generation"""

from prompts.genre_config import GENRE_PROFILES
from prompts.modality_config import SECTION_NAMES


def get_writing_strategy_prompt(
    topic_analysis: dict,
    reference_pool: list,
    landscape: dict,
    modality: str = "eeg",
    genre: str = "research_introduction",
) -> tuple[str, str]:
    """Get prompts for generating a writing strategy / outline

    Args:
        topic_analysis: Parsed topic analysis
        reference_pool: Selected reference papers
        landscape: Literature landscape analysis
        modality: Detected modality ("eeg", "psg", or "mixed")
        genre: Document genre key from GENRE_PROFILES

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    profile = GENRE_PROFILES.get(genre, GENRE_PROFILES["research_introduction"])
    section = SECTION_NAMES.get(modality, SECTION_NAMES["eeg"])

    # Build system prompt from genre profile
    role_text = profile["role"].format(
        domain_label=section["domain_label"],
        journal_examples=section["journal_examples"],
    )

    system_prompt = f"""{role_text}

{profile["structure_guide"]}

{profile["proportion_guide"]}

SYNTHESIS PLANNING GUIDE:
When planning key_points for each paragraph, formulate them as SYNTHESIZED claims
that combine findings from multiple papers. Use one of these synthesis patterns:

  CONSENSUS: "Multiple studies agree that X [refs]" — when 3+ papers report similar findings
  TREND: "The field has evolved from X to Y [refs]" — when papers show temporal progression
  META-FINDING: "Across N studies, the range of X is Y-Z% [refs]" — when papers report different numbers on the same metric
  CONVERGENT: "Evidence from both A and B approaches supports X [refs]" — when different methodologies reach similar conclusions
  LIMITATION-SYNTHESIS: "Existing studies share common constraints including X, Y, Z [refs]" — when multiple papers have similar weaknesses

BAD key_point: "Smith et al. found 85% accuracy using CNN"
GOOD key_point: {{"claim": "Deep learning approaches have achieved 85-92% accuracy across multiple architectures", "supporting_papers": [1,3,5], "synthesis_pattern": "CONSENSUS"}}

Your task is to create a detailed paragraph-by-paragraph outline with:
- A clear narrative arc from broad context to specific study rationale
- Explicit mapping of which references support each paragraph
- Smooth transition plans between paragraphs
- key_points as SYNTHESIZED claims (preferably as dict with claim, supporting_papers, synthesis_pattern)

Respond ONLY with valid JSON, no additional text or markdown formatting."""

    refs_text = _format_refs_for_strategy(reference_pool)

    key_findings = landscape.get("key_findings", [])
    knowledge_gaps = landscape.get("knowledge_gaps", [])
    trends = landscape.get("methodological_trends", [])

    findings_text = "\n".join(f"  - {f}" for f in key_findings[:8]) if key_findings else "  (none identified)"
    gaps_text = "\n".join(f"  - {g}" for g in knowledge_gaps[:6]) if knowledge_gaps else "  (none identified)"
    trends_text = "\n".join(f"  - {t}" for t in trends[:4]) if trends else "  (none identified)"

    # Focus drift prevention: extract concept hierarchy and primary focus
    disease = topic_analysis.get('disease', '') or ""
    concept_hierarchy = topic_analysis.get('concept_hierarchy', []) or []
    disease_subtypes = topic_analysis.get('disease_subtypes', []) or []
    primary_focus = concept_hierarchy[-1] if concept_hierarchy else (topic_analysis.get('key_intervention_or_focus', '') or disease or "")

    concept_chain = ""
    if concept_hierarchy:
        concept_chain = " → ".join(concept_hierarchy)

    focus_constraint = ""
    if primary_focus and primary_focus.lower() != disease.lower():
        focus_constraint = f"""
FOCUS CONSTRAINT (BINDING):
Your strategy MUST center on "{primary_focus}" specifically, NOT on {disease} in general.
- Paragraph 1 may introduce the broader disease, but MUST narrow to the specific focus by end
- All subsequent paragraphs MUST be about or directly relevant to: {primary_focus}
- Prioritize papers about {primary_focus} over general {disease} papers
"""

    subtypes_text = ""
    if disease_subtypes:
        subtypes_text = "\n- Disease subtypes: " + ", ".join(disease_subtypes[:5])

    additional_context = topic_analysis.get("additional_context", "")

    user_prompt = f"""Create a paragraph-by-paragraph writing strategy for an introduction on:

RESEARCH TOPIC:
- Disease: {disease}
- Data type: {topic_analysis.get('data_type', '')}
- Methodology: {topic_analysis.get('methodology', '')}
- Outcome: {topic_analysis.get('outcome', '')}
- Focus: {topic_analysis.get('key_intervention_or_focus', '')}
- Primary focus (most specific): {primary_focus}{subtypes_text}
{f"- Additional context: {additional_context}" if additional_context else ""}
{f"- Concept hierarchy: {concept_chain}" if concept_chain else ""}
{focus_constraint}
LITERATURE LANDSCAPE:
Key findings:
{findings_text}

Knowledge gaps:
{gaps_text}

Methodological trends:
{trends_text}

AVAILABLE REFERENCES ({len(reference_pool)} papers):
{refs_text}

Return JSON:
{{
    "paragraphs": [
        {{
            "paragraph_number": 1,
            "topic": "Disease background & clinical burden",
            "key_points": [
                {{"claim": "Synthesized claim text", "supporting_papers": [1, 3, 5], "synthesis_pattern": "CONSENSUS"}},
                {{"claim": "Another synthesized claim", "supporting_papers": [2, 4], "synthesis_pattern": "TREND"}},
                "Plain string key_point is also acceptable"
            ],
            "supporting_papers": [1, 2, 3, 4, 5],
            "transition_to_next": "How this connects to the next paragraph"
        }},
        ...
    ],
    "narrative_arc": "Brief description of the overall narrative flow from broad to specific",
    "estimated_word_count": 550,
    "total_references_planned": 25
}}

REQUIREMENTS:
- Plan 3-5 paragraphs (target 500-600 words — concise, every sentence must carry weight; average 15-25 words per sentence, no sentence exceeding 35 words)
- Each paragraph must reference at least 3 papers by their [N] number
- The final paragraph must end with study aims
- Total planned references should be 10-25 unique papers
- Narrative must narrow from broad to specific within first 1-2 paragraphs, then STAY specific
- Narrative arc should flow: disease burden -> clinical gaps -> prior research landscape (classical, ML/DL, cross-modality findings & limitations) -> current approach rationale -> study aims
- Content proportions must follow the CONTENT PROPORTION CONSTRAINTS above"""

    return system_prompt, user_prompt


def _format_refs_for_strategy(reference_pool: list) -> str:
    """Format references compactly for strategy prompt"""
    lines = []
    for i, paper in enumerate(reference_pool, 1):
        authors = ", ".join(paper.get("authors", [])[:2])
        if len(paper.get("authors", [])) > 2:
            authors += " et al."
        title = paper.get("title", "")[:80]
        year = paper.get("pub_year", "")
        journal = paper.get("journal_iso", paper.get("journal", ""))
        lines.append(f"[{i}] {authors}. {title}... {journal} {year}")
    return "\n".join(lines)
