"""Prompt templates for topic parsing"""

from prompts.modality_config import QUERY_EXAMPLES


def get_topic_parsing_prompt(research_topic: str, modality: str = "eeg") -> tuple[str, str]:
    """Get system and user prompts for comprehensive topic parsing

    Args:
        research_topic: The research topic from user
        modality: Detected modality ("eeg", "psg", or "mixed")

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    system_prompt = """You are an expert medical researcher specializing in comprehensive literature analysis.
Your task is to analyze a research topic in DEPTH and create a strategic research blueprint.

You must respond ONLY with valid JSON, with no additional text or markdown formatting.

Provide maximum depth and specificity. This analysis will guide a comprehensive literature search and introduction writing strategy.

SEARCH QUERY STRATEGY:
- Output raw PubMed queries only — no labels, prefixes, or descriptions
- Use Boolean operators explicitly: (term1 OR term2) AND (term3 OR term4)
- Include OR for synonyms and abbreviations in EVERY query (e.g., "(REM sleep behavior disorder OR RBD OR iRBD)")
- Keep queries 4-8 terms long (excluding Boolean operators). Avoid keyword chains >8 terms — they return zero results on PubMed
- Vary breadth: 3-4 broad queries (2-4 terms), 8-9 moderate queries (4-6 terms), 2-3 narrow queries (6-8 terms)
- Include at least 2 review/meta-analysis queries with explicit Boolean structure
- Combine terms from different angles: epidemiology, mechanism, biomarker, methodology, clinical outcome
- Avoid repeating the same structure — each query should target a distinct literature segment"""

    # Use EEG examples as default for mixed modality
    query_key = modality if modality in QUERY_EXAMPLES else "eeg"
    query_examples = QUERY_EXAMPLES[query_key]

    user_prompt = f"""Conduct a DEEP hierarchical analysis of this research topic:

"{research_topic}"

Return a JSON object with EXACTLY this structure (all fields required):
{{
    "disease": "Main disease/condition name",
    "disease_subtypes": ["Disease subtype 1", "Disease subtype 2", "..."],

    "key_intervention_or_focus": "Primary intervention/focus (e.g., 'Clozapine' or 'Deep Learning')",

    "data_type": "Type of data used (e.g., 'EEG', 'fMRI')",
    "methodology": "Analysis methodology category — broad level only (e.g., 'Deep Learning', 'Machine Learning', 'Statistical Analysis'). Do NOT specify a particular architecture (not CNN, LSTM, Transformer, etc.)",
    "outcome": "Primary outcome/prediction target",

    "concept_hierarchy": [
        "Broadest concept (e.g., 'Schizophrenia')",
        "Intermediate concept (e.g., 'Treatment-resistant schizophrenia')",
        "Specific concept (e.g., 'Clozapine-resistant schizophrenia')",
        "Ultra-specific concept (e.g., 'Predicting clozapine non-response using EEG biomarkers')",
        "... more levels as needed ..."
    ],

    "key_concepts": [
        "concept 1: brief explanation why important",
        "concept 2: brief explanation",
        "... at least 10-15 key concepts ..."
    ],

    "knowledge_areas_to_research": [
        "Area 1 (e.g., disease prevalence and burden)",
        "Area 2 (e.g., current treatment approaches and limitations)",
        "... comprehensive coverage of what introduction should discuss ..."
    ],

{query_examples},

    "expected_discovery_areas": {{
        "landmark_findings": "What are the seminal/foundational studies you hope to find?",
        "ongoing_debates": "What are the controversies or knowledge gaps in this area?",
        "methodological_gaps": "What analysis approaches are NOT yet well-developed?",
        "clinical_applications": "What clinical impact would success have?"
    }}
}}

REQUIREMENTS:
- concept_hierarchy: Must have AT LEAST 5 levels, from very broad to ultra-specific
- search_queries: Must have 15 DISTINCT queries covering all angles
- key_concepts: At least 12 concepts that a comprehensive introduction should address
- knowledge_areas_to_research: At least 8 major areas to cover"""

    return system_prompt, user_prompt


def get_landscape_analysis_prompt(abstracts_by_category: dict) -> tuple[str, str]:
    """Get prompts for analyzing collected literature landscape

    Args:
        abstracts_by_category: Dict of category -> list of article dicts

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    system_prompt = """You are an expert at synthesizing scientific literature into comprehensive research landscapes.

Analyze the provided collection of abstracts and identify:
- Key findings and trends across the field
- Landmark/foundational papers
- Knowledge gaps and controversies
- Methodological developments

Respond as JSON only with the provided structure."""

    abstracts_text = _format_abstracts_for_landscape(abstracts_by_category)
    total_abstracts = sum(len(v) for v in abstracts_by_category.values())

    user_prompt = f"""Based on this collection of {total_abstracts} research abstracts across {len(abstracts_by_category)} categories:

{abstracts_text}

Analyze the literature landscape and return JSON:
{{
    "field_overview": "2-3 paragraph summary of current state of this specific research area",

    "key_findings": [
        "Finding 1 (with evidence prevalence across papers)",
        "Finding 2",
        "... at least 8-10 key findings ..."
    ],

    "landmark_papers": [
        "PMID: Brief reason (e.g., '12345678: First to combine EEG with deep learning')",
        "... 5-10 foundational or highly cited papers ..."
    ],

    "knowledge_gaps": [
        "Gap 1: What is NOT well understood",
        "Gap 2",
        "... 5-8 major gaps ..."
    ],

    "methodological_trends": [
        "Trend 1: Evolution of analysis methods",
        "Trend 2",
        "... 3-5 important methodological shifts ..."
    ],

    "controversies_or_debates": [
        "Controversy 1",
        "... unresolved questions or conflicting findings ..."
    ],

    "underexplored_areas": [
        "Area 1: Something that should be researched more",
        "... 3-5 opportunities for novel research ..."
    ],

    "journal_distribution": "Summary of journal impact",
    "article_type_distribution": "% review vs original vs meta-analysis",
    "temporal_trend": "Are recent papers increasing? What's changing?",
    "recommendations_for_introduction": "What MUST be covered to show understanding of this specific field?"
}}"""

    return system_prompt, user_prompt


def _format_abstracts_for_landscape(abstracts_by_category: dict) -> str:
    """Format abstracts for landscape analysis

    Args:
        abstracts_by_category: Dict of category -> list of article dicts

    Returns:
        Formatted string
    """
    formatted = []
    total_papers = sum(len(v) for v in abstracts_by_category.values())

    # If no papers, return minimal prompt hint
    if total_papers == 0:
        return "[No abstracts provided - minimal analysis will be performed]"

    for category, papers in abstracts_by_category.items():
        if not papers:
            continue

        formatted.append(f"\n=== CATEGORY: {category.upper()} ({len(papers)} papers) ===\n")
        paper_count = 0
        for i, paper in enumerate(papers[:25], 1):  # Sample first 25 per category
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")[:1500]  # Truncate
            journal = paper.get("journal", "")
            year = paper.get("pub_year", "")
            pmid = paper.get("pmid", "")

            # Skip papers with no meaningful content
            if not title and not abstract:
                continue

            formatted.append(
                f"[{i}] PMID:{pmid} | {journal} {year}\n"
                f"    {title}\n"
                f"    {abstract}...\n"
            )
            paper_count += 1

        if paper_count == 0:
            formatted.append(f"    [No abstracts available for this category]\n")

    result = "\n".join(formatted)

    # If result is empty or very short, add context hint
    if not result or len(result) < 100:
        result = f"Minimal abstract data available ({total_papers} papers total). Please provide general landscape analysis.\n" + result

    return result
