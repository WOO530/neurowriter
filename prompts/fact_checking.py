"""Prompt templates for fact checking"""
import json


def get_fact_checking_prompts(
    introduction: str,
    articles: list
) -> tuple[str, str]:
    """Get system and user prompts for fact checking

    Args:
        introduction: Generated introduction text
        articles: List of source articles

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    system_prompt = """You are an expert fact-checker for medical research papers.
Your task is to verify that statements in the introduction are supported by the provided source articles.

Respond as JSON with this exact structure:
{
    "overall_accuracy": "HIGH|MEDIUM|LOW",
    "issues_found": [
        {
            "issue_type": "MISSING_CITATION|INACCURACY|UNSUPPORTED_CLAIM|NUMERICAL_ERROR",
            "description": "Description of issue",
            "suggested_fix": "How to fix it"
        }
    ],
    "citations_verified": {
        "[1]": true/false,
        "[2]": true/false
    },
    "summary": "Brief assessment of overall accuracy"
}

If no issues are found, return an empty issues_found array."""

    # Format articles for reference
    articles_text = _format_articles_for_fact_check(articles)

    user_prompt = f"""Fact-check this medical research introduction against the provided source articles.

Verify these specific points:
1. All numerical claims (prevalence, percentages, incidence) have citations
2. Those citations correspond to the actual articles in the source list
3. The specific claims made are supported by those articles' abstracts
4. No claims contradict the source material
5. Direct attributions are accurate

Source Articles (use these ONLY to verify claims):
{articles_text}

Introduction to verify:
{introduction}

Respond with JSON assessment. Focus on accuracy of factual claims, not writing quality."""

    return system_prompt, user_prompt


def _format_articles_for_fact_check(articles: list) -> str:
    """Format articles for fact-checking prompt

    Args:
        articles: List of article dictionaries

    Returns:
        Formatted string
    """
    formatted = []

    for i, article in enumerate(articles, 1):
        pmid = article.get("pmid", "Unknown")
        authors = ", ".join(article.get("authors", [])[:2])
        if len(article.get("authors", [])) > 2:
            authors += " et al."

        title = article.get("title", "")
        journal = article.get("journal", "")
        year = article.get("pub_year", "")

        abstract = article.get("abstract", "")
        if len(abstract) > 1500:
            abstract = abstract[:1497] + "..."

        formatted.append(
            f"[{i}] PMID: {pmid}\n"
            f"    Authors: {authors}\n"
            f"    Title: {title}\n"
            f"    Journal: {journal} ({year})\n"
            f"    Abstract: {abstract}"
        )

    return "\n\n".join(formatted)


def get_claim_citation_mapping_prompt(
    introduction: str,
    articles: list
) -> tuple[str, str]:
    """Get prompts for verifying claim-citation mappings

    Extracts each factual claim with citations from the introduction and
    verifies whether the cited article abstracts actually support that claim.

    Args:
        introduction: Generated introduction text
        articles: List of source articles

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    system_prompt = """You are an expert at verifying claim-citation mappings in medical research papers.

For each factual claim in the introduction that has a citation, classify the claim-citation relationship using a THREE-TIER system:

1. **SUPPORTED** (is_supported=true, support_level="direct"): The abstract explicitly states or directly supports the claim. Numbers match, directions match, findings are clearly reported.

2. **REASONABLE** (is_supported=true, support_level="reasonable"): The claim is a reasonable inference, paraphrase, or synthesis consistent with the abstract. This includes:
   - Paraphrasing findings in different words while preserving meaning
   - Synthesizing across multiple cited papers into a unified claim
   - Drawing reasonable inferences from reported data (e.g., if a study reports "accuracy 85-92%", citing "high accuracy" is reasonable)
   - Using approximate language for specific numbers ("approximately 30%" when abstract says "29.3%")
   - Abstracts are truncated — if the claim is consistent with the abstract's topic and direction, give benefit of the doubt

3. **UNSUPPORTED** (is_supported=false, support_level="unsupported"): The abstract CONTRADICTS the claim, OR the paper is about a completely different topic, OR the claim attributes specific findings that the abstract explicitly does NOT report.

KEY PRINCIPLE: is_supported=false means "the abstract CONTRADICTS or is IRRELEVANT to the claim", NOT "the abstract doesn't explicitly mention every detail."

Respond ONLY with valid JSON, no additional text or markdown formatting."""

    articles_text = _format_articles_for_fact_check(articles)

    user_prompt = f"""Extract ALL factual claims with citations from this introduction, then verify each claim against the cited article's abstract.

Source Articles:
{articles_text}

Introduction:
{introduction}

Return JSON:
{{
    "claim_mappings": [
        {{
            "claim": "The exact claim text from the introduction",
            "citation_numbers": [1, 2],
            "is_supported": true,
            "support_level": "direct|reasonable|unsupported",
            "issue": "null if supported, otherwise describe the specific contradiction or irrelevance"
        }},
        ...
    ],
    "numerical_mismatches": [
        {{
            "claim": "The claim with a number",
            "citation_number": 1,
            "claimed_value": "The value stated in the introduction",
            "actual_value": "The value in the abstract (or 'not found')",
            "severity": "minor|major"
        }},
        ...
    ]
}}

REQUIREMENTS:
- Extract EVERY factual claim that has a citation [N]
- For each claim, check ALL cited articles
- Use the THREE-TIER classification:
  * "direct": abstract explicitly states the claim
  * "reasonable": claim is a fair paraphrase, synthesis, or inference consistent with the abstract
  * "unsupported": abstract CONTRADICTS the claim or is about a completely different topic
- Mark is_supported=false ONLY when the abstract contradicts the claim or is entirely irrelevant — NOT when the abstract merely lacks explicit detail
- For multi-paper citations [1-3], the COMBINED evidence across all cited papers should support the synthesized claim
- Numerical mismatches should be listed separately with severity
- minor: rounding differences, approximate vs exact values, or reasonable ranges
- major: completely different numbers or contradictory findings"""

    return system_prompt, user_prompt


def get_verification_prompt(claim: str, pmid: str, article_abstract: str) -> tuple[str, str]:
    """Get prompts to verify a specific claim against an article

    Args:
        claim: The claim to verify
        pmid: PMID of the article
        article_abstract: Abstract of the article

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    system_prompt = "You are a medical fact-checker. Verify if the given claim is supported by the article abstract. Respond with JSON: {\"is_supported\": true/false, \"reason\": \"brief explanation\"}"

    user_prompt = f"""Verify this claim against the article:

Claim: {claim}

Article PMID: {pmid}
Abstract: {article_abstract}

Is the claim supported by this article? Respond with JSON."""

    return system_prompt, user_prompt
