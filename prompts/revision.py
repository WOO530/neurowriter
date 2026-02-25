"""Prompt templates for revision requests"""


def get_revision_prompt(
    current_introduction: str,
    revision_request: str,
    articles_used: list
) -> tuple[str, str]:
    """Get system and user prompts for revising introduction

    Args:
        current_introduction: Current introduction text
        revision_request: User's revision request
        articles_used: List of available articles for citation

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    system_prompt = """You are an expert medical editor revising a research paper introduction.

Your task is to incorporate the user's revision request while:
1. Maintaining the same high academic standards
2. Using ONLY the provided references (do NOT add new ones)
3. Preserving the overall structure and flow when possible
4. Ensuring all citations remain valid and accurate
5. Updating citation numbers if necessary due to additions/removals
6. Separating each paragraph with a blank line (double newline)

CITATION RULES:
- Every factual claim must have a citation
- Multiple claims in one sentence = multiple separate citations
- When revising, renumber citations sequentially from 1
- If the revision requires information not in provided articles, note this limitation"""

    # Format articles for reference
    articles_text = _format_articles_for_revision(articles_used)

    user_prompt = f"""Please revise the following introduction according to this request:

REVISION REQUEST:
"{revision_request}"

CURRENT INTRODUCTION:
{current_introduction}

AVAILABLE REFERENCES (use ONLY these):
{articles_text}

INSTRUCTIONS:
1. Apply the revision request to the introduction
2. Maintain all citation accuracy - verify each [N] refers to the correct article
3. Renumber citations sequentially if needed
4. Preserve academic tone and scientific rigor
5. If the revision cannot be completed with available references, note what information is missing
6. Separate each paragraph with a blank line

Provide the revised introduction with updated citations."""

    return system_prompt, user_prompt


def _format_articles_for_revision(articles: list) -> str:
    """Format articles for revision prompt

    Args:
        articles: List of article dictionaries

    Returns:
        Formatted string for prompt
    """
    formatted = []

    for i, article in enumerate(articles, 1):
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
            f"[{i}] {authors}. {title}. {journal}. {year}.\n"
            f"    {abstract}"
        )

    return "\n\n".join(formatted)
