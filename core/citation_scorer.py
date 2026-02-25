"""Citation scorer for ranking article importance"""
import logging
import re
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


# Stopwords for keyword extraction
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
    "be", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "dare",
    "this", "that", "these", "those", "it", "its", "not", "no", "nor",
    "so", "if", "then", "than", "too", "very", "just", "about", "above",
    "after", "again", "all", "also", "am", "any", "because", "before",
    "between", "both", "each", "few", "further", "here", "how", "into",
    "more", "most", "other", "our", "out", "over", "own", "same", "she",
    "he", "some", "such", "there", "they", "through", "under", "until",
    "up", "we", "what", "when", "where", "which", "while", "who", "whom",
    "why", "you", "your", "using", "based", "study", "studies", "research",
    "analysis", "results", "method", "methods", "approach", "novel",
    "new", "proposed", "however", "although", "including", "among",
})


class CitationScorer:
    """Score articles based on multiple criteria for selection

    Scoring weights:
      - Journal tier:       40 points (max)
      - Relevance:          25 points (max)
      - Recency:            20 points (max)
      - Abstract quality:   15 points (max)
    Total possible: 100
    """

    # Enhanced journal tier definitions (5 tiers)
    TIER_1_JOURNALS = {
        "Nature", "Nature Medicine", "Nature Neuroscience", "Science",
        "NEJM", "New England Journal of Medicine",
        "Lancet", "The Lancet", "JAMA", "Journal of the American Medical Association"
    }

    TIER_2_JOURNALS = {
        "JAMA Psychiatry", "JAMA Neurology",
        "Lancet Neurology", "The Lancet Neurology", "Lancet Psychiatry", "The Lancet Psychiatry",
        "Brain", "Biological Psychiatry", "Molecular Psychiatry",
        "American Journal of Psychiatry",
        "Annals of Neurology", "Sleep Medicine Reviews",
        "Movement Disorders", "Epilepsia",
        "Acta Neuropathologica", "Neurology Clinical Practice"
    }

    TIER_3_JOURNALS = {
        "NeuroImage", "Clinical Neurophysiology", "Journal of Neural Engineering",
        "Psychiatry Research", "Journal of Affective Disorders",
        "Journal of Neuroscience", "Neurology", "Sleep",
        "American Journal of Medical Genetics", "Neurobiology of Disease",
        "Neural Engineering & Translation", "Computational Psychiatry",
        "Sleep Medicine", "Journal of Clinical Sleep Medicine",
        "Sleep and Breathing", "Journal of Sleep Research", "Sleep Health",
        "Cerebral Cortex", "Frontiers in Neurology", "Journal of Neurology",
        "European Journal of Neurology", "Journal of Neuropsychiatry"
    }

    # Standard review/meta-analysis indicators
    REVIEW_KEYWORDS = {"review", "meta-analysis", "systematic review", "meta-analytic", "systematic", "scoping review"}

    # Regex for quantitative data markers in abstracts
    _QUANT_PATTERNS = [
        re.compile(r'\d+\.?\d*\s*%'),           # percentages
        re.compile(r'p\s*[<>=]\s*0?\.\d+', re.IGNORECASE),  # p-values
        re.compile(r'\bCI\b', re.IGNORECASE),    # confidence intervals
        re.compile(r'\bn\s*=\s*\d+', re.IGNORECASE),  # sample sizes
    ]

    # Pattern-based fallback tiers (checked when static tier matching fails)
    _TIER_PATTERNS = {
        30: ["nature ", "lancet ", "jama ", "annals of ", "bmj"],
        20: ["journal of neurology", "journal of psychiatry", "journal of sleep",
             "brain ", "neurobiol", "neuropsych", "cerebr", "cortex",
             "sleep ", "epilep", "movement dis"],
        15: ["frontiers in", "plos", "scientific reports", "bmc ",
             "international journal"],
    }

    def __init__(self):
        """Initialize citation scorer"""
        self.current_year = datetime.now().year
        self.dynamic_tiers: Dict[str, float] = {}

    def score_article(
        self,
        article: Dict,
        relevance_score: float = 0.0,
    ) -> float:
        """Calculate composite score for an article

        Args:
            article: Article metadata dictionary
            relevance_score: Keyword-based relevance (0.0-1.0)

        Returns:
            Composite score (0-100)
        """
        score = 0.0

        # 1. Journal impact (40 points max)
        score += self._score_journal(article.get("journal", ""))

        # 2. Relevance (25 points max)
        score += min(max(relevance_score, 0.0), 1.0) * 25

        # 3. Recency (20 points max)
        score += self._score_recency(article.get("pub_year", ""))

        # 4. Abstract quality (15 points max)
        score += self._score_abstract_quality(article.get("abstract", ""))

        return min(score, 100)

    def register_journal_tier(self, journal_name: str, score: float):
        """Register a dynamically discovered journal tier

        Args:
            journal_name: Journal name (stored lowercase)
            score: Score to assign (max 40)
        """
        self.dynamic_tiers[journal_name.lower()] = min(score, 40)

    def _score_journal(self, journal_name: str) -> float:
        """Score article based on journal impact tier

        Lookup order: dynamic tiers -> static tiers -> pattern fallback -> default.

        Args:
            journal_name: Name of the journal

        Returns:
            Score points (max 40)
        """
        if not journal_name:
            return 18  # Default to tier 3

        journal_lower = journal_name.lower()

        # Check dynamic tiers first (from landscape analysis)
        for dyn_name, dyn_score in self.dynamic_tiers.items():
            if dyn_name in journal_lower:
                return dyn_score

        # Check tier 1 (highest impact)
        for tier1_journal in self.TIER_1_JOURNALS:
            if tier1_journal.lower() in journal_lower:
                return 40

        # Check tier 2
        for tier2_journal in self.TIER_2_JOURNALS:
            if tier2_journal.lower() in journal_lower:
                return 30

        # Check tier 3
        for tier3_journal in self.TIER_3_JOURNALS:
            if tier3_journal.lower() in journal_lower:
                return 20

        # Pattern-based fallback
        for score, patterns in self._TIER_PATTERNS.items():
            if any(pat in journal_lower for pat in patterns):
                return score

        # Tier 4 (general/other PubMed journals)
        return 10

    def _score_recency(self, pub_year: str) -> float:
        """Score article based on publication year

        Args:
            pub_year: Publication year as string

        Returns:
            Score points (max 20)
        """
        try:
            if not pub_year or not pub_year.isdigit():
                return 10

            year = int(pub_year)
            years_old = self.current_year - year

            if years_old <= 3:
                return 20
            elif years_old <= 7:
                return 16
            elif years_old <= 15:
                return 10
            else:
                return 6

        except (ValueError, TypeError):
            return 10

    def _score_abstract_quality(self, abstract: str) -> float:
        """Score abstract quality based on length, structure, and quantitative data

        Args:
            abstract: Abstract text

        Returns:
            Score points (max 15)
        """
        if not abstract:
            return 0

        score = 0.0

        # Length score (0-6 points)
        length = len(abstract)
        if length >= 1500:
            score += 6
        elif length >= 1000:
            score += 5
        elif length >= 500:
            score += 4
        elif length >= 200:
            score += 2

        # Structured abstract bonus (0-5 points) — check for section labels
        section_labels = re.findall(r'\[(?:BACKGROUND|METHODS?|RESULTS?|CONCLUSIONS?|OBJECTIVES?|DESIGN|SETTING|PARTICIPANTS?|MAIN OUTCOME|INTERVENTIONS?|AIMS?|PURPOSE|FINDINGS?|INTERPRETATION)\]', abstract, re.IGNORECASE)
        unique_sections = len(set(label.upper() for label in section_labels))
        if unique_sections >= 3:
            score += 5
        elif unique_sections == 2:
            score += 3
        elif unique_sections == 1:
            score += 1

        # Quantitative data bonus (0-4 points)
        quant_hits = sum(1 for pat in self._QUANT_PATTERNS if pat.search(abstract))
        score += min(quant_hits, 4)

        return min(score, 15)

    def _detect_article_type(self, article: Dict) -> str:
        """Detect if article is review, meta-analysis, or original research

        Args:
            article: Article dictionary

        Returns:
            "review", "meta-analysis", "original", or "unknown"
        """
        title = (article.get("title") or "").lower()
        abstract = (article.get("abstract") or "").lower()

        text = f"{title} {abstract}"

        if any(keyword in text for keyword in ["meta-analysis", "meta-analytic", "meta analysis"]):
            return "meta-analysis"
        elif any(keyword in text for keyword in ["systematic review", "scoping review", "literature review"]):
            return "review"
        elif "original research" in text or "original article" in text:
            return "original"
        else:
            return "unknown"

    # ------------------------------------------------------------------
    # Keyword-based relevance scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_keywords(topic: str) -> List[str]:
        """Extract meaningful keywords and bigrams from a topic string

        Args:
            topic: Research topic string

        Returns:
            List of lowercase keywords/bigrams
        """
        if not topic:
            return []

        # Normalise and tokenise
        text = re.sub(r'[^a-zA-Z0-9\s-]', ' ', topic.lower())
        tokens = [t for t in text.split() if t not in _STOPWORDS and len(t) > 2]

        # Unigrams
        keywords = list(tokens)

        # Bigrams
        for i in range(len(tokens) - 1):
            keywords.append(f"{tokens[i]} {tokens[i+1]}")

        return keywords

    @staticmethod
    def compute_relevance_score(
        article: Dict,
        research_topic: str,
        topic_keywords: Optional[List[str]] = None
    ) -> float:
        """Compute keyword-based relevance score for an article

        Title matches are weighted 2x compared to abstract matches.

        Args:
            article: Article metadata dict
            research_topic: Original research topic string
            topic_keywords: Pre-extracted keywords (computed if None)

        Returns:
            Score between 0.0 and 1.0
        """
        if topic_keywords is None:
            topic_keywords = CitationScorer._extract_keywords(research_topic)

        if not topic_keywords:
            return 0.5  # neutral default

        title = (article.get("title") or "").lower()
        abstract = (article.get("abstract") or "").lower()

        title_hits = sum(1 for kw in topic_keywords if kw in title)
        abstract_hits = sum(1 for kw in topic_keywords if kw in abstract)

        # Title matches worth 2x
        weighted_hits = title_hits * 2 + abstract_hits
        max_possible = len(topic_keywords) * 3  # 2 (title) + 1 (abstract) per keyword

        raw = weighted_hits / max_possible if max_possible > 0 else 0.0
        # Apply sqrt to spread distribution
        return min(raw ** 0.5, 1.0)

    # ------------------------------------------------------------------
    # Ranking helpers
    # ------------------------------------------------------------------

    def rank_articles_by_category(
        self,
        category_results: Dict[str, List[Dict]],
        top_n: int = 5
    ) -> Dict[str, List[Dict]]:
        """Rank articles within each category and select top articles

        Args:
            category_results: Dictionary mapping categories to article lists
            top_n: Number of top articles to select from each category

        Returns:
            Dictionary with selected top articles per category
        """
        ranked_results = {}

        for category, articles in category_results.items():
            if not articles:
                ranked_results[category] = []
                continue

            # Score each article
            scored_articles = []
            for article in articles:
                article_type = self._detect_article_type(article)

                score = self.score_article(article)

                article["score"] = score
                article["article_type"] = article_type
                scored_articles.append(article)

            # Sort by score descending
            sorted_articles = sorted(
                scored_articles,
                key=lambda x: x.get("score", 0),
                reverse=True
            )

            # Select top N
            top_articles = sorted_articles[:top_n]
            ranked_results[category] = top_articles

            logger.info(
                f"Category '{category}': Selected {len(top_articles)} "
                f"from {len(articles)} articles (avg score: {sum(a.get('score', 0) for a in scored_articles) / len(scored_articles):.1f})"
            )

        return ranked_results

    def get_top_articles(
        self,
        category_results: Dict[str, List[Dict]],
        total_articles: int = 20
    ) -> List[Dict]:
        """Get top articles across all categories with diversity

        Args:
            category_results: Dictionary mapping categories to article lists
            total_articles: Total number of articles to select

        Returns:
            List of top articles sorted by score, with category diversity
        """
        # Score all articles from all categories
        all_scored = []

        for category, articles in category_results.items():
            for article in articles:
                if "score" not in article:
                    article_type = self._detect_article_type(article)
                    score = self.score_article(article)
                    article["score"] = score
                    article["article_type"] = article_type

                article["category"] = category
                all_scored.append(article)

        # Sort by score and return top N
        top_articles = sorted(
            all_scored,
            key=lambda x: x.get("score", 0),
            reverse=True
        )[:total_articles]

        # Log category distribution
        category_counts = {}
        for article in top_articles:
            cat = article.get("category", "unknown")
            category_counts[cat] = category_counts.get(cat, 0) + 1

        logger.info(f"Selected {len(top_articles)} top articles. Category distribution: {category_counts}")

        return top_articles
