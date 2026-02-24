"""Deep research pipeline for comprehensive literature analysis"""
import logging
import re
from typing import Dict, List, Optional
from core.pubmed_client import PubmedClient
from core.topic_parser import TopicParser
from core.citation_scorer import CitationScorer
from core.llm_client import LLMClient, get_llm_client
from utils.pubmed_utils import has_valid_abstract

logger = logging.getLogger(__name__)

# Keywords for automatic paper categorisation
_CATEGORY_PATTERNS: Dict[str, List[str]] = {
    "epidemiology": [
        "prevalence", "incidence", "epidemiolog", "burden", "mortality",
        "morbidity", "population", "cohort study", "cross-sectional",
    ],
    "clinical_treatment": [
        "treatment", "therapy", "pharmacolog", "clinical trial",
        "randomized", "intervention", "antidepressant", "antipsychotic",
        "clozapine", "medication", "drug", "response", "remission",
    ],
    "biomarkers": [
        "biomarker", "neural correlate", "eeg", "erp", "spectral",
        "oscillat", "connectivity", "neurophysiolog", "electrophysiolog",
        "polysomnograph", "psg", "sleep stage",
    ],
    "methodology_ml": [
        "deep learning", "machine learning", "neural network", "cnn",
        "convolutional", "recurrent", "lstm", "transformer", "classification",
        "prediction model", "feature extraction", "automated",
    ],
    "reviews": [
        "systematic review", "meta-analysis", "meta-analytic", "scoping review",
        "literature review", "narrative review", "overview",
    ],
}


class DeepResearcher:
    """Conduct extensive literature research before introduction writing"""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        pubmed_client: Optional[PubmedClient] = None,
        topic_parser: Optional[TopicParser] = None
    ):
        """Initialize deep researcher

        Args:
            llm_client: LLM client for analysis
            pubmed_client: PubMed client for searches
            topic_parser: Topic parser for initial analysis
        """
        self.llm_client = llm_client or get_llm_client()
        self.pubmed_client = pubmed_client or PubmedClient()
        self.topic_parser = topic_parser or TopicParser(self.llm_client)
        self.scorer = CitationScorer()

    def conduct_deep_research(self, research_topic: str) -> Dict:
        """Execute comprehensive research pipeline

        Args:
            research_topic: Research topic from user

        Returns:
            Dictionary with:
            - topic_analysis: Hierarchical topic breakdown
            - literature_landscape: Analyzed field overview
            - paper_pool: 100-200 collected papers
            - reference_pool: 30-50 selected reference papers
        """
        logger.info("=== Starting Deep Research Pipeline ===")

        # Step 1: Hierarchical topic analysis
        logger.info("Step 1/4: Conducting hierarchical topic analysis...")
        topic_analysis = self.topic_parser.parse_topic_deep(research_topic)

        # Step 2: Multi-strategy literature collection
        logger.info("Step 2/4: Collecting literature from multiple search strategies...")
        paper_pool = self.collect_papers_multistrategy(topic_analysis)

        # Step 3: Analyze literature landscape
        logger.info("Step 3/4: Analyzing literature landscape...")
        landscape = self.analyze_landscape(paper_pool, topic_analysis)
        # Attach research_topic so downstream stages can use it
        landscape["_research_topic"] = research_topic

        # Step 4: Select reference pool
        logger.info("Step 4/4: Selecting optimal reference pool...")
        reference_pool = self.select_reference_pool(paper_pool, landscape)

        logger.info(f"=== Deep Research Complete ===")
        logger.info(f"Paper pool: {len(paper_pool)} papers")
        logger.info(f"Reference pool: {len(reference_pool)} papers")

        return {
            "topic_analysis": topic_analysis,
            "literature_landscape": landscape,
            "paper_pool": paper_pool,
            "reference_pool": reference_pool
        }

    def collect_papers_multistrategy(self, topic_analysis: Dict) -> List[Dict]:
        """Collect papers using multiple search strategies

        Papers without a valid abstract (>=100 chars) are excluded.

        Args:
            topic_analysis: Topic analysis with search queries

        Returns:
            Deduplicated list of collected papers
        """
        all_papers = {}  # Use dict to deduplicate by PMID
        search_queries = topic_analysis.get("search_queries", [])

        logger.info(f"Executing {len(search_queries)} search queries...")

        for i, query in enumerate(search_queries, 1):
            try:
                logger.info(f"  Query {i}/{len(search_queries)}: {query[:50]}...")

                # Search strategy 1: General search
                papers = self.pubmed_client.search_and_fetch(query, f"query_{i}", max_results=30)

                for paper in papers:
                    pmid = paper.get("pmid")
                    if pmid and pmid not in all_papers and has_valid_abstract(paper):
                        if paper.get("is_retracted", False):
                            logger.warning(f"    Skipping retracted paper: PMID {pmid}")
                            continue
                        all_papers[pmid] = paper

                logger.debug(f"    Found {len(papers)} papers, total unique: {len(all_papers)}")

            except Exception as e:
                logger.warning(f"  Error in query {i}: {e}")

        # Strategy: High-impact journal filter
        try:
            logger.info("  Adding high-impact journal search...")
            high_impact = self.pubmed_client.search_and_fetch(
                f"({topic_analysis.get('disease', '')}) AND (Nature[Journal] OR NEJM[Journal] OR Lancet[Journal] OR JAMA[Journal])",
                "high_impact",
                max_results=40
            )
            for paper in high_impact:
                pmid = paper.get("pmid")
                if pmid and pmid not in all_papers and has_valid_abstract(paper):
                    if paper.get("is_retracted", False):
                        logger.warning(f"    Skipping retracted paper: PMID {pmid}")
                        continue
                    all_papers[pmid] = paper
        except Exception as e:
            logger.warning(f"Error in high-impact search: {e}")

        paper_list = list(all_papers.values())
        logger.info(f"Total unique papers collected: {len(paper_list)}")

        return paper_list

    def analyze_landscape(self, paper_pool: List[Dict], topic_analysis: Dict) -> Dict:
        """Analyze literature landscape

        Args:
            paper_pool: All collected papers
            topic_analysis: Topic analysis for context

        Returns:
            Landscape analysis dictionary
        """
        # Pass entire pool — the prompt template already limits per-category display
        abstracts_by_category = {
            "all_papers": paper_pool
        }

        try:
            landscape = self.topic_parser.analyze_literature_landscape(abstracts_by_category)
            return landscape
        except Exception as e:
            logger.error(f"Error analyzing landscape: {e}")
            # Return basic stats if analysis fails
            return {
                "field_overview": f"Analyzed {len(paper_pool)} papers",
                "key_findings": [],
                "knowledge_gaps": [],
                "landmark_papers": [],
                "error": str(e)
            }

    # ------------------------------------------------------------------
    # Paper categorisation
    # ------------------------------------------------------------------

    @staticmethod
    def _categorize_papers(papers: List[Dict]) -> Dict[str, List[Dict]]:
        """Categorise papers into predefined groups using keyword matching

        Categories: epidemiology, clinical_treatment, biomarkers,
                    methodology_ml, reviews, other

        Args:
            papers: List of article dicts

        Returns:
            Dict mapping category name -> list of papers
        """
        categorised: Dict[str, List[Dict]] = {cat: [] for cat in _CATEGORY_PATTERNS}
        categorised["other"] = []

        for paper in papers:
            text = (
                (paper.get("title", "") + " " + paper.get("abstract", ""))
                .lower()
            )
            matched = False
            for cat, keywords in _CATEGORY_PATTERNS.items():
                if any(kw in text for kw in keywords):
                    categorised[cat].append(paper)
                    matched = True
                    break  # primary category only
            if not matched:
                categorised["other"].append(paper)

        # Log distribution
        dist = {cat: len(ps) for cat, ps in categorised.items() if ps}
        logger.info(f"Paper categorisation: {dist}")
        return categorised

    # ------------------------------------------------------------------
    # Reference pool selection with diversity
    # ------------------------------------------------------------------

    def select_reference_pool(self, paper_pool: List[Dict], landscape: Dict) -> List[Dict]:
        """Select optimal reference pool with diversity guarantees

        Strategy:
        1. Compute relevance score for each paper
        2. Score all papers
        3. Filter out abstract < 200 chars and score < median * 0.7
        4. Categorise papers
        5. Round-robin pick min 3 per category
        6. Ensure min 3 reviews/meta-analyses
        7. Fill remaining slots by global score

        Args:
            paper_pool: All collected papers
            landscape: Landscape analysis results

        Returns:
            Selected reference papers (30-50 papers)
        """
        research_topic = landscape.get("_research_topic", "")
        topic_keywords = CitationScorer._extract_keywords(research_topic)

        # --- Score all papers ---
        for paper in paper_pool:
            rel = CitationScorer.compute_relevance_score(paper, research_topic, topic_keywords)
            paper["relevance_score"] = rel
            paper["score"] = self.scorer.score_article(paper, relevance_score=rel)
            if "article_type" not in paper:
                paper["article_type"] = self.scorer._detect_article_type(paper)

        # --- Filter: abstract >= 200 chars ---
        candidates = [p for p in paper_pool if len(p.get("abstract", "")) >= 200]

        if not candidates:
            candidates = paper_pool  # fallback: don't filter if nothing survives

        # --- Filter: score >= median * 0.7 ---
        if len(candidates) > 10:
            scores_sorted = sorted(p["score"] for p in candidates)
            median_score = scores_sorted[len(scores_sorted) // 2]
            threshold = median_score * 0.7
            filtered = [p for p in candidates if p["score"] >= threshold]
            if len(filtered) >= 20:
                candidates = filtered

        # --- Target count ---
        target_count = min(50, max(30, len(paper_pool) // 4))

        # --- Categorise ---
        categorised = self._categorize_papers(candidates)

        selected_pmids = set()
        selected = []

        def _add(paper):
            pmid = paper.get("pmid")
            if pmid and pmid not in selected_pmids:
                selected_pmids.add(pmid)
                selected.append(paper)
                return True
            return False

        # --- Round-robin: min 3 per non-empty category ---
        min_per_cat = 3
        for cat, cat_papers in categorised.items():
            sorted_cat = sorted(cat_papers, key=lambda x: x.get("score", 0), reverse=True)
            count = 0
            for p in sorted_cat:
                if count >= min_per_cat:
                    break
                if _add(p):
                    count += 1

        # --- Ensure min 3 reviews/meta-analyses ---
        review_count = sum(
            1 for p in selected
            if p.get("article_type") in ("review", "meta-analysis")
        )
        if review_count < 3:
            review_candidates = sorted(
                [p for p in candidates if p.get("article_type") in ("review", "meta-analysis")],
                key=lambda x: x.get("score", 0),
                reverse=True,
            )
            for p in review_candidates:
                if review_count >= 3:
                    break
                if _add(p):
                    review_count += 1

        # --- Fill remaining by global score ---
        remaining = target_count - len(selected)
        if remaining > 0:
            global_sorted = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)
            for p in global_sorted:
                if remaining <= 0:
                    break
                if _add(p):
                    remaining -= 1

        # Sort final pool by score descending
        selected.sort(key=lambda x: x.get("score", 0), reverse=True)

        # Log category distribution
        cat_dist: Dict[str, int] = {}
        for p in selected:
            for cat, cat_papers in categorised.items():
                if any(pp.get("pmid") == p.get("pmid") for pp in cat_papers):
                    cat_dist[cat] = cat_dist.get(cat, 0) + 1
                    break

        logger.info(f"Selected {len(selected)} reference papers from {len(paper_pool)} total")
        logger.info(f"Reference pool category distribution: {cat_dist}")

        return selected

    # ------------------------------------------------------------------
    # Supplementary search for self-evolving pipeline
    # ------------------------------------------------------------------

    def search_supplementary(
        self,
        queries: List[str],
        existing_pool: List[Dict],
        max_per_query: int = 20
    ) -> List[Dict]:
        """Run targeted PubMed searches for unsupported claims

        Args:
            queries: Targeted search queries for unsupported claims
            existing_pool: Current paper pool (for dedup)
            max_per_query: Max results per query

        Returns:
            New papers not already in existing_pool
        """
        existing_pmids = {p.get("pmid") for p in existing_pool if p.get("pmid")}
        new_papers: Dict[str, Dict] = {}

        for i, query in enumerate(queries, 1):
            try:
                logger.info(f"  Supplementary query {i}/{len(queries)}: {query[:60]}...")
                papers = self.pubmed_client.search_and_fetch(
                    query, f"supplementary_{i}", max_results=max_per_query
                )
                for paper in papers:
                    pmid = paper.get("pmid")
                    if pmid and pmid not in existing_pmids and pmid not in new_papers and has_valid_abstract(paper):
                        if paper.get("is_retracted", False):
                            logger.warning(f"    Skipping retracted paper: PMID {pmid}")
                            continue
                        new_papers[pmid] = paper
            except Exception as e:
                logger.warning(f"  Error in supplementary query {i}: {e}")

        result = list(new_papers.values())
        logger.info(f"Supplementary search found {len(result)} new papers from {len(queries)} queries")
        return result

    # ------------------------------------------------------------------
    # Backward compatibility aliases
    # ------------------------------------------------------------------
    _collect_papers_multistrategy = collect_papers_multistrategy
    _analyze_landscape = analyze_landscape
    _select_reference_pool = select_reference_pool
