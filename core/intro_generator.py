"""Introduction generator orchestration"""
import logging
from typing import Dict, List, Optional
from core.llm_client import LLMClient, get_llm_client
from core.pubmed_client import PubmedClient
from core.topic_parser import TopicParser
from core.citation_scorer import CitationScorer
from core.deep_researcher import DeepResearcher
from prompts.intro_generation import get_introduction_generation_prompt
from utils.pubmed_utils import format_citation_vancouver
from utils.cache import PubmedCache

logger = logging.getLogger(__name__)


class IntroductionGenerator:
    """Orchestrate the introduction generation pipeline"""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        pubmed_client: Optional[PubmedClient] = None,
        use_deep_research: bool = True
    ):
        """Initialize introduction generator

        Args:
            llm_client: LLM client instance
            pubmed_client: PubMed client instance
            use_deep_research: Use deep research pipeline (default True)
        """
        self.llm_client = llm_client or get_llm_client()
        self.pubmed_client = pubmed_client or PubmedClient()
        self.topic_parser = TopicParser(self.llm_client)
        self.citation_scorer = CitationScorer()
        self.use_deep_research = use_deep_research
        self.deep_researcher = DeepResearcher(
            llm_client=self.llm_client,
            pubmed_client=self.pubmed_client,
            topic_parser=self.topic_parser
        ) if use_deep_research else None

    def generate_introduction(
        self,
        research_topic: str,
        user_preferences: Optional[str] = None,
        articles_per_category: int = 5,
        total_articles: int = 20
    ) -> Dict:
        """Generate introduction for a research topic

        Uses deep research pipeline (100-200 papers) by default, or basic pipeline if disabled.

        Args:
            research_topic: Research topic from user
            user_preferences: Additional user preferences or emphasis
            articles_per_category: Number of articles per category to select (basic pipeline only)
            total_articles: Total articles to use for generation

        Returns:
            Dictionary with:
            - introduction: Generated introduction text
            - references: List of formatted references
            - articles_used: List of articles used
            - parsing_result: Parsed topic components
            - landscape: Literature landscape analysis (deep research only)
            - paper_pool_size: Total papers collected (deep research only)
        """
        logger.info("Starting introduction generation pipeline")

        if self.use_deep_research:
            return self._generate_with_deep_research(research_topic, user_preferences)
        else:
            return self._generate_basic(research_topic, user_preferences, articles_per_category, total_articles)

    def _generate_with_deep_research(
        self,
        research_topic: str,
        user_preferences: Optional[str] = None
    ) -> Dict:
        """Generate introduction using deep research pipeline

        Args:
            research_topic: Research topic
            user_preferences: User preferences

        Returns:
            Dictionary with introduction and metadata
        """
        from prompts.modality_config import detect_modality

        logger.info("Using deep research pipeline (100-200 papers)")

        modality = detect_modality(research_topic)
        logger.info(f"Detected modality: {modality}")

        # Step 1: Conduct deep research
        logger.info("Step 1/4: Conducting deep hierarchical research...")
        research_results = self.deep_researcher.conduct_deep_research(research_topic)

        topic_analysis = research_results["topic_analysis"]
        landscape = research_results["literature_landscape"]
        reference_pool = research_results["reference_pool"]
        paper_pool = research_results["paper_pool"]

        if not reference_pool:
            logger.warning("No reference pool generated")
            return {
                "introduction": "Unable to generate introduction - no relevant articles found.",
                "references": [],
                "articles_used": [],
                "parsing_result": topic_analysis,
                "landscape": landscape,
                "paper_pool_size": len(paper_pool),
                "error": "No articles in reference pool"
            }

        logger.info(f"Deep research complete: {len(paper_pool)} papers collected, {len(reference_pool)} selected for references")

        # Step 2: Generate introduction with landscape context
        logger.info("Step 2/4: Generating introduction with landscape context...")
        system_prompt, user_prompt = get_introduction_generation_prompt(
            topic_analysis,
            reference_pool,
            landscape=landscape,
            modality=modality
        )

        try:
            introduction = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=2000,
                reasoning_effort="high",
            )

            logger.info("Introduction generated successfully")

            # Format references in Vancouver style
            references = self._format_references(reference_pool)

            return {
                "introduction": introduction,
                "references": references,
                "articles_used": reference_pool,
                "parsing_result": topic_analysis,
                "landscape": landscape,
                "paper_pool_size": len(paper_pool),
                "reference_pool_size": len(reference_pool)
            }

        except Exception as e:
            logger.error(f"Error generating introduction: {e}")
            return {
                "introduction": f"Error generating introduction: {str(e)}",
                "references": [],
                "articles_used": reference_pool,
                "parsing_result": topic_analysis,
                "landscape": landscape,
                "paper_pool_size": len(paper_pool),
                "error": str(e)
            }

    def _generate_basic(
        self,
        research_topic: str,
        user_preferences: Optional[str] = None,
        articles_per_category: int = 5,
        total_articles: int = 20
    ) -> Dict:
        """Generate introduction using basic pipeline (legacy)

        Args:
            research_topic: Research topic
            user_preferences: User preferences
            articles_per_category: Articles per category
            total_articles: Total articles

        Returns:
            Dictionary with introduction and metadata
        """
        logger.info("Using basic research pipeline (legacy)")

        # Step 1: Parse topic
        logger.info("Step 1/4: Parsing research topic...")
        parsed_topic = self.topic_parser.parse_topic(research_topic)
        logger.debug(f"Parsed topic: {parsed_topic}")

        # Step 2: Search and collect articles
        logger.info("Step 2/4: Searching PubMed for relevant articles...")
        search_results = self.pubmed_client.search_by_categories(
            disease=parsed_topic.get("disease", ""),
            data_type=parsed_topic.get("data_type", ""),
            methodology=parsed_topic.get("methodology", ""),
            outcome=parsed_topic.get("outcome", "")
        )

        # Step 3: Score and select articles
        logger.info("Step 3/4: Selecting top-ranked articles...")
        ranked_results = self.citation_scorer.rank_articles_by_category(
            search_results,
            top_n=articles_per_category
        )
        top_articles = self.citation_scorer.get_top_articles(
            ranked_results,
            total_articles=total_articles
        )

        if not top_articles:
            logger.warning("No articles found for generation")
            return {
                "introduction": "Unable to generate introduction - no relevant articles found.",
                "references": [],
                "articles_used": [],
                "parsing_result": parsed_topic,
                "error": "No articles found"
            }

        logger.info(f"Selected {len(top_articles)} articles for introduction")

        # Step 4: Generate introduction
        logger.info("Step 4/4: Generating introduction...")
        system_prompt, user_prompt = get_introduction_generation_prompt(
            parsed_topic,
            top_articles
        )

        try:
            introduction = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=2000,
                reasoning_effort="high",
            )

            logger.info("Introduction generated successfully")

            # Format references in Vancouver style
            references = self._format_references(top_articles)

            return {
                "introduction": introduction,
                "references": references,
                "articles_used": top_articles,
                "parsing_result": parsed_topic,
                "search_results_summary": {
                    category: len(articles) for category, articles in search_results.items()
                }
            }

        except Exception as e:
            logger.error(f"Error generating introduction: {e}")
            return {
                "introduction": f"Error generating introduction: {str(e)}",
                "references": [],
                "articles_used": top_articles,
                "parsing_result": parsed_topic,
                "error": str(e)
            }

    # ------------------------------------------------------------------
    # Discrete step methods for interactive pipeline
    # ------------------------------------------------------------------

    def step_parse_topic(self, research_topic: str, modality: str = "eeg") -> Dict:
        """Step: Parse research topic with deep hierarchical analysis

        Args:
            research_topic: Research topic string
            modality: Detected modality ("eeg", "psg", or "mixed")

        Returns:
            Topic analysis dictionary
        """
        logger.info("Step: Parsing research topic...")
        return self.deep_researcher.topic_parser.parse_topic_deep(research_topic, modality=modality)

    def step_collect_papers(self, topic_analysis: Dict) -> List[Dict]:
        """Step: Collect papers using multi-strategy search

        Args:
            topic_analysis: Topic analysis with search queries

        Returns:
            List of collected papers
        """
        logger.info("Step: Collecting papers...")
        return self.deep_researcher.collect_papers_multistrategy(topic_analysis)

    def step_analyze_landscape(self, paper_pool: List[Dict], research_topic: str) -> Dict:
        """Step: Analyze literature landscape

        Args:
            paper_pool: Collected papers
            research_topic: Original research topic (attached to landscape)

        Returns:
            Landscape analysis dictionary
        """
        logger.info("Step: Analyzing literature landscape...")
        abstracts_by_category = {"all_papers": paper_pool}
        landscape = self.deep_researcher.topic_parser.analyze_literature_landscape(abstracts_by_category)
        landscape["_research_topic"] = research_topic
        return landscape

    def step_select_references(self, paper_pool: List[Dict], landscape: Dict) -> List[Dict]:
        """Step: Select optimal reference pool

        Args:
            paper_pool: All collected papers
            landscape: Landscape analysis

        Returns:
            Selected reference papers
        """
        logger.info("Step: Selecting reference pool...")
        return self.deep_researcher.select_reference_pool(paper_pool, landscape)

    def step_generate_introduction(
        self,
        topic_analysis: Dict,
        reference_pool: List[Dict],
        landscape: Dict,
        modality: str = "eeg",
        writing_strategy: dict = None,
        evaluation_feedback: dict = None,
        unsupported_claims: list = None,
        user_feedback: str = "",
    ) -> str:
        """Step: Generate introduction text

        Args:
            topic_analysis: Topic analysis
            reference_pool: Selected reference papers
            landscape: Landscape analysis
            modality: Detected modality ("eeg", "psg", or "mixed")
            writing_strategy: Optional writing strategy with paragraph outline
            evaluation_feedback: Optional evaluation results from previous iteration
            unsupported_claims: Optional list of unsupported claims from previous iteration
            user_feedback: Optional user-provided feedback for revision

        Returns:
            Generated introduction text
        """
        logger.info("Step: Generating introduction...")
        system_prompt, user_prompt = get_introduction_generation_prompt(
            topic_analysis,
            reference_pool,
            landscape=landscape,
            modality=modality,
            writing_strategy=writing_strategy,
            evaluation_feedback=evaluation_feedback,
            unsupported_claims=unsupported_claims,
            user_feedback=user_feedback,
        )
        return self.llm_client.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=2000,
            reasoning_effort="high",
        )

    def _format_references(self, articles: List[Dict]) -> List[str]:
        """Format articles as Vancouver-style references

        Args:
            articles: List of article metadata

        Returns:
            List of formatted reference strings
        """
        references = []

        for i, article in enumerate(articles, 1):
            citation = format_citation_vancouver(article, i)
            references.append(citation)

        return references

    def generate_with_streaming(
        self,
        research_topic: str,
        user_preferences: Optional[str] = None,
        articles_per_category: int = 5,
        total_articles: int = 20
    ):
        """Generate introduction with streaming output

        Args:
            research_topic: Research topic from user
            user_preferences: Additional user preferences
            articles_per_category: Number of articles per category (basic pipeline only)
            total_articles: Total articles to use (basic pipeline only)

        Yields:
            Tuple of (event_type, content) where event_type is:
            - "parsing": Topic analysis in progress
            - "searching": Literature collection in progress
            - "analyzing": Landscape analysis in progress
            - "selecting": Reference pool selection in progress
            - "generating": Introduction generation in progress
            - "complete": Generation complete with results
        """
        if self.use_deep_research:
            yield from self._generate_with_streaming_deep_research(research_topic)
        else:
            yield from self._generate_with_streaming_basic(research_topic, articles_per_category, total_articles)

    def _generate_with_streaming_deep_research(self, research_topic: str):
        """Generate with streaming using deep research pipeline"""
        from prompts.modality_config import detect_modality

        # Step 0: Detect modality
        modality = detect_modality(research_topic)

        # Step 1: Parse topic deeply
        yield ("parsing", "Conducting deep hierarchical topic analysis...")
        topic_analysis = self.deep_researcher.topic_parser.parse_topic_deep(research_topic, modality=modality)
        topic_analysis["_detected_modality"] = modality
        yield ("parsing", f"Topic parsed: {topic_analysis.get('disease', 'Unknown')} + {topic_analysis.get('key_intervention_or_focus', 'Unknown')} (modality: {modality})")

        # Step 1.5: Review search queries
        yield ("confirmation", "Reviewing generated search strategies...")
        search_queries = topic_analysis.get('search_queries', [])[:20]
        yield ("confirmation", f"Generated {len(search_queries)} search queries for comprehensive literature collection")

        # Step 2: Multi-strategy search
        yield ("searching", "Executing multi-strategy literature search...")
        paper_pool = self.deep_researcher.collect_papers_multistrategy(topic_analysis)
        yield ("searching", f"Collected {len(paper_pool)} papers from {len(search_queries)} search strategies")

        # Step 3: Landscape analysis
        yield ("analyzing", "Analyzing literature landscape...")
        abstracts_by_category = {"all_papers": paper_pool}
        landscape = self.deep_researcher.topic_parser.analyze_literature_landscape(abstracts_by_category)
        landscape["_research_topic"] = research_topic
        yield ("analyzing", f"Identified {len(landscape.get('key_findings', []))} key findings and {len(landscape.get('knowledge_gaps', []))} knowledge gaps")

        # Step 4: Reference pool selection
        yield ("selecting", "Selecting optimal reference pool...")
        reference_pool = self.deep_researcher.select_reference_pool(paper_pool, landscape)
        yield ("selecting", f"Selected {len(reference_pool)} reference papers")

        # Step 5: Generate introduction
        yield ("generating", "Generating introduction with landscape context...")
        system_prompt, user_prompt = get_introduction_generation_prompt(
            topic_analysis,
            reference_pool,
            landscape=landscape,
            modality=modality
        )

        introduction = self.llm_client.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=2000,
            reasoning_effort="high",
        )

        references = self._format_references(reference_pool)

        result = {
            "introduction": introduction,
            "references": references,
            "articles_used": reference_pool,
            "parsing_result": topic_analysis,
            "landscape": landscape,
            "paper_pool_size": len(paper_pool),
            "reference_pool_size": len(reference_pool)
        }

        yield ("complete", result)

    def _generate_with_streaming_basic(
        self,
        research_topic: str,
        articles_per_category: int,
        total_articles: int
    ):
        """Generate with streaming using basic pipeline (legacy)"""
        # Step 1: Parse topic
        yield ("parsing", "Analyzing research topic...")
        parsed_topic = self.topic_parser.parse_topic(research_topic)
        yield ("parsing", f"Topic parsed: {parsed_topic.get('disease', 'Unknown')}")

        # Step 2: Search
        yield ("searching", "Searching PubMed...")
        search_results = self.pubmed_client.search_by_categories(
            disease=parsed_topic.get("disease", ""),
            data_type=parsed_topic.get("data_type", ""),
            methodology=parsed_topic.get("methodology", ""),
            outcome=parsed_topic.get("outcome", "")
        )

        total_found = sum(len(articles) for articles in search_results.values())
        yield ("searching", f"Found {total_found} articles across categories")

        # Step 3: Select articles
        yield ("selecting", "Ranking and selecting top articles...")
        ranked_results = self.citation_scorer.rank_articles_by_category(
            search_results,
            top_n=articles_per_category
        )
        top_articles = self.citation_scorer.get_top_articles(
            ranked_results,
            total_articles=total_articles
        )
        yield ("selecting", f"Selected {len(top_articles)} top articles")

        # Step 4: Generate
        yield ("generating", "Generating introduction with GPT-4o...")
        system_prompt, user_prompt = get_introduction_generation_prompt(
            parsed_topic,
            top_articles
        )

        introduction = self.llm_client.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=2000,
            reasoning_effort="high",
        )

        references = self._format_references(top_articles)

        result = {
            "introduction": introduction,
            "references": references,
            "articles_used": top_articles,
            "parsing_result": parsed_topic
        }

        yield ("complete", result)
