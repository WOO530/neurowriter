"""Pipeline orchestrator for interactive self-evolving introduction generation"""
import json
import logging
from typing import Dict, List, Optional, Tuple

from core.llm_client import LLMClient, get_llm_client
from core.pubmed_client import PubmedClient
from core.intro_generator import IntroductionGenerator
from core.self_evaluator import SelfEvaluator
from core.deep_researcher import DeepResearcher
from prompts.strategy_brief import get_writing_strategy_prompt
from prompts.claim_extraction import get_claim_extraction_prompt, get_supplementary_query_prompt, get_feedback_to_queries_prompt

logger = logging.getLogger(__name__)

# Maximum self-evolution iterations
MAX_EVOLUTION_ITERATIONS = 2
# Factual accuracy threshold for triggering self-evolution
FACTUAL_ACCURACY_THRESHOLD = 7


class PipelineOrchestrator:
    """Orchestrate the interactive self-evolving introduction pipeline

    This class provides discrete methods for each pipeline stage,
    enabling the Streamlit UI to insert user checkpoints between stages.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
    ):
        self.llm_client = get_llm_client(api_key=api_key, model=model)
        self.pubmed_client = PubmedClient()
        self.intro_generator = IntroductionGenerator(
            llm_client=self.llm_client,
            pubmed_client=self.pubmed_client,
            use_deep_research=True
        )
        self.evaluator = SelfEvaluator(self.llm_client)
        self.deep_researcher = self.intro_generator.deep_researcher

    # ------------------------------------------------------------------
    # Checkpoint 1 helpers
    # ------------------------------------------------------------------

    def parse_topic(self, research_topic: str) -> Dict:
        """Parse research topic into structured analysis

        Returns:
            Topic analysis dict with search_queries, disease, etc.
        """
        return self.intro_generator.step_parse_topic(research_topic)

    def update_queries(self, topic_analysis: Dict, edited_queries: List[str]) -> Dict:
        """Replace search queries in topic analysis with user-edited ones

        Args:
            topic_analysis: Current topic analysis
            edited_queries: User-edited list of queries

        Returns:
            Updated topic analysis
        """
        updated = dict(topic_analysis)
        updated["search_queries"] = [q.strip() for q in edited_queries if q.strip()]
        return updated

    # ------------------------------------------------------------------
    # Research phase
    # ------------------------------------------------------------------

    def run_research(
        self, topic_analysis: Dict, research_topic: str
    ) -> Tuple[List[Dict], Dict, List[Dict]]:
        """Run full research pipeline: collect -> analyze -> select

        Returns:
            (paper_pool, landscape, reference_pool)
        """
        paper_pool = self.intro_generator.step_collect_papers(topic_analysis)
        landscape = self.intro_generator.step_analyze_landscape(paper_pool, research_topic)
        reference_pool = self.intro_generator.step_select_references(paper_pool, landscape)
        return paper_pool, landscape, reference_pool

    def generate_writing_strategy(
        self, topic_analysis: Dict, reference_pool: List[Dict], landscape: Dict
    ) -> Dict:
        """Generate paragraph-by-paragraph writing strategy

        Returns:
            Strategy dict with paragraphs, narrative_arc, etc.
        """
        system_prompt, user_prompt = get_writing_strategy_prompt(
            topic_analysis, reference_pool, landscape
        )
        response = self.llm_client.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.5,
            max_tokens=2000
        )
        try:
            clean = _strip_code_fences(response)
            return json.loads(clean)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Could not parse writing strategy as JSON")
            return {"paragraphs": [], "narrative_arc": response[:500], "parse_error": True}

    def run_supplementary_research(
        self,
        additional_queries: List[str],
        paper_pool: List[Dict],
        landscape: Dict,
        research_topic: str,
        topic_analysis: Dict,
    ) -> Tuple[List[Dict], Dict, List[Dict]]:
        """Run supplementary research with additional queries, then re-select

        Args:
            additional_queries: Extra search queries
            paper_pool: Current paper pool
            landscape: Current landscape
            research_topic: Original research topic
            topic_analysis: Topic analysis

        Returns:
            (expanded_paper_pool, updated_landscape, new_reference_pool)
        """
        new_papers = self.deep_researcher.search_supplementary(additional_queries, paper_pool)
        expanded_pool = paper_pool + new_papers
        # Re-analyze landscape with expanded pool
        updated_landscape = self.intro_generator.step_analyze_landscape(expanded_pool, research_topic)
        new_ref_pool = self.intro_generator.step_select_references(expanded_pool, updated_landscape)
        return expanded_pool, updated_landscape, new_ref_pool

    def generate_queries_from_feedback(
        self,
        user_feedback: str,
        writing_strategy: dict,
        topic_analysis: dict,
        landscape: dict
    ) -> dict:
        """Generate PubMed queries from user's natural language feedback

        Args:
            user_feedback: Natural language feedback from user
            writing_strategy: Current writing strategy
            topic_analysis: Topic analysis
            landscape: Literature landscape

        Returns:
            Dict with 'interpretation' and 'queries' list of query strings
        """
        system_prompt, user_prompt = get_feedback_to_queries_prompt(
            user_feedback, writing_strategy, topic_analysis, landscape
        )
        response = self.llm_client.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.4,
            max_tokens=1500
        )
        try:
            clean = _strip_code_fences(response)
            data = json.loads(clean)
            return {
                "interpretation": data.get("interpretation", ""),
                "queries": [q["query"] for q in data.get("queries", []) if "query" in q]
            }
        except (json.JSONDecodeError, TypeError):
            logger.warning("Could not parse feedback-to-queries response")
            return {"interpretation": "", "queries": []}

    def run_fact_check(
        self,
        introduction: str,
        articles_used: list
    ) -> dict:
        """Run comprehensive fact-check on introduction

        Args:
            introduction: Generated introduction text
            articles_used: Articles used in generation

        Returns:
            Fact-check results dict
        """
        from core.fact_checker import FactChecker
        checker = FactChecker(
            llm_client=self.llm_client,
            pubmed_client=self.pubmed_client
        )
        return checker.verify_introduction(introduction, articles_used)

    # ------------------------------------------------------------------
    # Generation phase
    # ------------------------------------------------------------------

    def generate_introduction(
        self, topic_analysis: Dict, reference_pool: List[Dict], landscape: Dict
    ) -> str:
        """Generate introduction text"""
        return self.intro_generator.step_generate_introduction(
            topic_analysis, reference_pool, landscape
        )

    def evaluate_introduction(
        self,
        introduction: str,
        reference_pool: List[Dict],
        topic_analysis: Dict,
        landscape: Dict
    ) -> Dict:
        """Run 8-criterion self-evaluation"""
        return self.evaluator.evaluate_introduction(
            introduction, reference_pool, topic_analysis, landscape
        )

    # ------------------------------------------------------------------
    # Self-evolution phase
    # ------------------------------------------------------------------

    @staticmethod
    def needs_self_evolution(evaluation: Dict) -> bool:
        """Check if factual_accuracy score is below threshold"""
        score = evaluation.get("scores", {}).get("factual_accuracy", 10)
        return score < FACTUAL_ACCURACY_THRESHOLD

    def extract_unsupported_claims(self, evaluation: Dict, introduction: str) -> List[Dict]:
        """Extract unsupported claims from evaluation feedback

        Returns:
            List of claim dicts with 'claim', 'issue', 'needed_evidence'
        """
        system_prompt, user_prompt = get_claim_extraction_prompt(evaluation, introduction)
        response = self.llm_client.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.4,
            max_tokens=1500
        )
        try:
            clean = _strip_code_fences(response)
            data = json.loads(clean)
            return data.get("unsupported_claims", [])
        except (json.JSONDecodeError, TypeError):
            logger.warning("Could not parse claim extraction response")
            return []

    def generate_supplementary_queries(
        self, claims: List[Dict], topic_analysis: Dict
    ) -> List[str]:
        """Generate targeted PubMed queries for unsupported claims

        Returns:
            List of query strings
        """
        system_prompt, user_prompt = get_supplementary_query_prompt(claims, topic_analysis)
        response = self.llm_client.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.4,
            max_tokens=1000
        )
        try:
            clean = _strip_code_fences(response)
            data = json.loads(clean)
            return [q["query"] for q in data.get("queries", []) if "query" in q]
        except (json.JSONDecodeError, TypeError):
            logger.warning("Could not parse supplementary query response")
            return []

    def run_supplementary_search(
        self, queries: List[str], existing_pool: List[Dict]
    ) -> List[Dict]:
        """Run supplementary PubMed searches"""
        return self.deep_researcher.search_supplementary(queries, existing_pool)

    def expand_reference_pool(
        self,
        current_pool: List[Dict],
        new_papers: List[Dict],
        landscape: Dict
    ) -> List[Dict]:
        """Expand reference pool with new papers and re-select

        Returns:
            New reference pool after incorporating new papers
        """
        combined = current_pool + new_papers
        # Use the paper pool (combined) to re-select
        return self.deep_researcher.select_reference_pool(combined, landscape)


def _strip_code_fences(text: str) -> str:
    """Strip markdown code fences from LLM response"""
    if not text:
        return ""
    clean = text.strip()
    if clean.startswith("```"):
        lines = clean.split("\n")
        json_lines = [l for l in lines if not l.startswith("```")]
        clean = "\n".join(json_lines).strip()
    return clean
