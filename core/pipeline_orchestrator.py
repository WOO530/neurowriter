"""Pipeline orchestrator for interactive self-evolving introduction generation"""
import json
import logging
import re
from typing import Dict, List, Optional, Tuple

from core.llm_client import LLMClient, get_llm_client
from core.pubmed_client import PubmedClient
from core.intro_generator import IntroductionGenerator
from core.self_evaluator import SelfEvaluator
from core.deep_researcher import DeepResearcher
from prompts.strategy_brief import get_writing_strategy_prompt
from prompts.claim_extraction import get_claim_extraction_prompt, get_supplementary_query_prompt, get_feedback_to_queries_prompt
from prompts.modality_config import detect_modality

logger = logging.getLogger(__name__)

# Maximum self-evolution iterations
MAX_EVOLUTION_ITERATIONS = 4
# Factual accuracy threshold for triggering self-evolution
FACTUAL_ACCURACY_THRESHOLD = 7
# Overall score threshold for triggering self-evolution
OVERALL_SCORE_THRESHOLD = 7.8


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

        Detects modality (eeg/psg/mixed) and stores it in the result
        so downstream stages can use modality-specific prompts.

        Returns:
            Topic analysis dict with search_queries, disease, etc.
        """
        modality = detect_modality(research_topic)
        logger.info(f"Detected modality: {modality}")
        topic_analysis = self.intro_generator.step_parse_topic(research_topic, modality=modality)
        topic_analysis["_detected_modality"] = modality
        return topic_analysis

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
        modality = topic_analysis.get("_detected_modality", "eeg")
        system_prompt, user_prompt = get_writing_strategy_prompt(
            topic_analysis, reference_pool, landscape,
            modality=modality,
        )
        response = self.llm_client.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.5,
            max_tokens=2000,
            reasoning_effort="high",
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
            max_tokens=1500,
            reasoning_effort="medium",
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
        self,
        topic_analysis: Dict,
        reference_pool: List[Dict],
        landscape: Dict,
        writing_strategy: dict = None,
        evaluation_feedback: dict = None,
        unsupported_claims: list = None,
    ) -> str:
        """Generate introduction text"""
        modality = topic_analysis.get("_detected_modality", "eeg")
        text = self.intro_generator.step_generate_introduction(
            topic_analysis, reference_pool, landscape, modality=modality,
            writing_strategy=writing_strategy,
            evaluation_feedback=evaluation_feedback,
            unsupported_claims=unsupported_claims,
        )
        return _normalize_paragraph_breaks(text)

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
        """Check if factual_accuracy or overall score is below threshold"""
        fa_score = evaluation.get("scores", {}).get("factual_accuracy", 10)
        overall = evaluation.get("overall_score", 10)
        return fa_score < FACTUAL_ACCURACY_THRESHOLD or overall < OVERALL_SCORE_THRESHOLD

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
            max_tokens=1500,
            reasoning_effort="medium",
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
            max_tokens=1000,
            reasoning_effort="medium",
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
        landscape: Dict,
        current_introduction: str = "",
    ) -> List[Dict]:
        """Expand reference pool with new papers, preserving cited papers

        If current_introduction is provided, papers already cited ([N]) are
        kept at their original positions so citation numbers stay valid.

        Returns:
            New reference pool after incorporating new papers
        """
        # Identify papers currently cited in the introduction
        cited_indices: set = set()
        if current_introduction:
            for m in re.finditer(r'\[(\d+)\]', current_introduction):
                cited_indices.add(int(m.group(1)))

        # Separate cited papers (preserve) vs non-cited
        cited_papers: List[Dict] = []
        cited_pmids: set = set()
        for idx in sorted(cited_indices):
            if 1 <= idx <= len(current_pool):
                paper = current_pool[idx - 1]
                cited_papers.append(paper)
                pmid = paper.get("pmid")
                if pmid:
                    cited_pmids.add(pmid)

        # Select from new papers (excluding already cited)
        combined = current_pool + new_papers
        reselected = self.deep_researcher.select_reference_pool(combined, landscape)

        # Build final pool: cited first (original order), then new selections
        additional = [
            p for p in reselected
            if p.get("pmid") not in cited_pmids
        ]

        max_pool = 55
        final_pool = cited_papers + additional
        return final_pool[:max_pool]

    @staticmethod
    def validate_citation_range(introduction: str, reference_pool_size: int) -> str:
        """Remove citation numbers that fall outside [1, reference_pool_size]

        Also cleans up empty brackets resulting from removal.

        Args:
            introduction: Introduction text with [N] citations
            reference_pool_size: Number of papers in the reference pool

        Returns:
            Cleaned introduction text
        """
        if not introduction or reference_pool_size <= 0:
            return introduction

        def _replace_bracket(match):
            inner = match.group(1)
            # Handle ranges like "3-5" and lists like "3,4,5"
            parts = re.split(r'[,;]\s*', inner)
            valid_parts = []
            for part in parts:
                part = part.strip()
                range_m = re.match(r'^(\d+)\s*[-–]\s*(\d+)$', part)
                if range_m:
                    lo, hi = int(range_m.group(1)), int(range_m.group(2))
                    if 1 <= lo <= reference_pool_size and 1 <= hi <= reference_pool_size:
                        valid_parts.append(part)
                elif part.isdigit():
                    n = int(part)
                    if 1 <= n <= reference_pool_size:
                        valid_parts.append(part)
            if not valid_parts:
                return ""
            return "[" + ",".join(valid_parts) + "]"

        cleaned = re.sub(r'\[([^\]]+)\]', _replace_bracket, introduction)
        # Remove leftover empty space from removed citations
        cleaned = re.sub(r'[ \t]{2,}', ' ', cleaned)
        return cleaned


def _normalize_paragraph_breaks(text: str) -> str:
    """Normalize paragraph breaks for consistent markdown rendering.

    Handles three cases:
    1. Already has double newlines → clean up intra-paragraph single newlines
    2. Only single newlines → convert to double newlines between paragraphs
    3. No newlines (wall of text) → return as-is to avoid incorrect splitting
    """
    if not text:
        return text
    # Normalize line endings
    text = text.replace('\r\n', '\n')

    # Case 1: Has double newlines — split into paragraphs, clean up
    # single newlines within each paragraph (join wrapped lines)
    if '\n\n' in text:
        paragraphs = text.split('\n\n')
        cleaned = []
        for p in paragraphs:
            stripped = p.strip()
            if stripped:
                # Replace single newlines within paragraph with spaces
                cleaned.append(re.sub(r'\n', ' ', stripped))
        return '\n\n'.join(cleaned)

    # Case 2: Only single newlines — each non-empty line is a paragraph
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if len(lines) >= 3:
        return '\n\n'.join(lines)

    # Case 3: No meaningful line breaks — return as-is
    return text


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
