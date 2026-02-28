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
from core.citation_scorer import CitationScorer
from prompts.strategy_brief import get_writing_strategy_prompt
from prompts.claim_extraction import get_claim_extraction_prompt, get_supplementary_query_prompt, get_feedback_to_queries_prompt, get_completeness_gap_prompt
from prompts.topic_parsing import get_query_regeneration_prompt
from prompts.modality_config import detect_modality

logger = logging.getLogger(__name__)

# Maximum self-evolution iterations
MAX_EVOLUTION_ITERATIONS = 4
# Factual accuracy threshold for triggering self-evolution
FACTUAL_ACCURACY_THRESHOLD = 8
# Overall score threshold for triggering self-evolution
OVERALL_SCORE_THRESHOLD = 8.0
# Completeness score threshold for triggering completeness gap extraction
COMPLETENESS_THRESHOLD = 8


class PipelineOrchestrator:
    """Orchestrate the interactive self-evolving introduction pipeline

    This class provides discrete methods for each pipeline stage,
    enabling the Streamlit UI to insert user checkpoints between stages.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        provider: str = "openai",
        azure_endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        base_model: Optional[str] = None,
    ):
        self.llm_client = get_llm_client(
            provider=provider,
            api_key=api_key,
            model=model,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            base_model=base_model,
        )
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

    def regenerate_queries(
        self, topic_analysis: Dict, resolutions: Dict, existing_queries: List[str]
    ) -> List[str]:
        """Regenerate search queries after disambiguation resolution

        Uses an LLM prompt to update queries based on the user's
        clarified research intent.

        Args:
            topic_analysis: Parsed topic analysis
            resolutions: Resolved ambiguities {aspect: {"choice": str, "note": str}}
            existing_queries: Current list of search queries

        Returns:
            Updated list of search query strings (falls back to existing on failure)
        """
        system_prompt, user_prompt = get_query_regeneration_prompt(
            topic_analysis, resolutions, existing_queries
        )
        try:
            response = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.4,
                max_tokens=2000,
                reasoning_effort="medium",
            )
            clean = _strip_code_fences(response)
            data = json.loads(clean)
            new_queries = data.get("search_queries", [])
            if new_queries and isinstance(new_queries, list):
                return [q.strip() for q in new_queries if isinstance(q, str) and q.strip()]
        except (json.JSONDecodeError, TypeError, Exception) as e:
            logger.warning(f"Query regeneration failed, keeping existing queries: {e}")
        return existing_queries

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
        user_feedback: str = "",
        current_introduction: str = "",
    ) -> str:
        """Generate introduction text"""
        modality = topic_analysis.get("_detected_modality", "eeg")
        text = self.intro_generator.step_generate_introduction(
            topic_analysis, reference_pool, landscape, modality=modality,
            writing_strategy=writing_strategy,
            evaluation_feedback=evaluation_feedback,
            unsupported_claims=unsupported_claims,
            user_feedback=user_feedback,
            current_introduction=current_introduction,
        )
        return _normalize_paragraph_breaks(text)

    def evaluate_introduction(
        self,
        introduction: str,
        reference_pool: List[Dict],
        topic_analysis: Dict,
        landscape: Dict
    ) -> Dict:
        """Run 10-criterion self-evaluation"""
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

    def extract_completeness_gaps(
        self, evaluation: Dict, introduction: str, landscape: Dict
    ) -> List[Dict]:
        """Extract completeness gaps by comparing introduction to landscape

        Only runs when completeness score is below COMPLETENESS_THRESHOLD.

        Args:
            evaluation: Self-evaluation results
            introduction: Current introduction text
            landscape: Literature landscape analysis

        Returns:
            List of gap dicts with 'source', 'item_number', 'item_text',
            'gap_description', 'needed_evidence'
        """
        comp_score = evaluation.get("scores", {}).get("completeness", 10)
        if comp_score >= COMPLETENESS_THRESHOLD:
            return []

        system_prompt, user_prompt = get_completeness_gap_prompt(
            landscape, introduction, evaluation
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
            return data.get("completeness_gaps", [])
        except (json.JSONDecodeError, TypeError):
            logger.warning("Could not parse completeness gap extraction response")
            return []

    @staticmethod
    def _completeness_gaps_as_claims(gaps: List[Dict]) -> List[Dict]:
        """Convert completeness gaps to claim format for supplementary query generation

        Transforms gap dicts into the same format as unsupported claims
        so they can be fed into generate_supplementary_queries().

        Args:
            gaps: List of completeness gap dicts

        Returns:
            List of claim-format dicts with 'claim', 'issue', 'needed_evidence'
        """
        claims = []
        for gap in gaps:
            claims.append({
                "claim": f"[MISSING CONTENT] {gap.get('item_text', '')}",
                "issue": gap.get("gap_description", "Missing from introduction"),
                "needed_evidence": gap.get("needed_evidence", "Evidence to cover this topic"),
            })
        return claims

    def generate_supplementary_queries(
        self, claims: List[Dict], topic_analysis: Dict
    ) -> List[Dict]:
        """Generate targeted PubMed queries for unsupported claims

        Returns:
            List of dicts with 'query' and 'rationale' keys
        """
        system_prompt, user_prompt = get_supplementary_query_prompt(claims, topic_analysis)

        for attempt in range(2):
            response = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.4,
                max_tokens=1500,
                reasoning_effort="high" if attempt > 0 else "medium",
            )
            try:
                clean = _strip_code_fences(response)
                data = json.loads(clean)
                result = [
                    {"query": q["query"], "rationale": q.get("rationale", "")}
                    for q in data.get("queries", [])
                    if "query" in q
                ]
                if result:
                    return result
            except (json.JSONDecodeError, TypeError):
                # Fallback: try to extract queries from text with regex
                queries = _extract_queries_from_text(response)
                if queries:
                    logger.info(f"Used fallback query extraction: {len(queries)} queries")
                    return queries
                logger.warning(
                    f"Could not parse supplementary query response (attempt {attempt + 1}/2)"
                )

        return []

    @staticmethod
    def _extract_query_strings(queries: List) -> List[str]:
        """Extract plain query strings from query list (handles both str and dict formats)"""
        result = []
        for q in queries:
            if isinstance(q, dict):
                result.append(q.get("query", ""))
            else:
                result.append(str(q))
        return [s for s in result if s]

    def run_supplementary_search(
        self, queries: List, existing_pool: List[Dict]
    ) -> List[Dict]:
        """Run supplementary PubMed searches

        Args:
            queries: List of query strings or dicts with 'query' key
            existing_pool: Current paper pool to deduplicate against
        """
        query_strings = self._extract_query_strings(queries)
        return self.deep_researcher.search_supplementary(query_strings, existing_pool)

    def expand_reference_pool(
        self,
        current_pool: List[Dict],
        new_papers: List[Dict],
        landscape: Dict,
        current_introduction: str = "",
    ) -> List[Dict]:
        """Expand reference pool with new papers using stable merge strategy

        Three-tier approach for deterministic results:
        1. LOCK: Papers cited in current introduction — never removed
        2. KEEP: Uncited papers already in pool — retained
        3. ADD: New papers that score higher than the weakest uncited paper — swap in

        This avoids calling select_reference_pool() which causes
        non-deterministic re-selection each iteration.

        Returns:
            Expanded reference pool
        """
        # Identify papers currently cited in the introduction
        cited_indices: set = set()
        if current_introduction:
            for m in re.finditer(r'\[([^\]]+)\]', current_introduction):
                for n in _expand_citation_group(m.group(1)):
                    cited_indices.add(n)

        # LOCK: Cited papers (preserve at original positions)
        cited_papers: List[Dict] = []
        cited_pmids: set = set()
        uncited_papers: List[Dict] = []
        pool_pmids: set = set()

        for idx, paper in enumerate(current_pool, 1):
            pmid = paper.get("pmid")
            if pmid:
                pool_pmids.add(pmid)
            if idx in cited_indices:
                cited_papers.append(paper)
                if pmid:
                    cited_pmids.add(pmid)
            else:
                uncited_papers.append(paper)

        # Score uncited pool papers and new papers for comparison
        scorer = CitationScorer()
        for paper in uncited_papers:
            if "relevance_score" not in paper:
                paper["relevance_score"] = scorer.score_article(paper)

        # Filter new papers: exclude duplicates already in pool
        candidates = []
        for paper in new_papers:
            pmid = paper.get("pmid")
            if pmid and pmid not in pool_pmids:
                if "relevance_score" not in paper:
                    paper["relevance_score"] = scorer.score_article(paper)
                candidates.append(paper)

        # ADD: Swap in new papers that score >= 90% of weakest uncited (relaxed threshold)
        # Also allow net growth up to max_pool
        if uncited_papers and candidates:
            # Sort uncited by score ascending (weakest first)
            uncited_papers.sort(key=lambda p: p.get("relevance_score", 0))
            min_score = uncited_papers[0].get("relevance_score", 0)
            # Relaxed threshold: new paper needs only 90% of weakest score to swap
            swap_threshold = min_score * 0.9

            # Sort candidates by score descending (strongest first)
            candidates.sort(key=lambda p: p.get("relevance_score", 0), reverse=True)

            swapped = 0
            for candidate in candidates:
                if candidate.get("relevance_score", 0) > swap_threshold and uncited_papers:
                    # Replace weakest uncited paper
                    uncited_papers.pop(0)
                    uncited_papers.append(candidate)
                    swapped += 1
                    # Re-sort after insertion
                    uncited_papers.sort(key=lambda p: p.get("relevance_score", 0))
                    min_score = uncited_papers[0].get("relevance_score", 0)
                    swap_threshold = min_score * 0.9
                else:
                    break  # No more candidates beat the threshold

            # Also allow adding top remaining candidates beyond swaps (net growth)
            remaining = [c for c in candidates if c not in uncited_papers]
            remaining.sort(key=lambda p: p.get("relevance_score", 0), reverse=True)
            uncited_papers.extend(remaining[:5])  # Add up to 5 extra papers

            if swapped > 0 or remaining:
                logger.info(f"Reference pool: swapped {swapped}, added {min(len(remaining), 5)} new papers")
        elif not uncited_papers:
            # Pool was entirely cited — just append candidates
            uncited_papers = candidates

        max_pool = 65
        final_pool = cited_papers + uncited_papers
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

    @staticmethod
    def renumber_citations(introduction: str, reference_pool: list) -> tuple:
        """Renumber citations so they appear sequentially (1, 2, 3, ...) in order of first appearance.

        Also reorders reference_pool to match the new numbering.

        Args:
            introduction: Introduction text with [N] citations
            reference_pool: List of paper dicts

        Returns:
            (renumbered_text, reordered_pool)
        """
        if not introduction or not reference_pool:
            return introduction, reference_pool

        # Pass 1: scan left-to-right, build old_to_new mapping
        old_to_new = {}
        counter = 1
        for m in re.finditer(r'\[([^\]]+)\]', introduction):
            nums = _expand_citation_group(m.group(1))
            for n in nums:
                if n not in old_to_new and 1 <= n <= len(reference_pool):
                    old_to_new[n] = counter
                    counter += 1

        # Early return if already sequential
        if all(k == v for k, v in old_to_new.items()):
            return introduction, reference_pool

        # Pass 2: replace all bracket groups
        def _replace(match):
            nums = _expand_citation_group(match.group(1))
            mapped = [old_to_new[n] for n in nums if n in old_to_new]
            if not mapped:
                return match.group(0)
            return "[" + _compact_citation_list(mapped) + "]"

        renumbered = re.sub(r'\[([^\]]+)\]', _replace, introduction)

        # Reorder pool: cited papers in appearance order, then uncited
        cited_indices = sorted(old_to_new.keys(), key=lambda k: old_to_new[k])
        cited_set = set(cited_indices)
        reordered = [reference_pool[i - 1] for i in cited_indices]
        uncited = [reference_pool[i] for i in range(len(reference_pool)) if (i + 1) not in cited_set]
        reordered.extend(uncited)

        return renumbered, reordered


def _extract_queries_from_text(text: str) -> List[Dict]:
    """Fallback: extract PubMed-style queries from unstructured LLM text

    Looks for quoted strings or lines that look like PubMed queries.
    """
    if not text:
        return []
    queries = []
    # Try quoted strings first
    for m in re.finditer(r'"([^"]{10,120})"', text):
        candidate = m.group(1).strip()
        # Filter out non-query strings (must contain medical/scientific terms)
        if any(c in candidate.lower() for c in ["and", "or", "[", "eeg", "sleep", "neural",
                                                   "brain", "deep learning", "machine learning",
                                                   "clinical", "disorder", "disease", "treatment"]):
            queries.append({"query": candidate, "rationale": "extracted from text"})
    # If no quoted strings, try lines with PubMed-style syntax
    if not queries:
        for line in text.split("\n"):
            line = line.strip().lstrip("0123456789.-) ")
            if len(line) > 15 and ("AND" in line or "OR" in line or "[" in line):
                queries.append({"query": line[:150], "rationale": "extracted from text"})
    return queries[:8]


def _expand_citation_group(inner: str) -> list:
    """Parse bracket contents like '3,5-7' into [3, 5, 6, 7]"""
    nums = []
    for part in re.split(r'[,;]\s*', inner.strip()):
        part = part.strip()
        range_m = re.match(r'^(\d+)\s*[-–]\s*(\d+)$', part)
        if range_m:
            lo, hi = int(range_m.group(1)), int(range_m.group(2))
            nums.extend(range(lo, hi + 1))
        elif part.isdigit():
            nums.append(int(part))
    return nums


def _compact_citation_list(nums: list) -> str:
    """Format [1,2,3,5] as '1-3,5'. Consecutive 2 → comma, 3+ → range."""
    if not nums:
        return ""
    sorted_nums = sorted(set(nums))
    groups = []
    start = prev = sorted_nums[0]
    for n in sorted_nums[1:]:
        if n == prev + 1:
            prev = n
        else:
            groups.append((start, prev))
            start = prev = n
    groups.append((start, prev))

    parts = []
    for lo, hi in groups:
        if lo == hi:
            parts.append(str(lo))
        elif hi - lo == 1:
            parts.append(f"{lo},{hi}")
        else:
            parts.append(f"{lo}-{hi}")
    return ",".join(parts)


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
