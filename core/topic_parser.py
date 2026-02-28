"""Topic parser for deep hierarchical analysis"""
import json
import logging
import re
from typing import Dict
from core.llm_client import LLMClient, get_llm_client
from prompts.topic_parsing import get_topic_parsing_prompt, get_landscape_analysis_prompt

logger = logging.getLogger(__name__)


def _repair_json(text: str) -> str:
    """Attempt to repair common LLM JSON errors.

    Applies progressively aggressive fixes:
    1. Strip code fences
    2. Remove JS-style comments
    3. Remove trailing commas before ] or }
    4. Remove control characters
    5. Extract first {...} block if surrounded by extra text
    """
    if not text:
        return text

    s = text.strip()

    # 1. Strip code fences
    if s.startswith("```"):
        lines = s.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        s = "\n".join(lines).strip()

    # 2. Remove JS-style comments (// to EOL, /* ... */)
    s = re.sub(r'//[^\n]*', '', s)
    s = re.sub(r'/\*.*?\*/', '', s, flags=re.DOTALL)

    # 3. Remove trailing commas before ] or }
    s = re.sub(r',\s*([}\]])', r'\1', s)

    # 4. Remove control characters except \n and \t
    s = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', s)

    # 5. Extract first {...} block if surrounded by extra text
    match = re.search(r'\{', s)
    if match and match.start() > 0:
        s = s[match.start():]
    if s.startswith('{'):
        depth = 0
        for i, ch in enumerate(s):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
            if depth == 0:
                s = s[:i + 1]
                break

    return s


class TopicParser:
    """Comprehensive topic analysis with hierarchical decomposition"""

    def __init__(self, llm_client: LLMClient = None):
        """Initialize topic parser

        Args:
            llm_client: LLM client instance. If None, creates OpenAI client.
        """
        self.llm_client = llm_client or get_llm_client()

    def parse_topic_deep(self, research_topic: str, modality: str = "eeg") -> Dict:
        """Parse research topic with deep hierarchical analysis

        Returns comprehensive structured analysis including:
        - Concept hierarchy from broad to ultra-specific
        - 15+ targeted search queries
        - Key concepts to research
        - Expected discovery areas

        Args:
            research_topic: The research topic from user
            modality: Detected modality ("eeg", "psg", or "mixed")

        Returns:
            Dictionary with deep topic analysis
        """
        logger.info(f"Parsing topic deeply: {research_topic[:60]}...")

        system_prompt, user_prompt = get_topic_parsing_prompt(research_topic, modality=modality)

        try:
            response = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.4,
                max_tokens=3000,
                reasoning_effort="medium",
            )

            # Guard against empty / None response
            if not response or not response.strip():
                raise ValueError("LLM returned empty response for topic analysis")

            # Repair and parse JSON response
            clean_response = _repair_json(response)
            try:
                parsed_data = json.loads(clean_response)
            except json.JSONDecodeError as e1:
                logger.warning(f"JSON parse failed (attempt 1): {e1}")
                logger.debug(f"Cleaned response was: {clean_response[:500]}...")

                # Retry with more deterministic settings
                try:
                    response2 = self.llm_client.generate(
                        prompt=user_prompt,
                        system_prompt=system_prompt,
                        temperature=0.2,
                        max_tokens=3000,
                        reasoning_effort="high",
                    )
                    if response2 and response2.strip():
                        clean_response2 = _repair_json(response2)
                        parsed_data = json.loads(clean_response2)
                    else:
                        raise json.JSONDecodeError("Empty retry response", "", 0)
                except json.JSONDecodeError as e2:
                    logger.warning(f"JSON parse failed (attempt 2): {e2}")
                    logger.warning("Falling back to minimal topic extraction from raw text")
                    parsed_data = self._extract_minimal_topic(research_topic, response)

            # Sanitize None → defaults (LLM may return "disease": null)
            _str_fields = [
                "disease", "data_type", "methodology", "outcome",
                "key_intervention_or_focus", "additional_context",
            ]
            _list_fields = [
                "disease_subtypes", "concept_hierarchy", "key_concepts",
                "knowledge_areas_to_research", "search_queries",
                "potential_ambiguities",
            ]
            for f in _str_fields:
                if f in parsed_data and parsed_data[f] is None:
                    parsed_data[f] = ""
            for f in _list_fields:
                if f in parsed_data and parsed_data[f] is None:
                    parsed_data[f] = []

            # Validate structure
            required_fields = [
                "disease", "disease_subtypes", "key_intervention_or_focus",
                "data_type", "methodology", "outcome", "concept_hierarchy",
                "key_concepts", "knowledge_areas_to_research", "search_queries"
            ]

            missing_fields = [f for f in required_fields if f not in parsed_data]

            if missing_fields:
                logger.warning(f"Missing fields in parsed topic: {missing_fields}")

            logger.info(f"Topic parsing completed. Found {len(parsed_data.get('search_queries', []))} search queries")
            return parsed_data

        except Exception as e:
            logger.error(f"Error parsing topic: {e}")
            raise

    def _extract_minimal_topic(self, research_topic: str, raw_response: str) -> Dict:
        """Extract a minimal but usable topic analysis from raw LLM text when JSON parsing fails.

        Uses the original research topic and regex to salvage what we can.

        Args:
            research_topic: Original user input
            raw_response: Raw LLM response text

        Returns:
            Minimal topic analysis dict with required fields
        """
        logger.info("Building minimal topic analysis from raw text")

        # Try to extract search queries from raw text (quoted strings or bullet points)
        queries = []
        # Match quoted strings that look like search queries
        quoted = re.findall(r'"([^"]{10,120})"', raw_response or "")
        for q in quoted:
            # Filter out things that look like field values rather than queries
            if any(kw in q.lower() for kw in ["pubmed", "search", "eeg", "psg", "deep learning",
                                                "classify", "detect", "predict", "diagnos"]):
                queries.append(q)
        # Also look for bullet-point style queries
        bullets = re.findall(r'[-•]\s*(.{10,120}?)(?:\n|$)', raw_response or "")
        for b in bullets:
            clean_b = b.strip().strip('"').strip("'")
            if clean_b and clean_b not in queries:
                queries.append(clean_b)

        # Limit to reasonable number
        queries = queries[:20] if queries else [research_topic]

        # Try to extract disease/condition from the topic
        disease = ""
        data_type = ""
        methodology = ""
        for token in ["EEG", "PSG", "polysomnography", "electroencephalogr"]:
            if token.lower() in research_topic.lower():
                data_type = token
                break
        for token in ["deep learning", "machine learning", "CNN", "transformer", "neural network"]:
            if token.lower() in research_topic.lower():
                methodology = token
                break

        return {
            "disease": disease,
            "disease_subtypes": [],
            "key_intervention_or_focus": research_topic,
            "data_type": data_type,
            "methodology": methodology,
            "outcome": "",
            "additional_context": "",
            "concept_hierarchy": [
                {"level": "broad", "concept": research_topic}
            ],
            "key_concepts": [research_topic],
            "knowledge_areas_to_research": [
                "Background and clinical significance",
                "Current methodology and limitations",
                "Recent advances and future directions"
            ],
            "search_queries": queries,
            "_fallback": True,
        }

    def analyze_literature_landscape(self, abstracts_by_category: Dict[str, list]) -> Dict:
        """Analyze collected literature to understand the research landscape

        Args:
            abstracts_by_category: Dict mapping category -> list of article metadata

        Returns:
            Dictionary with landscape analysis (key findings, gaps, trends, etc.)
        """
        total_papers = sum(len(v) for v in abstracts_by_category.values())
        logger.info(f"Analyzing literature landscape from {total_papers} abstracts across {len(abstracts_by_category)} categories...")

        system_prompt, user_prompt = get_landscape_analysis_prompt(abstracts_by_category)

        try:
            response = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.5,
                max_tokens=2500,
                reasoning_effort="high",
            )

            # Handle empty response
            if not response or not response.strip():
                logger.warning("Empty response from LLM for landscape analysis. Returning minimal landscape.")
                return self._get_minimal_landscape(abstracts_by_category)

            # Try to extract JSON from markdown formatting if present
            clean_response = response.strip()
            if clean_response.startswith("```"):
                # Remove markdown code blocks
                lines = clean_response.split("\n")
                # Find JSON content between code blocks
                json_lines = [l for l in lines if not l.startswith("```")]
                clean_response = "\n".join(json_lines).strip()

            # Fix trailing commas before ] or }
            clean_response = re.sub(r',\s*([}\]])', r'\1', clean_response)

            landscape = json.loads(clean_response)

            logger.info("Literature landscape analysis completed")
            logger.debug(f"Found {len(landscape.get('key_findings', []))} key findings, "
                        f"{len(landscape.get('knowledge_gaps', []))} knowledge gaps")

            return landscape

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing landscape analysis: {e}")
            logger.warning(f"Response was: {response[:300] if response else 'EMPTY'}...")
            logger.warning("Returning minimal landscape as fallback")
            return self._get_minimal_landscape(abstracts_by_category)
        except Exception as e:
            logger.error(f"Error analyzing landscape: {e}")
            logger.warning("Returning minimal landscape as fallback")
            return self._get_minimal_landscape(abstracts_by_category)

    def _get_minimal_landscape(self, abstracts_by_category: Dict[str, list]) -> Dict:
        """Return a minimal landscape analysis when full analysis fails

        Args:
            abstracts_by_category: Dict of papers by category

        Returns:
            Minimal but valid landscape dictionary
        """
        total_papers = sum(len(v) for v in abstracts_by_category.values())

        # Extract basic info from papers
        all_papers = []
        for papers in abstracts_by_category.values():
            all_papers.extend(papers)

        # Get top papers by journal tier
        scored_papers = sorted(
            all_papers,
            key=lambda x: self._estimate_tier(x.get("journal", "")),
            reverse=True
        )[:5]

        landmarks = [f"{p.get('pmid', 'N/A')}: {p.get('title', '')[:80]}" for p in scored_papers]

        return {
            "field_overview": f"Analyzed {total_papers} papers from provided abstracts.",
            "key_findings": [
                "Multiple approaches to analyzing neurophysiological data are present in literature",
                "Traditional and machine learning methods both contribute to the field"
            ],
            "landmark_papers": landmarks if landmarks else ["Unable to identify landmark papers"],
            "knowledge_gaps": [
                "Integration of multiple data modalities remains limited",
                "Translation to clinical practice needs improvement"
            ],
            "methodological_trends": [
                "Increased adoption of deep learning approaches",
                "Growing interest in multimodal biomarkers"
            ],
            "controversies_or_debates": [],
            "underexplored_areas": [
                "Clinical translation of biomarker findings",
                "Standardization across studies"
            ],
            "journal_distribution": "Mixed impact journals represented",
            "article_type_distribution": "Mix of original research and reviews",
            "temporal_trend": "Recent papers represent growing field interest",
            "recommendations_for_introduction": "Cover methodological approaches, current applications, and open questions",
            "_fallback": True
        }

    def _estimate_tier(self, journal_name: str) -> int:
        """Estimate journal tier (for fallback landscape generation)

        Args:
            journal_name: Name of journal

        Returns:
            Tier number (higher = better)
        """
        tier1 = ["Nature", "Science", "NEJM", "Lancet", "JAMA"]
        tier2 = ["Psychiatry", "Brain", "Biological", "Molecular"]

        journal_lower = journal_name.lower()

        if any(t.lower() in journal_lower for t in tier1):
            return 3
        elif any(t.lower() in journal_lower for t in tier2):
            return 2
        else:
            return 1

    # Keep backward compatibility
    def parse_topic(self, research_topic: str) -> Dict:
        """Legacy method - calls parse_topic_deep for backward compatibility

        Args:
            research_topic: The research topic from user

        Returns:
            Dictionary with parsed topic components
        """
        try:
            result = self.parse_topic_deep(research_topic)
            # Flatten for legacy compatibility
            return {
                "disease": result.get("disease", ""),
                "data_type": result.get("data_type", ""),
                "methodology": result.get("methodology", ""),
                "outcome": result.get("outcome", ""),
                "additional_context": result.get("key_intervention_or_focus", ""),
                # Keep full analysis
                "full_analysis": result
            }
        except Exception as e:
            logger.error(f"Error in parse_topic: {e}")
            return {
                "disease": "",
                "data_type": "",
                "methodology": "",
                "outcome": "",
                "additional_context": f"Error: {str(e)}"
            }
