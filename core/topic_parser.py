"""Topic parser for deep hierarchical analysis"""
import json
import logging
from typing import Dict
from core.llm_client import LLMClient, get_llm_client
from prompts.topic_parsing import get_topic_parsing_prompt, get_landscape_analysis_prompt

logger = logging.getLogger(__name__)


class TopicParser:
    """Comprehensive topic analysis with hierarchical decomposition"""

    def __init__(self, llm_client: LLMClient = None):
        """Initialize topic parser

        Args:
            llm_client: LLM client instance. If None, creates OpenAI client.
        """
        self.llm_client = llm_client or get_llm_client()

    def parse_topic_deep(self, research_topic: str) -> Dict:
        """Parse research topic with deep hierarchical analysis

        Returns comprehensive structured analysis including:
        - Concept hierarchy from broad to ultra-specific
        - 15+ targeted search queries
        - Key concepts to research
        - Expected discovery areas

        Args:
            research_topic: The research topic from user

        Returns:
            Dictionary with deep topic analysis
        """
        logger.info(f"Parsing topic deeply: {research_topic[:60]}...")

        system_prompt, user_prompt = get_topic_parsing_prompt(research_topic)

        try:
            response = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.4,  # Lower temp for structured analysis
                max_tokens=3000
            )

            # Guard against empty / None response
            if not response or not response.strip():
                raise ValueError("LLM returned empty response for topic analysis")

            # Strip markdown code fences if present
            clean_response = response.strip()
            if clean_response.startswith("```"):
                lines = clean_response.split("\n")
                json_lines = [l for l in lines if not l.startswith("```")]
                clean_response = "\n".join(json_lines).strip()

            # Parse JSON response
            parsed_data = json.loads(clean_response)

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

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing LLM response as JSON: {e}")
            logger.debug(f"Response was: {response[:500]}...")
            raise ValueError(f"Failed to parse topic analysis: {str(e)}")
        except Exception as e:
            logger.error(f"Error parsing topic: {e}")
            raise

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
                max_tokens=2500
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
