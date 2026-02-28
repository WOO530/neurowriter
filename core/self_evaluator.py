"""Self-evaluation module for generated introductions"""
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional
from core.llm_client import LLMClient, get_llm_client
from prompts.self_evaluation import get_evaluation_prompt

logger = logging.getLogger(__name__)


class SelfEvaluator:
    """Evaluate generated introduction against quality criteria"""

    # Evaluation criteria with descriptions
    EVALUATION_CRITERIA = {
        "topic_specificity": {
            "description": "Topic-specific content vs generic content",
            "prompt_hint": "Is this about schizophrenia in general or specifically about TRS + clozapine?"
        },
        "reference_density": {
            "description": "Average citations per sentence",
            "prompt_hint": "Target: 2+ citations per sentence"
        },
        "reference_quality": {
            "description": "Quality and relevance of cited papers",
            "prompt_hint": "Are landmark/high-impact papers cited appropriately?"
        },
        "academic_tone": {
            "description": "Top-tier medical journal writing level",
            "prompt_hint": "Compare to Nature Medicine, NEJM, JAMA Psychiatry standards"
        },
        "logical_flow": {
            "description": "Coherence and transitions between paragraphs",
            "prompt_hint": "Does the narrative build logically?"
        },
        "depth": {
            "description": "Substantive detail and avoiding superficiality",
            "prompt_hint": "Are claims supported with evidence and context?"
        },
        "completeness": {
            "description": "Coverage of key concepts from knowledge gaps",
            "prompt_hint": "Are all important areas addressed?"
        },
        "factual_accuracy": {
            "description": "Accuracy of claims and citations",
            "prompt_hint": "Do cited references actually support the claims?"
        },
        "originality": {
            "description": "Originality of phrasing vs source abstracts (plagiarism risk)",
            "prompt_hint": "Does the text contain phrases copied from abstracts, or genuinely paraphrased?"
        },
        "ai_detectability": {
            "description": "Authenticity of writing style vs generic AI output",
            "prompt_hint": "Does this read like expert academic writing or generic AI text?"
        }
    }

    def __init__(self, llm_client: Optional[LLMClient] = None):
        """Initialize evaluator

        Args:
            llm_client: LLM client for evaluation
        """
        self.llm_client = llm_client or get_llm_client()

    def evaluate_introduction(
        self,
        introduction: str,
        reference_pool: List[Dict],
        topic_analysis: Dict,
        landscape: Dict
    ) -> Dict:
        """Evaluate generated introduction across all criteria

        Args:
            introduction: Generated introduction text
            reference_pool: Available reference papers
            topic_analysis: Topic analysis for context
            landscape: Literature landscape analysis

        Returns:
            Dictionary with:
            - scores: {criterion: 0-10 score}
            - feedback: {criterion: detailed feedback}
            - improvements: List of suggested improvements
            - overall_score: Average score
            - passed: Boolean (all scores >= 7) or False
        """
        logger.info("Evaluating introduction across 10 criteria...")

        evaluation_results = {
            "scores": {},
            "feedback": {},
            "improvements": [],
            "criterion_details": {}
        }

        # Evaluate all criteria in parallel (each is an independent LLM call)
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(
                    self._evaluate_criterion,
                    criterion_key, introduction, reference_pool,
                    topic_analysis, landscape, criterion_info
                ): criterion_key
                for criterion_key, criterion_info in self.EVALUATION_CRITERIA.items()
            }

            for future in as_completed(futures):
                criterion_key = futures[future]
                criterion_info = self.EVALUATION_CRITERIA[criterion_key]
                try:
                    score, feedback, improvement = future.result()

                    evaluation_results["scores"][criterion_key] = score
                    evaluation_results["feedback"][criterion_key] = feedback
                    evaluation_results["criterion_details"][criterion_key] = criterion_info

                    if improvement:
                        evaluation_results["improvements"].append({
                            "criterion": criterion_key,
                            "score": score,
                            "improvement": improvement
                        })

                except Exception as e:
                    logger.error(f"Error evaluating {criterion_key}: {e}")
                    evaluation_results["scores"][criterion_key] = 5  # Default neutral score
                    evaluation_results["feedback"][criterion_key] = f"Error: {str(e)}"

        # Calculate overall score
        scores = list(evaluation_results["scores"].values())
        overall_score = sum(scores) / len(scores) if scores else 0
        evaluation_results["overall_score"] = round(overall_score, 1)

        # Determine if passed (all criteria >= 7)
        evaluation_results["passed"] = all(s >= 7 for s in scores)

        # Sort improvements by severity (lowest scores first)
        evaluation_results["improvements"].sort(key=lambda x: x["score"])

        logger.info(f"Evaluation complete. Overall score: {overall_score}/10. Passed: {evaluation_results['passed']}")

        return evaluation_results

    def _evaluate_criterion(
        self,
        criterion: str,
        introduction: str,
        reference_pool: List[Dict],
        topic_analysis: Dict,
        landscape: Dict,
        criterion_info: Dict
    ) -> tuple:
        """Evaluate a single criterion

        Args:
            criterion: Criterion name
            introduction: Introduction text
            reference_pool: Reference papers
            topic_analysis: Topic analysis
            landscape: Landscape analysis
            criterion_info: Criterion description

        Returns:
            Tuple of (score: 0-10, feedback: str, improvement: str or None)
        """
        system_prompt, user_prompt = get_evaluation_prompt(
            criterion,
            introduction,
            reference_pool,
            topic_analysis,
            landscape,
            criterion_info
        )

        response = self.llm_client.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.2,
            max_tokens=800,
            reasoning_effort="medium",
        )

        try:
            # Strip markdown code fences if present
            clean_response = response.strip()
            if clean_response.startswith("```"):
                lines = clean_response.split("\n")
                json_lines = [l for l in lines if not l.startswith("```")]
                clean_response = "\n".join(json_lines).strip()

            result = json.loads(clean_response)
            score = result.get("score", 5)
            feedback = result.get("feedback", "")
            improvement = result.get("improvement", None) if score < 8 else None

            return score, feedback, improvement

        except json.JSONDecodeError:
            logger.warning(f"Could not parse evaluation response for {criterion}")
            score = 5  # Default
            feedback = response[:200] if response else "Evaluation failed"
            improvement = "Review and improve this criterion"

            return score, feedback, improvement

    def get_improvement_summary(self, evaluation_result: Dict) -> str:
        """Get a summary of needed improvements

        Args:
            evaluation_result: Evaluation result dictionary

        Returns:
            Human-readable summary
        """
        if evaluation_result.get("passed"):
            return "✅ All criteria met (all scores >= 7/10)"

        summary_lines = ["⚠️ Areas for Improvement:\n"]

        for improvement in evaluation_result.get("improvements", []):
            criterion = improvement["criterion"]
            score = improvement["score"]
            text = improvement.get("improvement", "Needs improvement")

            summary_lines.append(f"• {criterion.replace('_', ' ').title()} ({score}/10)\n  {text}\n")

        return "".join(summary_lines)

    def get_score_display(self, evaluation_result: Dict) -> Dict:
        """Get scores formatted for UI display

        Args:
            evaluation_result: Evaluation result

        Returns:
            Dictionary suitable for display (with colors, etc.)
        """
        scores = evaluation_result.get("scores", {})
        display = {}

        for criterion, score in scores.items():
            # Determine color/status
            if score >= 8:
                status = "Excellent"
                color = "green"
            elif score >= 7:
                status = "Good"
                color = "blue"
            elif score >= 5:
                status = "Fair"
                color = "orange"
            else:
                status = "Needs Work"
                color = "red"

            display[criterion] = {
                "score": score,
                "status": status,
                "color": color
            }

        return display
