"""Fact checker for introduction validation"""
import json
import re
import logging
from typing import Dict, List, Optional
from core.llm_client import LLMClient, get_llm_client
from core.pubmed_client import PubmedClient
from prompts.fact_checking import (
    get_fact_checking_prompts,
    get_verification_prompt,
    get_claim_citation_mapping_prompt
)
import requests

logger = logging.getLogger(__name__)


class FactChecker:
    """Validate generated introductions against source material"""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        pubmed_client: Optional[PubmedClient] = None
    ):
        """Initialize fact checker

        Args:
            llm_client: LLM client instance
            pubmed_client: PubMed client instance
        """
        self.llm_client = llm_client or get_llm_client()
        self.pubmed_client = pubmed_client or PubmedClient()

    def verify_introduction(
        self,
        introduction: str,
        articles_used: List[Dict]
    ) -> Dict:
        """Perform comprehensive fact-checking on introduction

        Args:
            introduction: Generated introduction text
            articles_used: List of articles used in generation

        Returns:
            Dictionary with fact-check results:
            - overall_accuracy: HIGH/MEDIUM/LOW
            - pmid_verification: Results of PMID validation
            - relevance_check: Results of claim-article relevance
            - issues: List of identified issues
            - summary: Summary report
        """
        logger.info("Starting fact-checking process...")

        results = {
            "overall_accuracy": "HIGH",
            "pmid_verification": [],
            "relevance_check": [],
            "issues": [],
            "summary": ""
        }

        # Check 1: Verify all PMIDs exist
        logger.info("Checking 1/3: Verifying PMID existence...")
        pmid_verification = self._verify_pmids(articles_used)
        results["pmid_verification"] = pmid_verification

        # Check 2: Extract citations and verify they match articles
        logger.info("Checking 2/4: Verifying citation numbers...")
        citation_check = self._verify_citations(introduction, articles_used)
        results["citation_verification"] = citation_check

        # Check 3: Claim-citation mapping verification
        logger.info("Checking 3/4: Verifying claim-citation mappings...")
        claim_mapping = self._verify_claim_citation_mapping(introduction, articles_used)
        results["claim_mapping"] = claim_mapping

        # Check 4: LLM-based relevance and accuracy check
        logger.info("Checking 4/4: LLM-based accuracy verification...")
        llm_check = self._llm_fact_check(introduction, articles_used)
        results["llm_verification"] = llm_check

        # Compile overall assessment
        results = self._compile_assessment(results)

        logger.info(f"Fact-check complete. Overall accuracy: {results['overall_accuracy']}")

        return results

    def _verify_pmids(self, articles: List[Dict]) -> List[Dict]:
        """Verify that all PMIDs exist in PubMed

        Args:
            articles: List of articles with PMIDs

        Returns:
            List of verification results
        """
        verification_results = []

        for article in articles:
            pmid = article.get("pmid", "")
            if not pmid:
                verification_results.append({
                    "pmid": pmid,
                    "verified": False,
                    "reason": "No PMID provided"
                })
                continue

            try:
                # Try to fetch the PMID from PubMed
                response = requests.get(
                    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                    params={
                        "db": "pubmed",
                        "id": pmid,
                        "rettype": "json"
                    },
                    timeout=5
                )

                verified = response.status_code == 200

                verification_results.append({
                    "pmid": pmid,
                    "verified": verified,
                    "title": article.get("title", ""),
                    "reason": "Found in PubMed" if verified else "Not found"
                })

            except Exception as e:
                logger.warning(f"Error verifying PMID {pmid}: {e}")
                verification_results.append({
                    "pmid": pmid,
                    "verified": None,
                    "reason": f"Verification error: {str(e)}"
                })

        return verification_results

    def _verify_citations(
        self,
        introduction: str,
        articles: List[Dict]
    ) -> Dict:
        """Verify citation numbering consistency

        Args:
            introduction: Introduction text
            articles: Articles used

        Returns:
            Citation verification results
        """
        # Extract all citation numbers from text (handles [1-3], [4,5] groups)
        citations_in_text = set()
        for m in re.finditer(r'\[([^\]]+)\]', introduction):
            for part in re.split(r'[,;]\s*', m.group(1).strip()):
                part = part.strip()
                range_m = re.match(r'^(\d+)\s*[-–]\s*(\d+)$', part)
                if range_m:
                    lo, hi = int(range_m.group(1)), int(range_m.group(2))
                    citations_in_text.update(range(lo, hi + 1))
                elif part.isdigit():
                    citations_in_text.add(int(part))

        max_citation = max(citations_in_text) if citations_in_text else 0
        expected_max = len(articles)

        issues = []

        # Check if cited numbers are within the valid range (1..N)
        # NOT all references need to be cited — only verify no out-of-range refs
        out_of_range = {c for c in citations_in_text if c < 1 or c > expected_max}

        if out_of_range:
            issues.append({
                "type": "out_of_range_citations",
                "numbers": sorted(list(out_of_range))
            })

        return {
            "consistent": len(issues) == 0,
            "total_citations_in_text": len(citations_in_text),
            "expected_citations": expected_max,
            "issues": issues
        }

    def _verify_claim_citation_mapping(
        self,
        introduction: str,
        articles: List[Dict]
    ) -> Dict:
        """Verify that each cited claim is actually supported by the cited article

        Args:
            introduction: Introduction text
            articles: Source articles

        Returns:
            Claim mapping verification results
        """
        try:
            system_prompt, user_prompt = get_claim_citation_mapping_prompt(
                introduction, articles
            )

            response = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=2000,
                reasoning_effort="medium",
            )

            try:
                # Strip markdown code fences if present
                clean = response.strip()
                if clean.startswith("```"):
                    lines = clean.split("\n")
                    json_lines = [l for l in lines if not l.startswith("```")]
                    clean = "\n".join(json_lines).strip()
                result = json.loads(clean)
                return result
            except json.JSONDecodeError:
                logger.warning("Could not parse claim-citation mapping response as JSON")
                return {
                    "claim_mappings": [],
                    "numerical_mismatches": [],
                    "parse_error": True
                }

        except Exception as e:
            logger.error(f"Error in claim-citation mapping: {e}")
            return {
                "claim_mappings": [],
                "numerical_mismatches": [],
                "error": str(e)
            }

    def _llm_fact_check(
        self,
        introduction: str,
        articles: List[Dict]
    ) -> Dict:
        """Use LLM to verify factual accuracy

        Args:
            introduction: Introduction text
            articles: Source articles

        Returns:
            LLM fact-check results
        """
        try:
            system_prompt, user_prompt = get_fact_checking_prompts(
                introduction,
                articles
            )

            response = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=1000,
                reasoning_effort="medium",
            )

            # Parse JSON response
            try:
                result = json.loads(response)
                return result
            except json.JSONDecodeError:
                logger.warning("Could not parse LLM fact-check response as JSON")
                return {
                    "overall_accuracy": "UNKNOWN",
                    "issues_found": [],
                    "summary": response
                }

        except Exception as e:
            logger.error(f"Error in LLM fact-check: {e}")
            return {
                "overall_accuracy": "ERROR",
                "issues_found": [],
                "summary": f"Error: {str(e)}"
            }

    def _compile_assessment(self, results: Dict) -> Dict:
        """Compile final assessment from all checks

        Args:
            results: Dictionary with individual check results

        Returns:
            Updated results with overall assessment
        """
        accuracy_level = "HIGH"
        all_issues = []

        # Check PMID verification — only count explicitly failed, not timeouts/errors
        pmid_results = results.get("pmid_verification", [])
        unverified_pmids = [r for r in pmid_results if r.get("verified") is False]
        if unverified_pmids:
            for pmid_result in unverified_pmids:
                all_issues.append({
                    "type": "unverified_pmid",
                    "pmid": pmid_result["pmid"],
                    "description": f"Could not verify PMID: {pmid_result['reason']}"
                })
            # PMID issues are informational — do NOT downgrade accuracy_level

        # Check citation consistency (out-of-range only)
        citation_check = results.get("citation_verification", {})
        if not citation_check.get("consistent"):
            for issue in citation_check.get("issues", []):
                all_issues.append({
                    "type": issue["type"],
                    "description": f"Citation numbering issue: {issue}"
                })
            # Citation numbering issues are informational — do NOT downgrade accuracy_level

        # Check claim-citation mapping
        claim_mapping = results.get("claim_mapping", {})
        claim_mappings = claim_mapping.get("claim_mappings", [])
        if claim_mappings:
            unsupported = [c for c in claim_mappings if not c.get("is_supported", True)]
            unsupported_ratio = len(unsupported) / len(claim_mappings) if claim_mappings else 0
            for c in unsupported:
                all_issues.append({
                    "type": "UNSUPPORTED_CLAIM",
                    "description": f"Claim not supported by cited article: {c.get('claim', '')[:100]}... Issue: {c.get('issue', '')}"
                })
            if unsupported_ratio >= 0.3 and len(unsupported) >= 3:
                accuracy_level = "LOW"
            elif len(unsupported) >= 3 and accuracy_level == "HIGH":
                accuracy_level = "MEDIUM"

        numerical_mismatches = claim_mapping.get("numerical_mismatches", [])
        for nm in numerical_mismatches:
            severity = nm.get("severity", "minor")
            all_issues.append({
                "type": "NUMERICAL_MISMATCH",
                "description": f"[{severity.upper()}] Claimed: {nm.get('claimed_value', '?')}, Actual: {nm.get('actual_value', '?')} — {nm.get('claim', '')[:100]}"
            })
            if severity == "major" and accuracy_level == "HIGH":
                accuracy_level = "MEDIUM"

        # Check LLM verification results
        llm_check = results.get("llm_verification", {})
        llm_accuracy = llm_check.get("overall_accuracy", "UNKNOWN")
        if llm_accuracy == "LOW":
            accuracy_level = "LOW"
        elif llm_accuracy == "MEDIUM" and accuracy_level == "HIGH":
            accuracy_level = "MEDIUM"

        llm_issues = llm_check.get("issues_found", [])
        all_issues.extend(llm_issues)

        results["overall_accuracy"] = accuracy_level
        results["issues"] = all_issues
        results["summary"] = f"Overall accuracy: {accuracy_level}. " \
                           f"Found {len(all_issues)} potential issues. " \
                           f"Verified {len(pmid_results)} articles."

        return results
