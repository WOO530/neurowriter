"""PubMed client for searching and retrieving article metadata"""
import requests
import time
import logging
import threading
from typing import List, Dict, Optional
from utils.pubmed_utils import parse_pubmed_xml, has_valid_abstract
from utils.cache import PubmedCache
import config

logger = logging.getLogger(__name__)


class PubmedClient:
    """Client for interacting with PubMed E-utilities API"""

    def __init__(self, cache: Optional[PubmedCache] = None):
        """Initialize PubMed client

        Args:
            cache: PubmedCache instance. If None, creates new instance.
        """
        self.base_url = config.PUBMED_BASE_URL
        self.batch_size = config.PUBMED_BATCH_SIZE
        self.delay = config.PUBMED_DELAY
        self.cache = cache or PubmedCache()
        self.last_request_time = 0
        self._throttle_lock = threading.Lock()

    def _throttle_request(self):
        """Implement rate limiting for API requests (thread-safe)"""
        with self._throttle_lock:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.delay:
                time.sleep(self.delay - elapsed)
            self.last_request_time = time.time()

    def search_pubmed(self, query: str, category: str, max_results: int = 20) -> List[str]:
        """Search PubMed and return list of PMIDs

        Args:
            query: Search query
            category: Category of search (for caching/identification)
            max_results: Maximum number of results to retrieve

        Returns:
            List of PMIDs
        """
        # Check cache first
        cached_results = self.cache.get_search_results(query, category)
        if cached_results:
            logger.info(f"Using cached results for query: {query[:50]}...")
            return cached_results[:max_results]

        logger.info(f"Searching PubMed: {query[:50]}... (Category: {category})")

        pmids = []
        try:
            self._throttle_request()

            # Use esearch for initial search
            search_url = f"{self.base_url}esearch.fcgi"
            params = {
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "retmode": "json",
            }

            response = requests.get(search_url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            pmids = data.get("esearchresult", {}).get("idlist", [])
            logger.info(f"Found {len(pmids)} articles for category: {category}")

            # Cache the results
            self.cache.cache_search_results(query, category, pmids)

        except Exception as e:
            logger.error(f"Error searching PubMed: {e}")

        return pmids

    def fetch_article_details(self, pmids: List[str]) -> List[Dict]:
        """Fetch detailed metadata for articles using batch efetch

        PubMed efetch supports comma-separated PMIDs. We batch 20 at a time
        to reduce HTTP requests from N to N/20.

        Args:
            pmids: List of PMIDs

        Returns:
            List of article metadata dictionaries
        """
        articles = []
        uncached_pmids = []

        # 1. Check cache first for all PMIDs
        for pmid in pmids:
            cached_article = self.cache.get_article(pmid)
            if cached_article:
                articles.append(cached_article)
            else:
                uncached_pmids.append(pmid)

        if not uncached_pmids:
            return articles

        # 2. Fetch uncached PMIDs in batches of 20
        batch_size = 20
        for i in range(0, len(uncached_pmids), batch_size):
            batch = uncached_pmids[i:i + batch_size]
            try:
                self._throttle_request()

                fetch_url = f"{self.base_url}efetch.fcgi"
                params = {
                    "db": "pubmed",
                    "id": ",".join(batch),
                    "rettype": "xml",
                }

                response = requests.get(fetch_url, params=params, timeout=15)
                response.raise_for_status()

                parsed_articles = parse_pubmed_xml(response.text)

                for article in parsed_articles:
                    self.cache.cache_article(article)
                    if not has_valid_abstract(article):
                        logger.debug(f"Skipping PMID {article.get('pmid', '?')}: no valid abstract")
                        continue
                    articles.append(article)

                logger.info(f"Batch fetched {len(parsed_articles)} articles ({len(batch)} PMIDs)")

            except Exception as e:
                logger.error(f"Error batch-fetching PMIDs {batch[:3]}...: {e}")
                # Fallback: try individually for this batch
                for pmid in batch:
                    try:
                        self._throttle_request()
                        fetch_url = f"{self.base_url}efetch.fcgi"
                        resp = requests.get(fetch_url, params={
                            "db": "pubmed", "id": pmid, "rettype": "xml"
                        }, timeout=10)
                        resp.raise_for_status()
                        parsed = parse_pubmed_xml(resp.text)
                        if parsed:
                            self.cache.cache_article(parsed[0])
                            if has_valid_abstract(parsed[0]):
                                articles.append(parsed[0])
                    except Exception as e2:
                        logger.error(f"Error fetching article {pmid}: {e2}")

        return articles

    def search_and_fetch(
        self, query: str, category: str, max_results: int = 20
    ) -> List[Dict]:
        """Search PubMed and fetch article details in one operation

        Args:
            query: Search query
            category: Category of search
            max_results: Maximum number of results

        Returns:
            List of article metadata dictionaries
        """
        pmids = self.search_pubmed(query, category, max_results)
        return self.fetch_article_details(pmids)

    def get_related_citations(self, pmid: str, max_results: int = 5) -> List[Dict]:
        """Get related citations for an article (using PubMed Central)

        Note: This requires PMC ID. Attempts to fetch from PMC API if available.

        Args:
            pmid: PubMed ID
            max_results: Maximum related articles to retrieve

        Returns:
            List of related article metadata
        """
        logger.info(f"Fetching related citations for PMID: {pmid}")

        related_articles = []
        try:
            self._throttle_request()

            # Get related articles via elink
            link_url = f"{self.base_url}elink.fcgi"
            params = {
                "dbfrom": "pubmed",
                "db": "pubmed",
                "id": pmid,
                "retmax": max_results,
                "rettype": "json",
            }

            response = requests.get(link_url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            related_pmids = (
                data.get("linksets", [{}])[0]
                .get("linksetdbs", [{}])[0]
                .get("links", [])
            )

            # Fetch details for related PMIDs
            if related_pmids:
                related_articles = self.fetch_article_details(related_pmids)

        except Exception as e:
            logger.error(f"Error fetching related citations: {e}")

        return related_articles

    def batch_search(
        self, queries: List[str], category_prefix: str = "batch", max_results: int = 30
    ) -> List[Dict]:
        """Execute multiple search queries with deduplication by PMID

        Args:
            queries: List of search queries to execute sequentially
            category_prefix: Prefix for category naming (category_1, category_2, etc.)
            max_results: Max results per query

        Returns:
            List of unique articles (deduplicated by PMID)
        """
        all_papers = {}  # Use dict to deduplicate by PMID
        logger.info(f"Executing batch search with {len(queries)} queries...")

        for i, query in enumerate(queries, 1):
            try:
                category = f"{category_prefix}_{i}"
                logger.debug(f"  Batch query {i}/{len(queries)}: {query[:60]}...")

                papers = self.search_and_fetch(query, category, max_results=max_results)

                for paper in papers:
                    pmid = paper.get("pmid")
                    if pmid and pmid not in all_papers and has_valid_abstract(paper):
                        all_papers[pmid] = paper

                logger.debug(f"    Found {len(papers)} papers, total unique: {len(all_papers)}")

            except Exception as e:
                logger.warning(f"  Error in batch query {i}: {e}")

        result_list = list(all_papers.values())
        logger.info(f"Batch search complete: {len(result_list)} unique papers from {len(queries)} queries")
        return result_list

    def search_high_impact_journals(
        self, query: str, max_results: int = 40
    ) -> List[Dict]:
        """Search for papers in high-impact journals

        High-impact journals: Nature, Science, NEJM, Lancet, JAMA

        Args:
            query: Search query (will be combined with journal filter)
            max_results: Max results

        Returns:
            List of articles from high-impact journals
        """
        high_impact_query = f'({query}) AND (Nature[Journal] OR Science[Journal] OR "New England Journal of Medicine"[Journal] OR Lancet[Journal] OR JAMA[Journal])'
        logger.info(f"Searching high-impact journals for: {query[:50]}...")

        return self.search_and_fetch(
            high_impact_query, "high_impact_journals", max_results=max_results
        )

    def search_with_date_filter(
        self, query: str, category: str, start_year: int = 2015, max_results: int = 30
    ) -> List[Dict]:
        """Search PubMed with date range filter

        Args:
            query: Search query
            category: Category for caching
            start_year: Minimum publication year
            max_results: Max results

        Returns:
            List of articles from specified year onwards
        """
        date_query = f'({query}) AND ({start_year}[PDAT] : 3000[PDAT])'
        logger.info(f"Searching with date filter (from {start_year}): {query[:50]}...")

        return self.search_and_fetch(date_query, category, max_results=max_results)

    def search_by_categories(
        self,
        disease: str,
        data_type: str,
        methodology: str,
        outcome: Optional[str] = None,
    ) -> Dict[str, List[Dict]]:
        """Perform multi-category search for comprehensive literature collection

        Args:
            disease: Disease/condition name
            data_type: Data type (EEG, PSG, etc.)
            methodology: Methodology (Deep Learning, CNN, etc.)
            outcome: Outcome/prediction target

        Returns:
            Dictionary with search results by category
        """
        results = {}

        # Category 1: Disease epidemiology and burden
        query1 = f'({disease}) AND (prevalence OR burden OR epidemiology OR incidence OR "public health" OR "health burden")'
        results["epidemiology"] = self.search_and_fetch(query1, "epidemiology", max_results=35)

        # Category 2: Current diagnostic/treatment limitations
        query2 = f'({disease}) AND (diagnosis OR diagnostic OR treatment OR "clinical challenge" OR limitation OR unmet OR "treatment resistant" OR "treatment response")'
        results["clinical_challenges"] = self.search_and_fetch(query2, "clinical_challenges", max_results=35)

        # Category 3: EEG/PSG biomarker research
        query3 = f'({disease}) AND ({data_type}) AND (biomarker OR "neural correlate" OR correlation OR association OR abnormality OR "neural signature")'
        results["biomarkers"] = self.search_and_fetch(query3, "biomarkers", max_results=35)

        # Category 4: Deep Learning/ML application
        query4 = f'({disease}) AND ({methodology}) AND ({data_type} OR "EEG" OR "brainwave" OR "neural signal")'
        results["machine_learning"] = self.search_and_fetch(query4, "machine_learning", max_results=35)

        # Category 5: Review articles and meta-analysis
        query5 = f'({disease}) AND ({data_type}) AND ("systematic review" OR "meta-analysis" OR "literature review" OR "scoping review")'
        results["reviews"] = self.search_and_fetch(query5, "reviews", max_results=35)

        # Category 6: Landmark studies - high impact journals priority
        query6 = f'({disease}) AND ({data_type} OR {methodology})'
        results["landmark"] = self.search_and_fetch(query6, "landmark", max_results=25)

        return results
