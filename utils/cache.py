"""SQLite cache for PubMed metadata"""
import sqlite3
from typing import Optional, List, Dict
import json
import logging
from datetime import datetime
import config

logger = logging.getLogger(__name__)


class PubmedCache:
    """SQLite cache for PubMed article metadata and search results"""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize cache

        Args:
            db_path: Path to SQLite database. If None, uses config.DB_PATH
        """
        self.db_path = db_path or config.DB_PATH
        self._init_db()

    def _init_db(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Articles table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS articles (
                pmid TEXT PRIMARY KEY,
                title TEXT,
                abstract TEXT,
                journal TEXT,
                journal_iso TEXT,
                pub_year TEXT,
                authors TEXT,
                doi TEXT,
                pmc_id TEXT,
                citation_count INTEGER,
                cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # Search results cache
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS search_results (
                search_id TEXT PRIMARY KEY,
                query TEXT,
                category TEXT,
                pmids TEXT,
                cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        conn.commit()
        conn.close()

    def get_article(self, pmid: str) -> Optional[Dict]:
        """Get article from cache

        Args:
            pmid: PubMed ID

        Returns:
            Article metadata dictionary or None if not cached
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT * FROM articles WHERE pmid = ?", (pmid,))
            row = cursor.fetchone()

            if row:
                article = dict(row)
                # Parse JSON fields
                if article["authors"]:
                    article["authors"] = json.loads(article["authors"])
                return article

            return None
        except Exception as e:
            logger.debug(f"Cache miss or error for PMID {pmid}: {e}")
            return None
        finally:
            conn.close()

    def cache_article(self, article: Dict):
        """Cache article metadata

        Args:
            article: Article metadata dictionary
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            authors_json = json.dumps(article.get("authors", []))

            cursor.execute(
                """
                INSERT OR REPLACE INTO articles
                (pmid, title, abstract, journal, journal_iso, pub_year, authors, doi, pmc_id, citation_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    article.get("pmid"),
                    article.get("title"),
                    article.get("abstract"),
                    article.get("journal"),
                    article.get("journal_iso"),
                    article.get("pub_year"),
                    authors_json,
                    article.get("doi"),
                    article.get("pmc_id"),
                    article.get("citation_count"),
                ),
            )

            conn.commit()
        except Exception as e:
            logger.error(f"Error caching article: {e}")
        finally:
            conn.close()

    def get_search_results(self, query: str, category: str) -> Optional[List[str]]:
        """Get cached search results

        Args:
            query: Search query
            category: Search category

        Returns:
            List of PMIDs or None if not cached
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            search_id = f"{query}#{category}"
            cursor.execute(
                "SELECT pmids FROM search_results WHERE search_id = ?",
                (search_id,)
            )
            row = cursor.fetchone()

            if row:
                return json.loads(row[0])

            return None
        except Exception as e:
            logger.debug(f"Cache miss for search: {e}")
            return None
        finally:
            conn.close()

    def cache_search_results(self, query: str, category: str, pmids: List[str]):
        """Cache search results

        Args:
            query: Search query
            category: Search category
            pmids: List of PMIDs
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            search_id = f"{query}#{category}"
            pmids_json = json.dumps(pmids)

            cursor.execute(
                """
                INSERT OR REPLACE INTO search_results
                (search_id, query, category, pmids)
                VALUES (?, ?, ?, ?)
                """,
                (search_id, query, category, pmids_json),
            )

            conn.commit()
        except Exception as e:
            logger.error(f"Error caching search results: {e}")
        finally:
            conn.close()

    def get_stats(self) -> dict:
        """Get cache statistics

        Returns:
            Dictionary with article_count and search_count
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT COUNT(*) FROM articles")
            article_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM search_results")
            search_count = cursor.fetchone()[0]

            return {"article_count": article_count, "search_count": search_count}
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"article_count": 0, "search_count": 0}
        finally:
            conn.close()

    def clear_cache(self):
        """Clear all cached data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("DELETE FROM articles")
            cursor.execute("DELETE FROM search_results")
            conn.commit()
            logger.info("Cache cleared")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
        finally:
            conn.close()
