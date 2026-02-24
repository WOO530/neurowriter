"""Utility functions for PubMed data parsing"""
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def parse_pubmed_xml(xml_string: str) -> List[Dict]:
    """Parse PubMed XML response to extract article information

    Args:
        xml_string: XML response from PubMed E-utilities API

    Returns:
        List of dictionaries containing article metadata
    """
    articles = []

    try:
        root = ET.fromstring(xml_string)

        # Handle both PubmedArticle and different XML structures
        for article_elem in root.findall(".//PubmedArticle"):
            article_data = _extract_article_data(article_elem)
            if article_data:
                articles.append(article_data)

        return articles
    except ET.ParseError as e:
        logger.error(f"Error parsing PubMed XML: {e}")
        return []


def _extract_article_data(article_elem: ET.Element) -> Optional[Dict]:
    """Extract article metadata from PubmedArticle element

    Args:
        article_elem: XML element containing article data

    Returns:
        Dictionary with article metadata or None if extraction fails
    """
    try:
        # Extract PMID
        pmid_elem = article_elem.find(".//PMID")
        if pmid_elem is None:
            return None
        pmid = pmid_elem.text

        # Extract Article metadata
        article_meta = article_elem.find(".//Article")
        if article_meta is None:
            return None

        # Title
        title_elem = article_meta.find(".//ArticleTitle")
        title = title_elem.text if title_elem is not None else ""

        # Abstract — preserve section labels (BACKGROUND, METHODS, RESULTS, CONCLUSIONS)
        abstract_elem = article_meta.find(".//Abstract")
        abstract = ""
        if abstract_elem is not None:
            abstract_texts = []
            for text_elem in abstract_elem.findall(".//AbstractText"):
                if text_elem.text:
                    label = text_elem.get("Label", "")
                    if label:
                        abstract_texts.append(f"[{label.upper()}] {text_elem.text}")
                    else:
                        abstract_texts.append(text_elem.text)
            abstract = " ".join(abstract_texts)

        # Journal info
        journal_elem = article_meta.find(".//Journal")
        journal_title = ""
        journal_iso = ""
        if journal_elem is not None:
            title_elem = journal_elem.find("Title")
            journal_title = title_elem.text if title_elem is not None else ""
            iso_elem = journal_elem.find("ISOAbbreviation")
            journal_iso = iso_elem.text if iso_elem is not None else ""

        # Publication date
        pub_date_elem = article_meta.find(".//PubDate")
        pub_year = ""
        if pub_date_elem is not None:
            year_elem = pub_date_elem.find("Year")
            pub_year = year_elem.text if year_elem is not None else ""

        # Authors
        authors = []
        author_list = article_meta.find(".//AuthorList")
        if author_list is not None:
            for author_elem in author_list.findall("Author"):
                last_name_elem = author_elem.find("LastName")
                first_name_elem = author_elem.find("ForeName")
                initials_elem = author_elem.find("Initials")

                last_name = last_name_elem.text if last_name_elem is not None else ""
                first_name = first_name_elem.text if first_name_elem is not None else ""
                initials = initials_elem.text if initials_elem is not None else ""

                if last_name:
                    author_str = f"{last_name} {initials}" if initials else last_name
                    authors.append(author_str)

        # DOI
        doi = ""
        for article_id in article_meta.findall(".//ArticleId"):
            if article_id.get("IdType") == "doi":
                doi = article_id.text if article_id.text else ""
                break

        # PMC ID
        pmc_id = ""
        for article_id in article_meta.findall(".//ArticleId"):
            if article_id.get("IdType") == "pmc":
                pmc_id = article_id.text if article_id.text else ""
                break

        # Publication types & retraction detection
        publication_types = []
        is_retracted = False
        pub_type_list = article_elem.find(".//PublicationTypeList")
        if pub_type_list is not None:
            for pt_elem in pub_type_list.findall("PublicationType"):
                if pt_elem.text:
                    publication_types.append(pt_elem.text)
                    if "retract" in pt_elem.text.lower():
                        is_retracted = True

        return {
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "journal": journal_title,
            "journal_iso": journal_iso,
            "pub_year": pub_year,
            "authors": authors,
            "doi": doi,
            "pmc_id": pmc_id,
            "publication_types": publication_types,
            "is_retracted": is_retracted,
        }
    except Exception as e:
        logger.error(f"Error extracting article data: {e}")
        return None


def has_valid_abstract(article: Dict, min_length: int = 100) -> bool:
    """Check if article has a meaningful abstract

    Args:
        article: Article metadata dictionary
        min_length: Minimum character length for a valid abstract

    Returns:
        True if abstract exists and meets minimum length
    """
    abstract = article.get("abstract", "")
    return bool(abstract) and len(abstract.strip()) >= min_length


def format_citation_vancouver(article: Dict, citation_number: int) -> str:
    """Format article citation in Vancouver style

    Args:
        article: Article metadata dictionary
        citation_number: Citation number for this article

    Returns:
        Formatted citation string
    """
    authors = article.get("authors", [])
    title = article.get("title", "")
    journal = article.get("journal", "")
    pub_year = article.get("pub_year", "")
    pmid = article.get("pmid", "")

    # Format authors (first 3, then et al.)
    if len(authors) > 3:
        author_str = ", ".join(authors[:3]) + " et al."
    else:
        author_str = ", ".join(authors)

    # Build citation
    citation = f"{citation_number}. {author_str}. {title}. {journal}. {pub_year}. PMID: {pmid}."

    return citation
