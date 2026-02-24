"""Test script for NeuroWriter - End-to-End Testing"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    try:
        from core.llm_client import get_llm_client
        print("[OK] llm_client imported")

        from core.pubmed_client import PubmedClient
        print("[OK] pubmed_client imported")

        from core.topic_parser import TopicParser
        print("[OK] topic_parser imported")

        from core.citation_scorer import CitationScorer
        print("[OK] citation_scorer imported")

        from core.intro_generator import IntroductionGenerator
        print("[OK] intro_generator imported")

        from core.fact_checker import FactChecker
        print("[OK] fact_checker imported")

        from utils.cache import PubmedCache
        print("[OK] cache imported")

        from utils.pubmed_utils import parse_pubmed_xml, format_citation_vancouver
        print("[OK] pubmed_utils imported")

        print("\n[PASS] All imports successful!")
        return True

    except Exception as e:
        print(f"\n[FAIL] Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pubmed_search():
    """Test PubMed search functionality"""
    print("\n\nTesting PubMed search...")
    try:
        from core.pubmed_client import PubmedClient

        client = PubmedClient()

        # Test simple search
        print("Searching for depression EEG papers...")
        results = client.search_pubmed(
            "depression AND EEG",
            "test_category",
            max_results=5
        )

        print("Found {} PMIDs: {}".format(len(results), results[:3]))

        if results:
            print("\n[PASS] PubMed search successful!")
            return True
        else:
            print("[INFO] No results found (but API call succeeded)")
            return True

    except Exception as e:
        print("[FAIL] PubMed search failed: {}".format(e))
        import traceback
        traceback.print_exc()
        return False


def test_cache():
    """Test caching functionality"""
    print("\n\nTesting cache...")
    try:
        from utils.cache import PubmedCache
        import tempfile
        import os

        # Use a temporary file instead of :memory:
        fd, temp_db = tempfile.mkstemp(suffix='.db')
        os.close(fd)

        try:
            cache = PubmedCache(temp_db)

            # Test article caching
            test_article = {
                "pmid": "12345678",
                "title": "Test Article",
                "abstract": "Test abstract",
                "journal": "Test Journal",
                "pub_year": "2024",
                "authors": ["Smith J", "Doe J"],
                "doi": "10.1234/test",
                "pmc_id": "PMC123456"
            }

            cache.cache_article(test_article)
            retrieved = cache.get_article("12345678")

            if retrieved and retrieved["pmid"] == "12345678":
                print("[PASS] Cache test successful!")
                return True
            else:
                print("[FAIL] Cache retrieval failed - retrieved: {}".format(retrieved))
                return False
        finally:
            if os.path.exists(temp_db):
                os.remove(temp_db)

    except Exception as e:
        print("[FAIL] Cache test failed: {}".format(e))
        import traceback
        traceback.print_exc()
        return False


def test_citation_scorer():
    """Test citation scoring"""
    print("\n\nTesting citation scorer...")
    try:
        from core.citation_scorer import CitationScorer

        scorer = CitationScorer()

        test_article = {
            "pmid": "12345678",
            "title": "Test Article",
            "abstract": "[BACKGROUND] Depression is a major disorder. [METHODS] We used EEG. [RESULTS] Accuracy was 95% (p<0.001, n=200). [CONCLUSIONS] Deep learning works.",
            "journal": "Nature",
            "pub_year": "2024",
            "authors": ["Smith J", "Doe J"]
        }

        score = scorer.score_article(test_article, relevance_score=0.8)

        print("Article score: {}/100".format(score))

        if 0 <= score <= 100:
            print("[PASS] Citation scorer test successful!")
            return True
        else:
            print("[FAIL] Citation score out of range")
            return False

    except Exception as e:
        print("[FAIL] Citation scorer test failed: {}".format(e))
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("NeuroWriter - End-to-End Testing")
    print("=" * 60)

    results = []

    results.append(("Imports", test_imports()))
    results.append(("PubMed Search", test_pubmed_search()))
    results.append(("Cache", test_cache()))
    results.append(("Citation Scorer", test_citation_scorer()))

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print("{}: {}".format(test_name, status))

    all_passed = all(passed for _, passed in results)

    if all_passed:
        print("\n[SUCCESS] All tests passed!")
    else:
        print("\n[WARNING] Some tests failed. Please review the errors above.")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
