"""Configuration module for NeuroWriter"""
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# PubMed Configuration
PUBMED_BATCH_SIZE = 20
PUBMED_DELAY = 0.5  # seconds between requests (API recommends 1/3 per second)
PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

# Database Configuration
DB_PATH = os.getenv("DB_PATH", "neurowriter_cache.db")

# Application Configuration
APP_TITLE = "NeuroWriter"
APP_DESCRIPTION = "EEG/Deep Learning 의학논문 Introduction Generator"
MAX_INTRO_LENGTH = 1200  # words
MIN_INTRO_LENGTH = 800  # words
REFERENCE_VANCOUVER_FORMAT = True

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
