"""
Utility functions and constants for the GAIA benchmark web application.

This module provides path constants and initialization functions.
Initialization functions (setup_directories and load_environment) should be called
once during application startup, typically in core.py.
"""

import os
import pathlib
import logging
from dotenv import load_dotenv

# ===== Path Constants ===== #

# Project root directory (two levels up from this file)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directory for storing logs
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

# Directory for storing artifacts (images, documents, etc.)
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "logs", "artifacts")

# Directory for storing results
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Directory containing GAIA benchmark data
GAIA_DATA_DIR = os.path.join(PROJECT_ROOT, "owl", "data", "gaia")

# ===== Utility Functions ===== #

def setup_directories():
    """
    Creates necessary directories for logs, artifacts, and results.
    This function should be called once during application startup.
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    logging.info("Created required directories.")

def load_environment():
    """
    Load environment variables from .env file.
    This function should be called once during application startup.
    """
    env_path = pathlib.Path(__file__).parent / ".env"
    load_dotenv(dotenv_path=str(env_path) if env_path.exists() else None, override=True)
    logging.info("Environment variables loaded.")

def ensure_directory_exists(path: str) -> None:
    """
    Ensures that the specified directory exists, creating it if necessary.
    
    Args:
        path: Directory path to check/create
    """
    os.makedirs(path, exist_ok=True) 