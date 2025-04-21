"""
Data utility functions for the GAIA benchmark web application.
This module provides utilities for:
- Loading and managing GAIA metadata
- Storing and retrieving task results
- Task-specific logging
- Data persistence
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Any, Union

from .utils import (
    LOG_DIR,
    RESULTS_DIR,
    GAIA_DATA_DIR,
    ensure_directory_exists
)

# Global dictionary to store task-specific logs
logs_data = {}

# Dictionary to store task-specific loggers
TASK_LOGGERS = {}

# ===== Logging Functions ===== #

def setup_task_logging(task_id: Optional[str] = None) -> logging.Logger:
    """
    Set up logging for a specific task.
    
    Args:
        task_id: Optional task ID. If not provided, uses default logging.
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    ensure_directory_exists(LOG_DIR)
    
    # Generate timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Use task_id with timestamp for log filename or default
    if task_id:
        log_file = os.path.join(LOG_DIR, f"{task_id}_{timestamp}.log")
    else:
        log_file = os.path.join(LOG_DIR, f"default_{timestamp}.log")
    
    # Create logger
    logger_name = f"task_{task_id}" if task_id else "default"
    logger = logging.getLogger(logger_name)
    
    # Clear existing handlers to avoid duplicate logs
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    logger.setLevel(logging.INFO)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8", mode="w")
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Store logger reference
    if task_id:
        TASK_LOGGERS[task_id] = logger
    
    logger.info(f"Logging system initialized for {'task ' + task_id if task_id else 'default'}, log file: {log_file}")
    return logger

def log_task_message(task_id: Optional[str], message: str, level: str = "info") -> None:
    """
    Log a message for a specific task.
    
    Args:
        task_id: Task ID
        message: Message to log
        level: Log level (info, warning, error)
    """
    # Get or create logger for the task
    logger = TASK_LOGGERS.get(task_id) if task_id else logging.getLogger("default")
    if not logger:
        logger = setup_task_logging(task_id)
    
    # Log the message
    log_method = getattr(logger, level.lower(), logger.info)
    log_method(message)
    
    # Also store in logs_data for UI display
    log_entry = f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {level.upper()} - {message}"
    if task_id not in logs_data:
        logs_data[task_id] = []
    logs_data[task_id].append(log_entry)

def get_logs_for_task(task_id: Optional[str], max_lines: int = 200) -> str:
    """
    Get logs for a specific task as a string.
    
    Args:
        task_id: Optional task identifier
        max_lines: Maximum number of log lines to return
        
    Returns:
        str: Formatted log string
    """
    if not task_id:
        return "No task selected."
        
    if task_id not in logs_data:
        logs_data[task_id] = []  # Initialize empty list if no logs yet
        
    # Join log entries with newlines, limit to max_lines
    log_entries = logs_data[task_id][-max_lines:] if len(logs_data[task_id]) > max_lines else logs_data[task_id]
    return "\n".join(log_entries)

# ===== Result Management Functions ===== #

def get_result_file_path(task_id: str, timestamp: Optional[str] = None, results_dir: str = RESULTS_DIR) -> str:
    """
    Constructs the path for a task's result file.
    
    Args:
        task_id: Task identifier
        timestamp: Optional timestamp. If None, current timestamp will be used
        results_dir: Directory to store results
        
    Returns:
        str: Full path to the result file
    """
    if timestamp is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(results_dir, f"{task_id}_{timestamp}.json")

def load_results(results_dir: str = RESULTS_DIR) -> Dict[str, Any]:
    """
    Load previous results from task-specific files.
    
    Args:
        results_dir: Directory containing result files
        
    Returns:
        Dict mapping task IDs to their results
    """
    results_data = {}
    loaded_count = 0
    
    try:
        ensure_directory_exists(results_dir)
        
        # Group files by task_id
        task_files = {}
        for filename in os.listdir(results_dir):
            if filename.endswith('.json'):
                # Extract task_id from filename (format: task_id_timestamp.json)
                parts = filename[:-5].split('_')  # Remove .json and split by underscore
                if len(parts) < 2:  # Skip files without timestamp
                    continue
                task_id = '_'.join(parts[:-1])  # Rejoin task_id parts if it contains underscores
                timestamp = parts[-1]
                
                if task_id not in task_files:
                    task_files[task_id] = []
                task_files[task_id].append((filename, timestamp))
        
        # For each task, load the most recent result
        for task_id, files in task_files.items():
            # Sort by timestamp (most recent first)
            files.sort(key=lambda x: x[1], reverse=True)
            latest_file = files[0][0]  # Get the filename of the most recent result
            
            try:
                with open(os.path.join(results_dir, latest_file), "r", encoding="utf-8") as f:
                    result = json.load(f)
                    results_data[task_id] = result
                    loaded_count += 1
            except json.JSONDecodeError:
                logging.error(f"Error decoding JSON from {latest_file}")
            except Exception as e:
                logging.error(f"Error loading result from {latest_file}: {e}")
        
        logging.info(f"Loaded {loaded_count} previously executed results.")
        return results_data
        
    except Exception as e:
        logging.error(f"Error loading results: {e}", exc_info=True)
        return {}

def add_or_update_result_to_file(result: Dict[str, Any], results_dir: str = RESULTS_DIR) -> bool:
    """
    Add or update a result in its task-specific file.
    
    Args:
        result: Dictionary containing result data
        results_dir: Directory to store results
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        ensure_directory_exists(results_dir)
        task_id = result.get("task_id")
        if not task_id:
            logging.error("Result does not contain task_id")
            return False
            
        # Get the result file path
        result_file = get_result_file_path(task_id, results_dir=results_dir)
        
        # Write the result to the task-specific file
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            
        return True
        
    except Exception as e:
        logging.error(f"Error saving result: {e}", exc_info=True)
        return False

# ===== Data Loading Functions ===== #

def load_gaia_metadata(data_dir: str = GAIA_DATA_DIR) -> Dict[str, Any]:
    """
    Load GAIA benchmark metadata.
    
    Args:
        data_dir: Directory containing GAIA data
        
    Returns:
        Dict mapping task IDs to task data
    """
    try:
        from owl.utils import GAIABenchmark
        
        # Initialize benchmark and load data
        benchmark = GAIABenchmark(data_dir=data_dir)
        benchmark.load()
        
        # Create dict mapping task_id to task data
        gaia_data = {}
        for data_type in ["valid", "test"]:
            for task in benchmark._data[data_type]:
                task["dataset_type"] = data_type
                gaia_data[task["task_id"]] = task
                
        logging.info(f"Loaded {len(gaia_data)} GAIA tasks.")
        return gaia_data
        
    except Exception as e:
        logging.error(f"Error loading GAIA metadata: {str(e)}")
        return {} 