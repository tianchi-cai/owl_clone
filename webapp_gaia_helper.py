"""
Helper module for the GAIA benchmark webapp.
Contains utility functions, data loading, and log handling code.
"""

import os
import json
import time
import logging
import threading
import pathlib
import queue
from typing import Dict, List, Optional, Tuple, Any, Union

from dotenv import load_dotenv
from owl.utils import GAIABenchmark, extract_pattern
from owl.utils.enhanced_role_playing import OwlGAIARolePlaying, run_society

# Global dictionaries for storing data - key is task_id
gaia_data = {}         # Stores all GAIA benchmark data {task_id: task_dict}
results_data = {}      # Stores execution results {task_id: result_dict}
status_data = {}       # Stores current execution status {task_id: status_string}
logs_data = {}         # Stores execution logs {task_id: [log_string, ...]}

# Global constants
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "logs", "artifacts")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
GAIA_DATA_DIR = os.path.join(PROJECT_ROOT, "owl", "data", "gaia")

# Global variables for logging
LOG_QUEUE = queue.Queue()  # Log queue
STOP_LOG_THREAD = threading.Event()
TASK_LOGGERS = {}  # Dictionary to store task-specific loggers

# ===== Logging Functions ===== #

def setup_task_logging(task_id: Optional[str] = None) -> logging.Logger:
    """
    Set up logging for a specific task or use default logging if no task_id provided.
    
    Args:
        task_id: Optional task identifier. If None, uses default logging.
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    os.makedirs(LOG_DIR, exist_ok=True)
    
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

def log_task_message(task_id: Optional[str], level: str, message: str):
    """
    Logs a message related to a specific task_id.
    
    Args:
        task_id: Optional task identifier
        level: Log level (info, warning, error, etc.)
        message: Message to log
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
    Gets logs for a specific task_id as a string.
    
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

# ===== Data Loading Functions ===== #

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

def load_results(results_dir: str = RESULTS_DIR):
    """Loads previous results from task-specific files and stores them in global results_data and status_data dicts."""
    global results_data, status_data
    results_data = {}
    status_data = {}
    loaded_count = 0
    
    try:
        os.makedirs(results_dir, exist_ok=True)
        
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
                    # Update status based on loaded results
                    if result.get("score") == True:
                        status_data[task_id] = "Executed: Correct ✓"
                    elif result.get("score") == False:
                        status_data[task_id] = "Executed: Incorrect ✗"
                    else:
                        status_data[task_id] = "Executed"
            except json.JSONDecodeError:
                logging.error(f"Error decoding JSON from {latest_file}")
            except Exception as e:
                logging.error(f"Error loading result from {latest_file}: {e}")
        
        logging.info(f"Loaded {loaded_count} previously executed results into memory.")
    except Exception as e:
        logging.error(f"Error loading results: {e}", exc_info=True)

def add_or_update_result_to_file(result: Dict[str, Any], results_dir: str = RESULTS_DIR):
    """Adds or updates a result in the task-specific result file."""
    try:
        os.makedirs(results_dir, exist_ok=True)
        task_id = result.get("task_id")
        if not task_id:
            logging.error("Result does not contain task_id")
            return False
            
        # Get the result file path
        result_file = get_result_file_path(task_id, results_dir=results_dir)
        
        # Write the result to the task-specific file
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # Update the global dictionaries
        results_data[task_id] = result
        if result.get("score") == True:
            status_data[task_id] = "Executed: Correct ✓"
        elif result.get("score") == False:
            status_data[task_id] = "Executed: Incorrect ✗"
        else:
            status_data[task_id] = "Executed"
            
        return True
    except Exception as e:
        logging.error(f"Error saving result: {e}", exc_info=True)
        return False

# ===== Data Processing Functions ===== #

def get_filtered_tasks(dataset, levels, filter_query=""):
    """Helper function to get filtered task IDs based on UI selections."""
    levels_int = [int(l) for l in levels]

    filtered_task_ids = []
    for task_id, task_data in gaia_data.items():
        # Check dataset type and level
        if task_data.get("dataset_type") == dataset and task_data.get("Level") in levels_int:
            # Apply text filter if provided
            if filter_query and filter_query.strip():
                filter_terms = filter_query.lower().strip().split()
                if not any(term in task_data["Question"].lower() for term in filter_terms):
                    continue # Skip if filter doesn't match
            filtered_task_ids.append(task_id)
    
    # Sort by task ID for consistent ordering (optional)
    filtered_task_ids.sort()
    return filtered_task_ids

def process_question(question_data, user_agent_kwargs, assistant_agent_kwargs):
    """Processes a GAIA question with the given agent configurations."""
    task_id = question_data['task_id']
    
    # Set up task-specific logging
    setup_task_logging(task_id)
    
    start_time = time.time()
    log_task_message(task_id, "info", f"Started processing task: {task_id}")
    
    try:
        # Set status
        status_data[task_id] = "Processing"
        
        # Initialize benchmark with the task
        benchmark = GAIABenchmark(
            data_dir=GAIA_DATA_DIR
        )
        
        # Run the benchmark for this specific task
        result = benchmark.run(
            user_role_name="user",
            assistant_role_name="assistant",
            user_agent_kwargs=user_agent_kwargs,
            assistant_agent_kwargs=assistant_agent_kwargs,
            on=question_data["dataset_type"],  # "valid" or "test"
            level=question_data["Level"],
            task_ids=[task_id],  # Pass task_id directly
            save_result=False  # Don't save results in GAIABenchmark
        )
        
        # Process result
        execution_time = time.time() - start_time
        log_task_message(task_id, "info", f"Task completed in {execution_time:.2f} seconds")
        
        # Save result to task-specific file
        if result:
            add_or_update_result_to_file(result)
        
        # Update status based on result
        if result.get("score") is True:
            status_data[task_id] = "Executed: Correct ✓"
        elif result.get("score") is False:
            status_data[task_id] = "Executed: Incorrect ✗"
        else:
            status_data[task_id] = "Executed"
        
        return result, execution_time, None
        
    except Exception as e:
        error_msg = f"Error processing task {task_id}: {str(e)}"
        log_task_message(task_id, "error", error_msg)
        status_data[task_id] = f"Error: {str(e)}"
        return None, None, str(e)

# ===== Utility Functions ===== #

def setup_directories():
    """Creates necessary directories for logs, artifacts, and results."""
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    logging.info("Created required directories.")

def load_environment():
    """Load environment variables from .env file."""
    env_path = pathlib.Path(__file__).parent / ".env"
    load_dotenv(dotenv_path=str(env_path) if env_path.exists() else None, override=True)
    logging.info("Environment variables loaded.")


"""Initialize the application - load data and setup directories."""
setup_directories()
logging.info("GAIA Benchmark Evaluator initializing...")
load_environment()

# Initialize benchmark and load data
logging.info("Loading GAIA data...")
benchmark = GAIABenchmark(
    data_dir=GAIA_DATA_DIR
)
benchmark.load()

# Store GAIA data in global dict
global gaia_data
gaia_data = {}
for data_type in ["valid", "test"]:
    for task in benchmark._data[data_type]:
        task["dataset_type"] = data_type
        gaia_data[task["task_id"]] = task

logging.info(f"Loaded {len(gaia_data)} GAIA tasks into memory.")
logging.info("Loading previous results...")
load_results()

logging.info("Initialization complete.")