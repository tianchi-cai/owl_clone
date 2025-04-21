"""
Core functionality for the GAIA benchmark web application.
This module handles core data management and task processing.
"""

import os
import time
import logging
from typing import Dict, List, Optional

from .utils import (
    setup_directories,
    load_environment,
    LOG_DIR,
    RESULTS_DIR,
    GAIA_DATA_DIR
)
from .data_utils import (
    load_gaia_metadata,
    load_results,
    setup_task_logging,
    log_task_message,
    get_logs_for_task,
    add_or_update_result_to_file
)

# ===== Global Data ===== #

# Core data loaded from GAIA benchmark
gaia_data = None

# Results data loaded from result files
results_data = {}

# Status data for tracking task states
status_data = {}

# Logs data for storing task-specific logs
logs_data = {}

# ===== Initialization ===== #

def initialize():
    """
    Initialize the core module.
    This should be called once at application startup.
    """
    # Setup environment and directories
    load_environment()
    setup_directories()
    
    # Load core data
    global gaia_data
    gaia_data = load_gaia_metadata(GAIA_DATA_DIR)
    
    # Load existing results
    global results_data
    results_data = load_results()
    
    logging.info("Core module initialized")

def load_gaia_data():
    """Load GAIA benchmark data into memory."""
    global gaia_data
    
    logging.info("Loading GAIA data...")
    benchmark = GAIABenchmark(data_dir=GAIA_DATA_DIR)
    benchmark.load()
    
    # Store GAIA data in global dict
    for data_type in ["valid", "test"]:
        for task in benchmark._data[data_type]:
            task["dataset_type"] = data_type
            gaia_data[task["task_id"]] = task
    
    logging.info(f"Loaded {len(gaia_data)} GAIA tasks into memory.")

# ===== Core Functions ===== #

def process_question(question_data: Dict[str, Any], 
                     user_agent_kwargs: Dict[str, Any], 
                     assistant_agent_kwargs: Dict[str, Any]) -> Tuple[Dict, float, Optional[str]]:
    """
    Process a GAIA question with the given agent configurations.
    
    Args:
        question_data: Dictionary containing task data
        user_agent_kwargs: Arguments for the user agent
        assistant_agent_kwargs: Arguments for the assistant agent
        
    Returns:
        Tuple of (result_dict, execution_time, error_message)
    """
    task_id = question_data['task_id']
    
    # Set up task-specific logging
    setup_task_logging(task_id)
    
    start_time = time.time()
    log_task_message(task_id, f"Started processing task: {task_id}", "info")
    
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
        log_task_message(task_id, f"Task completed in {execution_time:.2f} seconds", "info")
        
        # Add execution time to result
        if result:
            result["execution_time"] = execution_time
            
            # Save result to task-specific file
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
        log_task_message(task_id, error_msg, "error")
        status_data[task_id] = f"Error: {str(e)}"
        return None, None, str(e)

def get_filtered_tasks(levels: Optional[List[str]] = None, dataset_type: str = "valid") -> List[Dict]:
    """
    Get filtered list of tasks based on level and dataset type.
    
    Args:
        levels: Optional list of levels to filter by
        dataset_type: Dataset type to filter by
        
    Returns:
        List of task dictionaries
    """
    filtered_task_ids = []
    
    # Convert to integers for comparison
    levels_int = [int(l) for l in levels] if levels else []
    
    for task_id, task_data in gaia_data.items():
        # Check dataset type and level
        if task_data.get("dataset_type") == dataset_type and (
            not levels_int or task_data.get("Level") in levels_int):
            filtered_task_ids.append(task_id)
    
    # Sort by task ID for consistent ordering
    filtered_task_ids.sort()
    
    # Convert task IDs to task dictionaries
    return [gaia_data[task_id] for task_id in filtered_task_ids] 