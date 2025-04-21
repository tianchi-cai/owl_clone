"""
GAIA Benchmark Web Application

This package provides a web interface for running and evaluating GAIA benchmark tasks.
"""

import logging

# Configure logging for the application
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(name)s - %(message)s'
)

# Import submodules to make them available
from .core import (
    initialize,
    process_question,
    get_filtered_tasks,
    gaia_data,
    results_data,
    status_data,
    logs_data
)

from .data_utils import (
    setup_task_logging,
    log_task_message,
    get_logs_for_task,
    load_results,
    add_or_update_result_to_file
)

from .utils import (
    setup_directories,
    load_environment,
    LOG_DIR,
    RESULTS_DIR,
    GAIA_DATA_DIR,
    PROJECT_ROOT
)

from .ui import create_ui

# Main application function
def main():
    """
    Initialize and launch the GAIA benchmark web application.
    """
    try:
        # Initialize the application
        initialize()
        
        # Create and launch the UI
        app = create_ui()
        app.queue().launch(share=False)
        
    except Exception as e:
        logging.error(f"Critical error during application startup: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        logging.info("Application closed.")

# Export public API
__all__ = [
    'initialize',
    'main',
    'create_ui',
    'process_question',
    'get_filtered_tasks',
    'setup_task_logging',
    'log_task_message',
    'get_logs_for_task',
    'setup_directories',
    'load_environment'
] 