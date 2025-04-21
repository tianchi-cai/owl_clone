# GAIA Benchmark Web Application

A web application for running and evaluating GAIA benchmark tasks with customizable agent configurations.

## Project Structure

```
owl/webapp/
├── __init__.py
├── core.py          # Core functionality and data management
├── data_utils.py       # Logging and file operations
├── ui.py           # UI components and event handling
└── utils.py        # Utility functions and constants
```

## Module Descriptions

### core.py
Core functionality and data management:
- Task processing and execution
- Model and agent creation
- Data management (tasks, results, status, logs)

### data_utils.py
Logging and file operations:
- Task-specific logging setup
- Log message handling
- Result file management
- File operations for logs and results

### ui.py
UI components and event handling:
- Main UI creation and layout
- Task selection interface
- Agent configuration interface
- Execution controls
- Results and logs display
- Event handlers for user interactions

### utils.py
Utility functions and constants:
- Directory setup
- Environment variable loading
- Path constants
- Helper functions

## Data Flow

1. User interacts with UI components
2. Events trigger core functionality
3. Core processes tasks and updates data
4. Logging handles file operations
5. UI updates to reflect changes

## Global Data

- `gaia_data`: Stores all GAIA benchmark tasks
- `results_data`: Stores execution results
- `status_data`: Stores task execution status
- `logs_data`: Stores execution logs

## File Organization

- Logs: `{task_id}_{timestamp}.log`
- Results: `{task_id}_{timestamp}.json`
- Artifacts: Stored in artifacts directory

## Usage

1. Initialize the application:
```python
from owl.webapp import core, data_utils, ui, utils

# Setup directories and load environment
utils.setup_directories()
utils.load_environment()

# Load GAIA data and results
core.load_gaia_data()
data_utils.load_results()

# Create and launch UI
ui.create_ui().launch()
```

2. Run tasks:
- Select a task from the task list
- Configure agent parameters
- Click "Execute" to run the task
- View results and logs in the UI

## Dependencies

- gradio
- python-dotenv
- owl (core library) 