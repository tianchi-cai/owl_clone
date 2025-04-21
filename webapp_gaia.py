"""
GAIA Benchmark Web Application - Main UI File
"""

import os
import json
import gradio as gr
import threading
import time
import logging

# Configure logging for the main application
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(name)s - %(message)s'
)

# Import helper module for data loading and processing
from owl.webapp_gaia_helper import (
    # Import global data stores
    gaia_data, results_data, status_data, logs_data, thread_local, stop_event,
    
    # Import data functions
    get_filtered_tasks,
    
    # Import logging functions
    log_task_message, get_logs_for_task,
    
    # Import processing functions
    process_question
)

from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.configs import ChatGPTConfig

from owl.utils.enhanced_role_playing import OwlGAIARolePlaying, run_society

# Import required toolkits
from camel.toolkits import (
    AudioAnalysisToolkit,
    CodeExecutionToolkit,
    ExcelToolkit,
    ImageAnalysisToolkit,
    SearchToolkit,
    VideoAnalysisToolkit,
    BrowserToolkit,
    FileWriteToolkit,
    HumanToolkit,
)

from owl.utils import DocumentProcessingToolkit

# Create UI
def create_ui():
    # Setup models and tools
    logging.info("Setting up models and tools...")
    models = {
        "user": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O,
            model_config_dict=ChatGPTConfig(temperature=0, top_p=1).as_dict(),
        ),
        "assistant": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O,
            model_config_dict=ChatGPTConfig(temperature=0, top_p=1).as_dict(),
        ),
        "browsing": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O,
            model_config_dict=ChatGPTConfig(temperature=0, top_p=1).as_dict(),
        ),
        "planning": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O,
            model_config_dict=ChatGPTConfig(temperature=0, top_p=1).as_dict(),
        ),
        "video": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O,
            model_config_dict=ChatGPTConfig(temperature=0, top_p=1).as_dict(),
        ),
        "image": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O,
            model_config_dict=ChatGPTConfig(temperature=0, top_p=1).as_dict(),
        ),
        "document": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O,
            model_config_dict=ChatGPTConfig(temperature=0, top_p=1).as_dict(),
        ),
    }
    
    # Configure tools
    logging.info("Configuring tools...")
        
    tools = [
        *BrowserToolkit(
            headless=False,
            web_agent_model=models["browsing"],
            planning_agent_model=models["planning"],
        ).get_tools(),
        *VideoAnalysisToolkit(model=models["video"]).get_tools(),
        *AudioAnalysisToolkit().get_tools(),
        *CodeExecutionToolkit(sandbox="subprocess", verbose=True).get_tools(),
        *ImageAnalysisToolkit(model=models["image"]).get_tools(),
        SearchToolkit().search_duckduckgo,
        SearchToolkit().search_wiki,
        *ExcelToolkit().get_tools(),
        *DocumentProcessingToolkit(model=models["document"]).get_tools(),
        *FileWriteToolkit(output_dir="./").get_tools(),
        HumanToolkit().ask_human_via_console,
    ]
    
    # Configure agent roles
    user_agent_kwargs = {"model": models["user"]}
    assistant_agent_kwargs = {"model": models["assistant"], "tools": tools}
    
    # Data is already loaded into global dicts by main(), no need to load here
    logging.info(f"Using pre-loaded GAIA data ({len(gaia_data)} tasks) and results ({len(results_data)} tasks).")
    
    # Create dataset selections
    logging.info("Creating UI components...")
    with gr.Blocks(title="GAIA Benchmark Evaluator") as app:
        gr.Markdown("# GAIA Benchmark Evaluator")
        
        # SECTION 1: DATASET SELECTION AT THE TOP
        with gr.Box():
            gr.Markdown("## Dataset Selection")
            
            with gr.Row():
                # Left controls
                with gr.Column(scale=1):
                    dataset_type = gr.CheckboxGroup(
                        choices=["valid", "test"], 
                        label="Dataset Type",
                        value=["valid"],
                    )
                    
                    level_selection = gr.CheckboxGroup(
                        choices=["1", "2", "3"],
                        label="Level Selection",
                        value=["1"]
                    )
                # Right controls
                with gr.Column(scale=1):
                    # Filter options
                    filter_text = gr.Textbox(
                        label="Search (Question or Task ID)",
                        placeholder="Search by text or task ID...",
                        value=""
                    )
                    
                    # Change status filter to checkbox group
                    status_filter = gr.CheckboxGroup(
                        choices=["Not Executed", "Correct", "Incorrect", "Error"],
                        label="Status Filter",
                        value=[]
                    )
                    
                    refresh_btn = gr.Button("Refresh Questions", variant="secondary")
            
            # Scrollable questions display
            question_display = gr.DataFrame(
                headers=["Index", "Task ID", "Question", "Level", "Status", "Tools Used", "Execution Time"],
                label="Available Questions",
                interactive=True,
                height=400,  # Increased height
                wrap=True # Enable text wrapping
                # value removed, will be populated by initial_load
            )
        
        # SECTION 2: LOWER SPLIT - QUESTION ANALYSIS AND CONVERSATION HISTORY
        with gr.Row():
            # SECTION 2.1: QUESTION ANALYSIS ON THE LOWER LEFT
            with gr.Column(scale=1):
                with gr.Box():
                    gr.Markdown("## Question Analysis")
                    
                    with gr.Row():
                        question_id = gr.Textbox(label="Task ID", interactive=False)
                        question_level = gr.Textbox(label="Level", interactive=False)
                        question_status = gr.Textbox(label="Status", interactive=False)
                    
                    question_text = gr.TextArea(label="Question", interactive=False, lines=3) # Increased lines
                    
                    with gr.Row():
                        model_answer = gr.Textbox(label="Model Answer", interactive=False)
                        ground_truth = gr.Textbox(label="Ground Truth", interactive=False)
                    
                    score_display = gr.Textbox(label="Score", interactive=False)
                    
                    file_display = gr.Textbox(label="Attached Files", interactive=False)
                    
                    # Add tabs for annotations
                    with gr.Tabs():
                        with gr.TabItem("Annotations"):
                            annotation_steps = gr.TextArea(label="Human Steps", interactive=False, lines=5)
                            with gr.Row():
                                annotation_tools = gr.Textbox(label="Tools Used", interactive=False)
                                annotation_time = gr.Textbox(label="Time Taken", interactive=False)
                    
                    status_output = gr.Textbox(label="Execution Status", value="Ready", interactive=False)
                    
                    run_btn = gr.Button("Run This Question", variant="primary")
            
            # SECTION 2.2: CONVERSATION HISTORY AND LOGS ON THE LOWER RIGHT
            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.TabItem("Conversation History"):
                        chat_history = gr.Chatbot(height=600, label="Conversation")
                        
                    with gr.TabItem("System Logs"):
                        logs_display = gr.TextArea(label="System Logs", interactive=False, lines=25)
                        
                        with gr.Row():
                            # Removed auto-refresh checkbox for logs
                            clear_logs = gr.Button("Clear Task Logs")
        
        # --- UI Helper Functions --- #

        # Function to update question list
        def update_questions_ui(dataset_types, levels, filter_query, status_filters=None):
            """Updates the question_display DataFrame based on filters."""
            if not dataset_types:
                dataset_types = ["valid"]  # Default to valid if nothing selected
                
            if not status_filters:
                status_filters = []  # Empty means show all
                
            logging.info(f"Updating questions UI with datasets={dataset_types}, levels={levels}, filter={filter_query}, statuses={status_filters}")
            
            # Get all task IDs that match dataset and level filters
            all_filtered_task_ids = []
            for dataset in dataset_types:
                task_ids = get_filtered_tasks(dataset, levels, "")  # Get all tasks for this dataset+levels
                all_filtered_task_ids.extend(task_ids)
            
            # Debug output
            print(f"update_questions_ui: found {len(all_filtered_task_ids)} filtered tasks")
            if not all_filtered_task_ids:
                print(f"WARNING: No tasks match the filters: datasets={dataset_types}, levels={levels}")
                # Return a placeholder DataFrame with "No results" message
                return gr.DataFrame(
                    value=[["", "No matching tasks", "Try different filters", "", "", "", ""]],
                    headers=["Index", "Task ID", "Question", "Level", "Status", "Tools Used", "Execution Time"]
                )
            
            rows = []
            for i, task_id in enumerate(all_filtered_task_ids):
                data = gaia_data.get(task_id)
                if not data: continue
                
                status = status_data.get(task_id, "Not Executed")

                # Apply status filters if any are selected
                if status_filters:  # Only apply filter if there are status filters selected
                    if not any(_filter in status for _filter in status_filters):
                        continue  # Skip this task if it doesn't match any selected status
                
                question_text = data["Question"]
                
                # Apply text filter - now check both question and task_id
                if filter_query and filter_query.strip():
                    filter_terms = filter_query.lower().strip().split()
                    # Check if any term matches in either question text or task_id
                    match_found = False
                    for term in filter_terms:
                        if term in question_text.lower() or term in task_id.lower():
                            match_found = True
                            break
                    if not match_found:
                        continue
                
                # Get tools used from annotator metadata
                tools_used = ""
                if "Annotator Metadata" in data:
                    tools_used = data["Annotator Metadata"].get("Tools", "").replace("\n", ", ")
                    if len(tools_used) > 50:
                        tools_used = tools_used[:50] + "..."
                
                # Get execution time from results if available
                execution_time = ""
                result = results_data.get(task_id)
                if result and "execution_time" in result:
                    execution_time = f"{result['execution_time']:.1f}s"

                rows.append([
                    i + 1,  # Index starting from 1
                    task_id,
                    question_text,
                    data["Level"],
                    status,
                    tools_used,
                    execution_time
                ])
            
            logging.info(f"Displaying {len(rows)} questions.")
            print(f"Created DataFrame with {len(rows)} rows")
            
            if not rows:
                # Provide a placeholder for empty results
                return gr.DataFrame(
                    value=[["", "No tasks found", "", "", "", "", ""]],
                    headers=["Index", "Task ID", "Question", "Level", "Status", "Tools Used", "Execution Time"]
                )
                
            # Verify data is non-empty before returning
            if any(row[1] for row in rows):  # Check if any task IDs are non-empty
                df = gr.DataFrame(value=rows, headers=["Index", "Task ID", "Question", "Level", "Status", "Tools Used", "Execution Time"])
                print(f"Returning DataFrame with {len(rows)} rows containing data")
                return df
            else:
                print("WARNING: Generated rows but all task IDs are empty!")
                return gr.DataFrame(
                    value=[["", "Data error", "Please refresh", "", "", "", ""]],
                    headers=["Index", "Task ID", "Question", "Level", "Status", "Tools Used", "Execution Time"]
                )

        # Function to load question details
        def load_question_details_ui(task_id_to_load):
            """Loads details for a specific task_id into the UI components."""
            if not task_id_to_load:
                 # Return 13 empty values (added 3 for annotations)
                 return "", "", "", "", "", "", "", "", [], "", "", "", ""
                 
            logging.info(f"Loading details for task ID: {task_id_to_load}")
            question_data = gaia_data.get(task_id_to_load)
            result_data = results_data.get(task_id_to_load)
            
            if not question_data:
                logging.warning(f"Task ID {task_id_to_load} not found in gaia_data")
                # Return 13 values, including task_id and empty logs
                return task_id_to_load, "", "Not Found", "", "", "", "", "", [], "", "", "", ""

            # Format file information
            file_info = question_data.get("file_name", "")
            if file_info:
                file_info = f"Attached file: {file_info}"

            # Format chat history
            chat = []
            if result_data and "history" in result_data:
                for entry in result_data["history"]:
                    user_msg = entry.get("user")
                    assistant_msg_content = entry.get("assistant")
                    tool_calls = entry.get("tool_calls")
                    
                    if user_msg:
                        chat.append((user_msg, None))
                    if assistant_msg_content:
                        full_assistant_msg = assistant_msg_content
                        if tool_calls and len(tool_calls) > 0:
                            tool_call_text = "\n\n**Tool Calls:**"
                            for tc in tool_calls:
                                func_name = tc.get('func_name') or tc.get('function', {}).get('name', 'unknown')
                                tool_call_text += f"\n- **{func_name}**"
                                tc_result = tc.get("result", "")
                                if isinstance(tc_result, dict):
                                     tc_result = f"```json\n{json.dumps(tc_result, indent=2)}\n```"
                                elif isinstance(tc_result, str) and len(tc_result) > 1000: # Basic truncation for logs
                                    tc_result = tc_result[:1000] + "... [truncated]"
                                tool_call_text += f"\n  *Result:* {tc_result}"
                            full_assistant_msg += tool_call_text
                        chat.append((None, full_assistant_msg))
            
            # Status and score
            status = status_data.get(task_id_to_load, "Not Executed")
            score = result_data.get("score", "") if result_data else ""
            score_text = "Correct ✓" if score is True else "Incorrect ✗" if score is False else str(score)
            
            # Load logs for this task
            task_logs = get_logs_for_task(task_id_to_load)
            
            # Get annotation metadata if available
            annotation_steps_text = ""
            annotation_tools_text = ""
            annotation_time_text = ""
            
            if "Annotator Metadata" in question_data:
                metadata = question_data["Annotator Metadata"]
                annotation_steps_text = metadata.get("Steps", "")
                annotation_tools_text = metadata.get("Tools", "")
                annotation_time_text = metadata.get("How long did this take?", "")

            # Return values for all output components
            return (
                task_id_to_load,
                str(question_data.get("Level", "")),
                status,
                question_data.get("Question", ""),
                result_data.get("model_answer", "") if result_data else "",
                question_data.get("Final answer", ""),
                score_text,
                file_info,
                chat,
                task_logs,
                annotation_steps_text,
                annotation_tools_text,
                annotation_time_text
            )

        # Function to execute a GAIA question
        def execute_question_ui(task_id_to_run):
            """Handles the button click to run a question."""
            if not task_id_to_run:
                return "No question selected.", "Waiting for question selection"
            
            question_data = gaia_data.get(task_id_to_run)
            if not question_data:
                error_msg = f"Question data not found for task_id: {task_id_to_run}"
                logging.error(error_msg)
                return "Error: Question data not found.", f"Error: Question not found for ID {task_id_to_run}"
            
            # Set trace_id and update status
            status_data[task_id_to_run] = "Processing"
            logs_data[task_id_to_run] = [] # Clear previous logs for this run
            log_task_message(task_id_to_run, "info", f"Execution requested for task: {task_id_to_run}")
            
            # Start execution in a separate thread
            execution_thread = threading.Thread(
                target=process_question_in_thread,
                args=(question_data, user_agent_kwargs, assistant_agent_kwargs),
                daemon=True,
                name=f"Exec-{task_id_to_run[:8]}" # Give thread a name
            )
            execution_thread.start()
            
            # Update UI immediately
            initial_logs = get_logs_for_task(task_id_to_run)
            return initial_logs, f"Processing question {task_id_to_run}..."
            
        def process_question_in_thread(question_data, user_agent_kwargs, assistant_agent_kwargs):
            """Thread wrapper around process_question."""
            task_id = question_data['task_id']
            try:
                # Process the question using the function from helper module
                process_question(question_data, user_agent_kwargs, assistant_agent_kwargs)
            except Exception as e:
                error_message = f"Error in thread execution: {str(e)}"
                log_task_message(task_id, "error", error_message)
                logging.error(f"Error in execution thread for {task_id}:", exc_info=True)
                status_data[task_id] = f"Error: {error_message[:100]}..." # Truncate long errors
            
        # Function to update logs display using global dictionary
        def update_logs_ui(current_task_id):
            """Gets logs for the currently selected task_id."""
            logs = get_logs_for_task(current_task_id, max_lines=1000)
            status = status_data.get(current_task_id, "")
            return f"Status: {status}\n\n{logs}"
        
        # Function to clear logs for the selected task
        def clear_logs_ui(task_id_to_clear):
            """Clears the logs in the global dictionary for the given task_id."""
            if task_id_to_clear in logs_data:
                logs_data[task_id_to_clear] = [f"{time.strftime('%Y-%m-%d %H:%M:%S')} - INFO - Logs cleared."]
                logging.info(f"Cleared logs for task: {task_id_to_clear}")
                return logs_data[task_id_to_clear][0] # Return confirmation message
            return "No logs to clear for this task."

        # Function to handle question selection from the DataFrame
        def handle_select_question(evt: gr.SelectData):
            """Handles the selection event from the DataFrame."""
            selected_task_id = None
            try:
                # Get the row index from the event
                row_idx = evt.index[0]
                col_idx = evt.index[1] if len(evt.index) > 1 else 0
                
                print(f"Selected row={row_idx}, col={col_idx}, value={evt.value}, index={evt.index}")
                
                # Get all displayed rows directly from the update_questions_ui function
                # Reuse the function to get the same data that's shown in the table
                dataset_types = dataset_type.value
                levels = level_selection.value
                filter_q = filter_text.value
                status_filters = status_filter.value
                
                # Get filtered task IDs that match what's shown in the table
                all_filtered_task_ids = []
                for dataset in dataset_types:
                    task_ids = get_filtered_tasks(dataset, levels, "")
                    all_filtered_task_ids.extend(task_ids)
                
                # Apply the same filters that would be applied in update_questions_ui
                filtered_task_ids = []
                for task_id in all_filtered_task_ids:
                    data = gaia_data.get(task_id)
                    if not data:
                        continue
                        
                    status = status_data.get(task_id, "Not Executed")
                    
                    # Apply status filters if any are selected
                    if status_filters:
                        status_match = False
                        for status_option in status_filters:  # Renamed from status_filter to status_option
                            if status_option in status:
                                status_match = True
                                break
                        if not status_match:
                            continue
                    
                    # Apply text filter
                    if filter_q and filter_q.strip():
                        filter_terms = filter_q.lower().strip().split()
                        match_found = False
                        for term in filter_terms:
                            if (term in data["Question"].lower() or 
                                term in task_id.lower()):
                                match_found = True
                                break
                        if not match_found:
                            continue
                    
                    filtered_task_ids.append(task_id)
                
                # If we have a valid row index and it's within range of our filtered tasks
                if 0 <= row_idx < len(filtered_task_ids):
                    # Get the task ID directly from our filtered list
                    selected_task_id = filtered_task_ids[row_idx]
                    print(f"Successfully extracted task ID: {selected_task_id}")
                else:
                    # Fallback: Try to extract directly from the selected value
                    # In some cases, the selected value might be the task ID itself
                    if isinstance(evt.value, str) and evt.value in gaia_data:
                        selected_task_id = evt.value
                        print(f"Using selected value as task ID: {selected_task_id}")
                    else:
                        print(f"Row index {row_idx} out of range. Filtered tasks: {len(filtered_task_ids)}")
            
            except Exception as e:
                logging.error(f"Error in handle_select_question: {e}", exc_info=True)
                print(f"Exception when selecting question: {str(e)}")
                selected_task_id = None

            # Debug what task ID we're sending to load details
            print(f"Loading details for task ID: {selected_task_id}")
            
            # Call the function to update the UI based on the extracted task_id
            return load_question_details_ui(selected_task_id)

        # --- Connect UI components --- #

        # When a row is selected in the table
        question_display.select(
            handle_select_question, 
            None, # Input is the event data
            [ # List of output components to update
                question_id, question_level, question_status, question_text,
                model_answer, ground_truth, score_display, file_display, 
                chat_history, logs_display, # Update logs too
                annotation_steps, annotation_tools, annotation_time # Add annotation fields
            ],
            show_progress="hidden" # Hide default progress bar for selection
        )
        
        # When the Run button is clicked
        run_btn.click(
            execute_question_ui, 
            [question_id], 
            [logs_display, status_output] # Update logs and status text
        )
        
        # When filter inputs change or refresh button is clicked
        refresh_inputs = [dataset_type, level_selection, filter_text, status_filter]
        refresh_btn.click(update_questions_ui, refresh_inputs, question_display)
        for inp in refresh_inputs:
             # Use submit for Textbox to trigger on Enter key
            if isinstance(inp, gr.Textbox):
                inp.submit(update_questions_ui, refresh_inputs, question_display)
            else:
                inp.change(update_questions_ui, refresh_inputs, question_display)
        
        # When Clear Logs button is clicked
        clear_logs.click(clear_logs_ui, [question_id], logs_display)
       
        # Initial loading function
        def initial_load_ui():
            """Performs the initial population of the questions table."""
            logging.info("Performing initial UI load...")
            
            # Debug: Check data availability
            print(f"gaia_data contains {len(gaia_data)} items")
            print(f"Available datasets: {set(task.get('dataset_type', 'unknown') for task in gaia_data.values())}")
            print(f"Available levels: {set(task.get('Level', 'unknown') for task in gaia_data.values())}")
            
            # Check if we have any valid data items
            valid_levels = [1]
            valid_items = [task_id for task_id, task in gaia_data.items() 
                          if task.get('dataset_type') == 'valid' and task.get('Level') in valid_levels]
            
            print(f"Found {len(valid_items)} items with dataset='valid' and level in {valid_levels}")
            
            if not valid_items:
                # If no items found, create a placeholder row to avoid empty DataFrame
                print("WARNING: No matching items found, will create placeholder data")
                df = gr.DataFrame(
                    value=[["No data", "Please check filters", "", "", "", "", ""]],
                    headers=["Index", "Task ID", "Question", "Level", "Status", "Tools Used", "Execution Time"]
                )
                return df
            
            # Call update_questions_ui with default values
            return update_questions_ui(dataset_types=["valid"], levels=["1"], filter_query="", status_filters=[])
        
        # Trigger initial load
        app.load(initial_load_ui, None, question_display)
    
    return app

# Main function
def main():
    try:
        # Initialize the application using the helper module
        logging.info("len gaia data: %d, len results data: %d", len(gaia_data), len(results_data))
        logging.info("Creating UI...")
        # Create and launch the UI
        app = create_ui()
        app.queue().launch(share=False)
        
    except Exception as e:
        logging.error(f"Critical error during application startup or runtime: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure background threads stop
        stop_event.set()
        logging.info("Application closed.")

if __name__ == "__main__":
    main()
