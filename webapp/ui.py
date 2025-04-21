"""
UI components and event handling for the GAIA benchmark web application.
This module handles:
- Main UI creation and layout
- Task selection interface
- Agent configuration interface
- Execution controls
- Results and logs display
- Event handlers for user interactions
"""

import gradio as gr
import logging
from typing import Dict, List, Optional

from .core import (
    process_question,
    get_filtered_tasks,
    gaia_data,
    results_data,
    status_data
)
from .data_utils import (
    get_logs_for_task,
    log_task_message
)

def create_ui():
    """
    Create the main UI for the GAIA benchmark web application.
    
    Returns:
        gr.Blocks: The Gradio interface
    """
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
                    
                    # Status filter
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
                height=400,
                wrap=True
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
                    
                    question_text = gr.TextArea(label="Question", interactive=False, lines=3)
                    
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
                            clear_logs = gr.Button("Clear Task Logs")
        
        # --- UI Helper Functions --- #
        
        def update_questions_ui(dataset_types, levels, filter_query, status_filters=None):
            """Updates the question_display DataFrame based on filters."""
            if not dataset_types:
                dataset_types = ["valid"]
                
            if not status_filters:
                status_filters = []
                
            logging.info(f"Updating questions UI with datasets={dataset_types}, levels={levels}, filter={filter_query}, statuses={status_filters}")
            
            # Get filtered tasks
            filtered_tasks = get_filtered_tasks(levels)
            
            rows = []
            for i, task in enumerate(filtered_tasks):
                task_id = task["task_id"]
                status = status_data.get(task_id, "Not Executed")
                
                # Apply status filters
                if status_filters and status not in status_filters:
                    continue
                
                # Apply text filter
                if filter_query and filter_query.strip():
                    filter_terms = filter_query.lower().strip().split()
                    if not any(term in task["Question"].lower() or term in task_id.lower() for term in filter_terms):
                        continue
                
                # Get tools used and execution time from results
                tools_used = ""
                execution_time = ""
                result = results_data.get(task_id)
                if result:
                    tools_used = result.get("tools_used", "")
                    execution_time = f"{result.get('execution_time', 0):.1f}s"
                
                rows.append([
                    i + 1,
                    task_id,
                    task["Question"],
                    task["Level"],
                    status,
                    tools_used,
                    execution_time
                ])
            
            return gr.DataFrame(value=rows)
        
        def load_question_details_ui(task_id):
            """Loads details for a specific task into the UI components."""
            if not task_id:
                return "", "", "", "", "", "", "", "", [], "", "", "", ""
            
            task = gaia_data.get(task_id)
            result = results_data.get(task_id)
            
            if not task:
                return task_id, "", "Not Found", "", "", "", "", "", [], "", "", "", ""
            
            # Format file information
            file_info = task.get("file_name", "")
            if file_info:
                file_info = f"Attached file: {file_info}"
            
            # Format chat history
            chat = []
            if result and "history" in result:
                for entry in result["history"]:
                    if "user" in entry:
                        chat.append((entry["user"], None))
                    if "assistant" in entry:
                        chat.append((None, entry["assistant"]))
            
            # Get status and score
            status = status_data.get(task_id, "Not Executed")
            score = result.get("score", "") if result else ""
            score_text = "Correct ✓" if score is True else "Incorrect ✗" if score is False else str(score)
            
            # Get logs
            task_logs = get_logs_for_task(task_id)
            
            # Get annotation metadata
            annotation_steps_text = ""
            annotation_tools_text = ""
            annotation_time_text = ""
            
            if "Annotator Metadata" in task:
                metadata = task["Annotator Metadata"]
                annotation_steps_text = metadata.get("Steps", "")
                annotation_tools_text = metadata.get("Tools", "")
                annotation_time_text = metadata.get("How long did this take?", "")
            
            return (
                task_id,
                str(task.get("Level", "")),
                status,
                task.get("Question", ""),
                result.get("model_answer", "") if result else "",
                task.get("Final answer", ""),
                score_text,
                file_info,
                chat,
                task_logs,
                annotation_steps_text,
                annotation_tools_text,
                annotation_time_text
            )
        
        def execute_question_ui(task_id):
            """Handles the button click to run a question."""
            if not task_id:
                return "No question selected.", "Waiting for question selection"
            
            task = gaia_data.get(task_id)
            if not task:
                return "Error: Question not found.", f"Error: Question not found for ID {task_id}"
            
            # Set status and clear logs
            status_data[task_id] = "Processing"
            log_task_message(task_id, "info", f"Execution requested for task: {task_id}")
            
            # Process the question
            result = process_question(task_id, task["Question"])
            
            # Update status and return logs
            if result and "error" not in result:
                status_data[task_id] = "Executed"
                return get_logs_for_task(task_id), "Question executed successfully"
            else:
                status_data[task_id] = f"Error: {result.get('error', 'Unknown error')}"
                return get_logs_for_task(task_id), f"Error executing question: {result.get('error', 'Unknown error')}"
        
        def clear_logs_ui(task_id):
            """Clears the logs for the given task."""
            if task_id in logs_data:
                logs_data[task_id] = []
                return "Logs cleared."
            return "No logs to clear for this task."
        
        # --- Connect UI components --- #
        
        # When a row is selected in the table
        def handle_select_question(evt: gr.SelectData):
            """Handles the selection event from the DataFrame."""
            try:
                # Get the row index from the event
                row_idx = evt.index[0]
                
                # Get all displayed rows
                dataset_types = dataset_type.value
                levels = level_selection.value
                filter_q = filter_text.value
                status_filters = status_filter.value
                
                # Get filtered tasks
                filtered_tasks = get_filtered_tasks(levels)
                
                # Apply the same filters as in update_questions_ui
                filtered_task_ids = []
                for task in filtered_tasks:
                    task_id = task["task_id"]
                    status = status_data.get(task_id, "Not Executed")
                    
                    # Apply status filters
                    if status_filters and status not in status_filters:
                        continue
                    
                    # Apply text filter
                    if filter_q and filter_q.strip():
                        filter_terms = filter_q.lower().strip().split()
                        if not any(term in task["Question"].lower() or term in task_id.lower() for term in filter_terms):
                            continue
                            
                    filtered_task_ids.append(task_id)
                
                # If we have a valid row index and it's within range
                if 0 <= row_idx < len(filtered_task_ids):
                    selected_task_id = filtered_task_ids[row_idx]
                    return load_question_details_ui(selected_task_id)
                    
            except Exception as e:
                logging.error(f"Error in handle_select_question: {e}", exc_info=True)
                
            return load_question_details_ui(None)  # Default to empty if error
            
        question_display.select(
            handle_select_question,
            None,  # Input is the event data
            [
                question_id, question_level, question_status, question_text,
                model_answer, ground_truth, score_display, file_display,
                chat_history, logs_display,
                annotation_steps, annotation_tools, annotation_time
            ]
        )
        
        # When the Run button is clicked
        run_btn.click(
            execute_question_ui,
            [question_id],
            [logs_display, status_output]
        )
        
        # When filter inputs change or refresh button is clicked
        refresh_inputs = [dataset_type, level_selection, filter_text, status_filter]
        refresh_btn.click(update_questions_ui, refresh_inputs, question_display)
        for inp in refresh_inputs:
            if isinstance(inp, gr.Textbox):
                inp.submit(update_questions_ui, refresh_inputs, question_display)
            else:
                inp.change(update_questions_ui, refresh_inputs, question_display)
        
        # When Clear Logs button is clicked
        clear_logs.click(clear_logs_ui, [question_id], logs_display)
    
    return app 