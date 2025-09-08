# run_optimization.py

"""
Generalized Prompt Optimization Workflow Runner.

This script serves as the main entry point for executing the prompt optimization process.
It handles:
1. Argument parsing (configuration, dataset, tracing options).
2. Dynamic creation of a unique results directory for each run.
3. Configuration of centralized logging.
4. Loading and validation of the configuration file and input dataset.
5. Initialization of the optimization state.
6. Compilation and execution of the LangGraph optimization workflow.
7. Processing and saving the final results and detailed execution traces.
"""

import logging
import uuid
import json
import argparse
import sys
import os
from datetime import datetime
from typing import List, Dict, Any

# NOTE: Pandas/Numpy imports are removed as analysis is now integrated into the workflow (nodes.py).

# -----------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------

# Initialize the logger variable here for module-level access.
# The actual configuration (handlers, formatters) will be set up in main() to allow dynamic file paths.
logger = logging.getLogger(__name__)

def configure_logging(run_dir: str):
    """
    Configures centralized logging for the optimization run.

    Sets up two handlers:
    1. Console handler: Outputs INFO level logs to standard output.
    2. Execution file handler: Writes detailed INFO/DEBUG logs to 'execution.log'
       within the specified run directory, overwriting previous logs (mode='w').

    Args:
        run_dir (str): The directory path where the log file will be created.
    """
    
    # Ensure the directory exists (defensive check)
    os.makedirs(run_dir, exist_ok=True)

    # Define the standard log format
    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    # 1. Console Handler (for standard output visibility)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)

    # 2. Execution Log File Handler (captures detailed execution flow)
    execution_log_path = os.path.join(run_dir, 'execution.log')
    # Use mode='w' to ensure a fresh log for the specific run
    execution_file_handler = logging.FileHandler(execution_log_path, encoding='utf-8', mode='w')
    # Set level to INFO. Can be lowered to DEBUG if more verbosity is needed.
    execution_file_handler.setLevel(logging.INFO) 
    execution_file_handler.setFormatter(log_format)

    # Configure the root logger
    root_logger = logging.getLogger()
    # Set the lowest level we want to capture globally
    root_logger.setLevel(logging.INFO) 

    # Clear existing handlers to prevent duplicate logs if the script is run multiple times in the same process (e.g., in tests)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    root_logger.addHandler(console_handler)
    root_logger.addHandler(execution_file_handler)

    # Use the module logger to confirm setup
    logger.info(f"--- Logging Initialized ---")
    logger.info(f"Execution log: {execution_log_path}")


# Import necessary components
try:
    # Import the callback handler for Langfuse observability
    from callbacks import get_langfuse_handler
except ImportError:
    # We use print here as logger might not be fully configured yet during initial imports
    print("WARNING: Could not import get_langfuse_handler from callbacks.py. Tracing might be disabled.")
    get_langfuse_handler = None

try:
    from prompt_optimizer.workflow import compile_optimizer_graph
    # Import core data models for state management and type hinting.
    from prompt_optimizer.models import OptimizationState, InputExample
    # Ensure nodes are imported so the LLM manager initializes (if required by environment setup)
    from prompt_optimizer.nodes import generate_prompts
except ImportError as e:
    print(f"Error importing prompt_optimizer modules: {e}")
    print("Please ensure the project structure is correct and the necessary packages are installed.")
    sys.exit(1)


def load_json_file(filepath: str) -> Any:
    """
    Loads a JSON file with robust error handling.

    Handles FileNotFoundError and JSONDecodeError, logging the issue and exiting
    the script if the file cannot be loaded, as configuration/data is essential.

    Args:
        filepath (str): The path to the JSON file.

    Returns:
        Any: The loaded JSON data (typically a Dict or List).
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        # Use logger if available, otherwise fall back to print
        msg = f"Error: File not found at {filepath}"
        if logger.hasHandlers():
            logger.error(msg)
        else:
            print(msg)
        sys.exit(1)
    except json.JSONDecodeError:
        msg = f"Error: Could not decode JSON from {filepath}"
        if logger.hasHandlers():
            logger.error(msg)
        else:
            print(msg)
        sys.exit(1)


def format_dataset(raw_dataset: List[Dict[str, Any]]) -> List[InputExample]:
    """
    Formats the raw dataset dictionary into the required InputExample structure.

    The workflow requires data to be structured as a list of InputExample objects, 
    each having a unique ID and a data payload (the variables for the prompt template).
    This function ensures the input data conforms to this structure.

    Args:
        raw_dataset (List[Dict[str, Any]]): The raw data loaded from the input JSON file.

    Returns:
        List[InputExample]: A list of formatted InputExample objects.
    """
    input_examples = []

    if not isinstance(raw_dataset, list):
        logger.error("Error: Dataset JSON must be a list of objects.")
        sys.exit(1)

    for item in raw_dataset:
        if not isinstance(item, dict):
            logger.warning(f"Skipping invalid item in dataset (not a dictionary): {item}")
            continue

        # We need a unique ID for tracking; use the provided 'id' or generate one
        example_id = str(item.get("id", str(uuid.uuid4())[:8]))

        # The 'data' field contains the actual inputs that will be formatted into the prompt template.
        # We exclude the 'id' key from the data payload itself, as it's metadata.
        data_payload = {k: v for k, v in item.items() if k != 'id'}

        if not data_payload:
            logger.warning(f"Skipping empty data item with id {example_id}")
            continue

        input_examples.append(InputExample(id=example_id, data=data_payload))

    if not input_examples:
        logger.error("Error: The dataset is empty or contains no valid examples after formatting.")
        sys.exit(1)

    return input_examples


# -----------------------------------------------------------------------
# Post-Hoc Analysis Helpers (REMOVED)
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------

def main(config_path: str, dataset_path: str, run_dir: str, enable_tracing: bool):
    """
    Executes the main optimization workflow.

    Orchestrates the entire process: initialization, data loading, state setup, 
    LangGraph execution, and result reporting.

    Args:
        config_path (str): Path to the optimization configuration JSON file.
        dataset_path (str): Path to the input dataset JSON file.
        run_dir (str): The specific directory for this run's outputs (logs, results).
        enable_tracing (bool): Flag to enable Langfuse observability tracing.
    """
    
    # 1. Initialize Logging using the dynamically determined run directory
    configure_logging(run_dir)

    # Define specific output file paths within the run directory
    results_output_path = os.path.join(run_dir, "results.json")
    # Use JSONL format for the streaming trace of prompt generation attempts
    prompt_trace_path = os.path.join(run_dir, "prompt_generation_trace.jsonl")

    logger.info(f"Starting optimization process.")
    logger.info(f"Loading configuration from: {config_path}")
    logger.info(f"Loading dataset from: {dataset_path}")
    # Log the path where prompt generation attempts will be streamed
    logger.info(f"Prompt generation trace (JSONL) will be streamed to: {prompt_trace_path}")

    # 2. Load Configuration and Data
    config = load_json_file(config_path)
    raw_dataset = load_json_file(dataset_path)

    # 3. Format Dataset
    input_dataset = format_dataset(raw_dataset)
    logger.info(f"Successfully formatted {len(input_dataset)} input examples.")

    # Extract parameters from config
    task_description = config.get("task_description")
    opt_params = config.get("optimization_parameters", {})
    model_config = config.get("model_configuration", {})
    # Extract early stopping parameters (if provided in config under "early_stopping" key)
    es_config = config.get("early_stopping", {})

    if not task_description or not opt_params or not model_config:
        logger.error("Error: Configuration file is missing required sections (task_description, optimization_parameters, or model_configuration).")
        sys.exit(1)

    # 4. Initialize Optimization State
    # This dictionary defines the initial state of the LangGraph application.
    initial_state = OptimizationState(
        # Core Configuration
        max_iterations=opt_params.get("max_iterations", 10),
        N_CANDIDATES=opt_params.get("N_CANDIDATES", 5),
        MINI_BATCH_SIZE=opt_params.get("MINI_BATCH_SIZE", 2),
        target_task_description=task_description,
        input_dataset=input_dataset,

        # Early Stopping Configuration
        es_min_iterations=es_config.get("min_iterations", 3),
        es_patience=es_config.get("patience", 2),
        es_threshold_percentage=es_config.get("threshold_percentage", 5.0),

        # Model Configuration
        optimizer_model=model_config.get("optimizer_model"),
        actor_model=model_config.get("actor_model"),
        voter_ensemble=model_config.get("voter_ensemble", []),

        # Initial state variables (boilerplate initialization for the workflow)
        current_iteration=0,
        current_candidates={},
        current_mini_batch=[],
        current_execution_results=[],
        current_votes=[],
        history=[],
        all_tested_prompts={},
        best_result=None,
        synthesized_critiques="",
        iteration_best_score_history=[],
        global_best_example=None,
        global_worst_example=None,
        
        # Initialize the detailed trace history which captures every execution and evaluation
        execution_trace_history=[],

        # Initialize the path for streaming prompt generation traces
        prompt_trace_file_path=prompt_trace_path,
    )

    # Validate models are specified
    if not initial_state['optimizer_model'] or not initial_state['actor_model'] or not initial_state['voter_ensemble']:
        logger.error("Error: Model configuration is incomplete in the config file.")
        sys.exit(1)

    # 5. Compile and Run the Graph
    logger.info("Compiling optimizer graph...")
    app = compile_optimizer_graph()

    # Configure Callbacks for Observability (Langfuse)
    # Initialize the LangGraph configuration with required settings
    runnable_config = {"recursion_limit": 150}

    if enable_tracing:
        if get_langfuse_handler:
            try:
                langfuse_handler = get_langfuse_handler()
                # LangGraph (like LangChain) expects callbacks under the "callbacks" key
                runnable_config["callbacks"] = [langfuse_handler]
                logger.info("Langfuse tracing enabled.")
            except Exception as e:
                # Catch potential initialization errors (e.g., missing API keys in .env)
                logger.warning(f"Could not initialize Langfuse handler. Tracing will be disabled. Error: {e}")
        else:
            logger.warning("Tracing requested but Langfuse handler could not be imported.")

    logger.info("Running optimization workflow...")
    try:
        # Using invoke to execute the graph and get the final state. Pass the configuration.
        final_state = app.invoke(initial_state, config=runnable_config)

        logger.info("\n--- Optimization Finished ---")

        # Extract results from the final state dictionary
        best_result = final_state.get('best_result')
        final_history = final_state.get('history', [])
        global_best_example = final_state.get('global_best_example')
        global_worst_example = final_state.get('global_worst_example')
        # Extract the detailed execution history
        execution_trace_history = final_state.get('execution_trace_history', [])

        if best_result:

            # Display Summary in the console
            print(f"\nTask Name: {config.get('task_name', 'N/A')}")
            print(f"Iterations Completed: {final_state.get('current_iteration', 'N/A')}")

            # Display clarified performance metrics
            print(f"\n--- Performance Metrics ---")
            raw_score = best_result.get('raw_average_score', 0.0)
            agg_score = best_result.get('aggregate_score', 0.0)

            print(f"Absolute Quality Score (Avg 1-10): {raw_score:.2f}")
            print(f"  (Average score given by voters. Best indicator of real-world quality.)")
            
            print(f"\nRelative Performance Score (Z-avg): {agg_score:.4f}")
            print(f"  (Normalized score used for optimization. Indicates performance relative to other candidates.)")
            print("-----------------------------\n")

            print("\n--- Best Prompt Template ---")
            print(best_result.get('prompt_text', '[Missing]'))
            print("----------------------------\n")

            # Display Global Best Example Snippet for interpretability
            if global_best_example:
                print("\n--- Global Best Execution Example (Highest Consensus Score Observed) ---")
                # Show both raw and normalized scores
                print(f"Consensus Raw Score (1-10): {global_best_example.get('consensus_raw_score', 0.0):.2f}")
                print(f"Consensus Normalized Score: {global_best_example.get('consensus_normalized_score', 0.0):.4f}")
                print(f"Candidate ID: {global_best_example.get('candidate_id')}")
                print(f"Executed Prompt Snippet: {global_best_example.get('executed_prompt_text', '')[:500]}...")
                print(f"\nOutput Snippet: {global_best_example.get('output', '')[:500]}...")
                print("---------------------------------------------------------------\n")

            # Save results to the JSON output file
            # The structure is designed for comprehensive traceability and analysis.
            results_output = {
                "task_name": config.get('task_name'),
                "configuration": config,
                "best_prompt_result": best_result,
                "summary_traceability": { 
                    "overall_best_execution": global_best_example,
                    "overall_worst_execution": global_worst_example,
                    "iteration_best_score_history": final_state.get('iteration_best_score_history', []),
                },
                "prompt_history": final_history, # History of prompt evolution (iterations)
                # Detailed history of every single execution and evaluation
                "execution_details": execution_trace_history,
            }
            with open(results_output_path, "w", encoding='utf-8') as f:
                # ensure_ascii=False is crucial for saving non-English characters correctly
                json.dump(results_output, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {results_output_path}")
            logger.info(f"Total detailed execution traces captured: {len(execution_trace_history)}")

        else:
            logger.warning("Optimization finished without finding a best result. Check logs for errors during the run.")

    except Exception as e:
        logger.error(f"An unexpected error occurred during execution: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Generalized Prompt Optimization Workflow.")
    parser.add_argument("--config", type=str, default="optimization_config.json", help="Path to the optimization configuration JSON file.")
    parser.add_argument("--dataset", type=str, default="input_dataset.json", help="Path to the input dataset JSON file.")
    # The generic --output argument is removed; we now enforce a structured directory system.
    parser.add_argument("--trace", action="store_true", help="Enable Langfuse tracing for observability.")
    # Optional argument to specify the run name manually instead of using the auto-generated name.
    parser.add_argument("--run_name", type=str, help="Optional name for the run directory. If not specified, generated from config and timestamp.")

    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Dynamic Directory Creation
    # -----------------------------------------------------------------------
    # This section ensures that every execution has a unique, organized directory
    # for storing logs, results, and intermediate traces.
    
    # 1. Determine the base directory for results
    results_base_dir = "optimization_results"
    # Ensure base directory exists
    os.makedirs(results_base_dir, exist_ok=True)

    # 2. Determine the specific run directory name
    if args.run_name:
        run_dir_name = args.run_name
    else:
        # Auto-generate the name based on the task name in the config file and the current timestamp.
        # Load config preliminarily to get the task name.
        # We must use basic loading here as the logger is not yet configured with the file path.
        task_name = 'optimization' # Default name if config fails to load
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                config_for_name = json.load(f)
            task_name = config_for_name.get('task_name', 'optimization')
        except Exception as e:
            print(f"Warning: Could not load config {args.config} for directory naming: {e}. Using default name '{task_name}'.")


        # Create timestamp and sanitize the task name for use in a file path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_task_name = "".join(c for c in task_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_task_name = safe_task_name.replace(' ', '_')
        run_dir_name = f"{safe_task_name}_{timestamp}"

    # 3. Create the run directory
    run_dir_path = os.path.join(results_base_dir, run_dir_name)

    try:
        # Ensure the directory is new for this run (exist_ok=False)
        os.makedirs(run_dir_path, exist_ok=False)
    except FileExistsError:
        # Handle if the directory already exists (e.g., user specified a name that exists, or rapid execution caused timestamp collision)
        print(f"Error: Run directory already exists: {run_dir_path}")
        # If it was auto-generated, try adding a unique suffix
        if not args.run_name:
            run_dir_path = os.path.join(results_base_dir, f"{run_dir_name}_{str(uuid.uuid4())[:4]}")
            print(f"Attempting to create new directory with suffix: {run_dir_path}")
            os.makedirs(run_dir_path)
        else:
            # If the user manually specified the name and it exists, require them to resolve the conflict.
            print("Please specify a unique --run_name or delete the existing directory.")
            sys.exit(1)
    except OSError as e:
        print(f"Error: Could not create results directory {run_dir_path}. {e}")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Execution
    # -----------------------------------------------------------------------

    # Execute the main function. Logging initialization occurs inside main() using the validated run_dir_path.
    main(args.config, args.dataset, run_dir_path, args.trace)